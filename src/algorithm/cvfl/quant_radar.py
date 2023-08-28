"""
Train VFL on ModelNet-10 dataset
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import datetime, pytz
import argparse
import numpy as np
import time
import random
import pickle
import math

from tqdm import tqdm
# from models.resnet2 import *
from models.resnet_top import *
from models.vertibench_cls_bottom import *
from models.vertibench_cls_top import *

import sys

import latbin

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

DATASET_SPLIT = [0, 0, 0, 0] # added by Junyi

def topk(tensor, compress_ratio):
    """
    Get topk elements in tensor
    """
    shape = tensor.shape
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(
        tensor.abs(),
        k,
        sorted=False,
    )
    values = torch.gather(tensor, 0, indices)
    numel = tensor.numel()
    tensor_decompressed = torch.zeros(
        numel, dtype=values.dtype, layout=values.layout, device=values.device
    )
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed.view(shape)



def quantize_vector(x, quant_min=0, quant_max=1, quant_level=5, dim=2):
    """Uniform vector quantization approach

    Notebook: C2S2_DigitalSignalQuantization.ipynb

    Args:
        x: Original signal
        quant_min: Minimum quantization level
        quant_max: Maximum quantization level
        quant_level: Number of quantization levels
        dim: dimension of vectors to quantize

    Returns:
        x_quant: Quantized signal

        Currently only works for 2 dimensions and
        quant_levels of 4, 8, and 16.
    """

    dither = np.random.uniform(
        -(quant_max - quant_min) / (2 * (quant_level - 1)),
        (quant_max - quant_min) / (2 * (quant_level - 1)),
        size=np.array(x).shape,
    )
    # Move into 0,1 range:
    x_normalize = x / np.max(x)
    x_normalize = x_normalize + dither

    A2 = latbin.lattice.ALattice(dim, scale=1 / (2 * math.log(quant_level, 2)))
    if quant_level == 4:
        A2 = latbin.lattice.ALattice(dim, scale=1 / 4)
    elif quant_level == 8:
        A2 = latbin.lattice.ALattice(dim, scale=1 / 8.5)
    elif quant_level == 16:
        A2 = latbin.lattice.ALattice(dim, scale=1 / 19)

    for i in range(0, x_normalize.shape[1], dim):
        x_normalize[:, i : (i + dim)] = A2.lattice_to_data_space(
            A2.quantize(x_normalize[:, i : (i + dim)])
        )

    # Move out of 0,1 range:
    x_normalize = np.max(x) * (x_normalize - dither)
    return torch.from_numpy(x_normalize).float().cuda(device)


def quantize_scalar(x, quant_min=0, quant_max=1, quant_level=5):
    """Uniform quantization approach

    Notebook: C2S2_DigitalSignalQuantization.ipynb

    Args:
        x: Original signal
        quant_min: Minimum quantization level
        quant_max: Maximum quantization level
        quant_level: Number of quantization levels

    Returns:
        x_quant: Quantized signal
    """
    x_normalize = np.array(x)

    # Move into 0,1 range:
    x_normalize = x_normalize / np.max(x)
    x_normalize = np.nan_to_num(x_normalize)

    dither = np.random.uniform(
        -(quant_max - quant_min) / (2 * (quant_level - 1)),
        (quant_max - quant_min) / (2 * (quant_level - 1)),
        size=x_normalize.shape,
    )
    x_normalize = x_normalize + dither

    x_normalize = (
        (x_normalize - quant_min) * (quant_level - 1) / (quant_max - quant_min)
    )
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)
    x_quant = (x_normalize_quant) * (quant_max - quant_min) / (
        quant_level - 1
    ) + quant_min

    # Move out of 0,1 range:
    x_quant = np.max(x) * (x_quant - dither)
    return torch.from_numpy(x_quant).float().cuda(device)



def save_eval(
    models, train_loader, test_loader, losses, accs_train, accs_test, step, train_size
):
    """
    Evaluate and save current loss and accuracy
    """
    avg_train_acc, avg_loss = eval(models, train_loader)
    avg_test_acc, _ = eval(models, test_loader)

    losses.append(avg_loss)
    accs_train.append(avg_train_acc)
    accs_test.append(avg_test_acc)

    # 禁止保存，加快速度
    #pickle.dump(losses, open(f"./{args.folder}/loss{suffix}.pkl", "wb"))
    #pickle.dump(accs_train, open(f"./{args.folder}/accs_train{suffix}.pkl", "wb"))
    #pickle.dump(accs_test, open(f"./{args.folder}/accs_test{suffix}.pkl", "wb"))

    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), 
        "Iter [%d/%d]: Avg Test Acc: %.2f - Avg Train Acc: %.2f - "
        % (
            step + 1,
            train_size,
            avg_test_acc.item(),
            avg_train_acc.item(),
        ),
        end=''
    )

    if args.dataset == "msd":
        print("Avg RMSE: %.4f" % (np.sqrt(avg_loss.item())))
    else:
        print("Avg Loss: %.4f" % (avg_loss.item()))


def train(models, optimizers, epoch):  # , centers):
    """
    Train all clients on all batches
    """
    global server_model_comp, server_optimizer_comp

    train_size = len(train_loader)
    server_model = models[-1]
    server_optimizer = optimizers[-1]

    Hs = np.empty((len(train_loader), num_clients), dtype=object) # (500, 4)
    Hs.fill([])
    grads_Hs = np.empty((num_clients), dtype=object)
    grads_Hs.fill([])

    ratio = 0
    comp = args.comp
    if args.quant_level > 0:
        ratio = math.log(args.quant_level, 2) / 32

    total_c2s = 0
    total_s2c = 0

    # step=0, inputs=(100,3,32,32), targets=(100)
    for step, (inputs, targets) in enumerate(train_loader):
        # Convert from list of 3D to 4D
        inputs = np.stack(inputs, axis=1) # 或许我们不需要把 (100,174) 转换成 (174,100)？（174是feature数量）
        inputs = torch.from_numpy(inputs)

        inputs, targets = inputs.cuda(device), targets.cuda(device)
        inputs, targets = Variable(inputs), Variable(targets)
        
        # 交换 embeddings，H_orig 是存放在 server 的 embeddings
        H_orig = [None] * num_clients # [None, None, None, None]
        
        for i in range(num_clients): # 取出 H_orig[i], 压缩后放入 H_orig[i]
            x_local = inputs[
                sum(DATASET_SPLIT[:i]) : sum(DATASET_SPLIT[: (i + 1)]), :
            ]
            x_local = torch.transpose(x_local, 0, 1) # 切分数据集
            with torch.no_grad(): # client[i] 跑出来结果放入 H_orig[i]
                H_orig[i] = models[i](x_local) # float32, shape=(512,100)

            # Compress embedding(取出 H_orig[i], 压缩后放入 H_orig[i])
            H_orig_need_to_transfer = [None] * num_clients # This variable was added by Junyi
            if comp != "":
                if comp == "topk" and not (epoch == 0 and step == 0):
                    # Choose top k elements based on grads_Hs[i]
                    # H_orig[i] = topk(H_orig[i], ratio)
                    H_tmp = H_orig[i].cpu().detach().numpy()
                    num = math.ceil(H_tmp.shape[1] * (1 - ratio)) # num 个 要被置为 0
                    grads = np.abs(grads_Hs[i])
                    idx = np.argpartition(grads, num)[:num]
                    indices = idx[np.argsort((grads)[idx])]
                    H_tmp[:, indices[:num]] = 0 # 后面传输的时候，只传输非0的
                    H_orig_need_to_transfer[i] = H_tmp.shape[1] - num
                    H_orig[i] = torch.from_numpy(H_tmp).float().cuda(device) # 更新本地的 H_orig[i]
                elif comp == "topk":
                    # If first iteration, do nothing
                    pass
                elif args.vecdim == 1:
                    # Scalar quantization
                    H_orig[i] = quantize_scalar(
                        H_orig[i].cpu().detach().numpy(), quant_level=args.quant_level
                    )
                else:
                    # Vector quantization
                    H_orig[i] = quantize_vector(
                        H_orig[i].cpu().detach().numpy(),
                        quant_level=args.quant_level,
                        dim=args.vecdim,
                    )
        
        total_number1 = np.sum([h.flatten().shape for h in H_orig])
        total_c2s += total_number1
        
        # 压缩 Server model （减少后续传输server model的communication开销，Train clients要用到 server model）
        k_numbers_need_to_transfer = {} # This variable was added by Junyi
        if comp != "":
            tmp_dict = server_model.state_dict()
            for key, value in tmp_dict.items():
                vdim = value.dim()
                shape = value.shape
                if comp == "topk":
                    tmp_dict[key], k_numbers_need_to_transfer[key] = topk(value, ratio)
                elif args.vecdim == 1:
                    if vdim == 1:
                        value = value.reshape(1, -1)
                    tmp_dict[key] = quantize_scalar(
                        value.cpu().detach().numpy(), quant_level=args.quant_level
                    ).reshape(shape)
                else:
                    if vdim == 1:
                        value = value.reshape(1, -1)
                    tmp_dict[key] = quantize_vector(
                        value.cpu().detach().numpy(),
                        quant_level=args.quant_level,
                        dim=args.vecdim,
                    ).reshape(shape)
            server_model_comp.load_state_dict(tmp_dict)
        else:
            server_model_comp = server_model

        # ========= 下面的 Train clients 通信统计 =========
        # 理论情况需要传输的 float32 数量
        total_number2 = np.sum([k_numbers_need_to_transfer[k] for k in k_numbers_need_to_transfer.keys()]) * num_clients
        total_s2c += total_number2 # total float32 numbers from server to client
        
        total_number3 = np.sum([h.flatten().shape for h in H_orig[1:]]) * num_clients # 不统计自己的 H_orig[i]，因为自己已经算过了。又因为 H_orig[i] 的shape都一样，所以随便扔掉一个就行
        total_s2c += total_number3
        # ========= 下面的 Train clients 通信统计 =========   

        # Train clients
        for i in range(num_clients):
            x_local = inputs[
                sum(DATASET_SPLIT[:i]) : sum(DATASET_SPLIT[: (i + 1)]), :
            ] # 切分属于 client[i] 的数据集
            x_local = torch.transpose(x_local, 0, 1)
            H = H_orig.copy() # client[i] 拿到 所有 压缩后的 Embedding（其实只需要传输 num_clients - 1 个 H_orig[]，因为自己有 H_orig[i] 而且 server 没有更新 H_orig）
            
            # Train client epoch
            model = models[i]
            optimizer = optimizers[i]
            # Calculate number of local iterations
            client_epochs = args.local_epochs
            # Train
            for le in range(client_epochs):
                # compute output
                outputs = model(x_local)
                H[i] = outputs
                outputs = server_model_comp(torch.cat(H, axis=1)) # 用 server 传输过来的 server_model_comp 来计算
                loss = criterion(outputs, targets)

                # compute gradient and do gradient step
                optimizer.zero_grad()
                server_optimizer_comp.zero_grad()
                loss.backward(retain_graph=True)
                params = []
                for param in model.parameters():
                    params.append(param.grad)
                params[-1] = params[-1].detach().cpu().numpy()
                grads_Hs[i] = np.array(params[-1]) # 保存 grad_Hs[i] 在本地，topk压缩的时候直接从本地拿
                optimizer.step()

                

        # Train server
        for le in range(args.local_epochs):
            H = H_orig.copy() # H_orig 属于 server，这步是本地拷贝，不需要统计
            # compute output
            outputs = server_model(torch.cat(H, axis=1))
            loss = criterion(outputs, targets)

            # compute gradient and do SGD step
            server_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            server_optimizer.step()

        if (step + 1) % args.print_freq == 0:
            print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), 
                "\tServer Iter [%d/%d] " % (step + 1, train_size),
                end=''
            )
            print(f"c -> s {num_clients} 个 client 压缩后的 H_orig[i] 更新完毕，发送给 server, total {total_number1} float32 numbers ({round(total_number1*8/1024/1024, 4)} MBytes)")
            print(f"s -> c 发送压缩后的 server model 给 {num_clients} 个 client, total {total_number2} float32 numbers ({round(total_number2*8/1024/1024, 4)} MBytes)")
            print(f"s -> c Server 发送 其他 client 的 H_orig 给 client[1 ~ {num_clients}], total {total_number3} float32 numbers ({round(total_number3*8/1024/1024, 4)} MBytes)")
            # if dataset is msd, calculate RMSE
            if args.dataset == "msd":
                print("RMSE: %.4f" % (np.sqrt(loss.item())))
            else:
                print("Loss: %.4f" % (loss.item()))

# Validation and Testing
def eval(models, data_loader):
    """
    Calculate loss and accuracy for a given data_loader
    """
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)
            # print("eval inputs shape", inputs.shape)
            # Get current embeddings
            H_new = [None] * num_clients
            for i in range(num_clients):
                x_local = inputs[
                    sum(DATASET_SPLIT[:i]) : sum(DATASET_SPLIT[: (i + 1)]), :
                ]
                # print(f"eval x_local shape {x_local.shape}, DATASET_SPLIT={DATASET_SPLIT}")
                x_local = torch.transpose(x_local, 0, 1) # 又给转回去了
                H_new[i] = models[i](x_local)
            # compute output
            outputs = models[-1](torch.cat(H_new, axis=1))
            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss


if __name__ == "__main__":
    # MVCNN = "mvcnn"
    # RESNET = "resnet"
    # MODELS = [RESNET, MVCNN]

    # Set up input arguments
    

    parser = argparse.ArgumentParser(description="MVCNN-PyTorch")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--num_clients",
        type=int,
        help="Number of clients to split data between vertically",
        default=2,
    )
    # parser.add_argument(
    #     "--depth",
    #     choices=[18, 34, 50, 101, 152],
    #     type=int,
    #     metavar="N",
    #     default=18,
    #     help="resnet depth (default: resnet18)",
    # )
    # parser.add_argument(
    #     "--model",
    #     "-m",
    #     metavar="MODEL",
    #     default=RESNET,
    #     choices=MODELS,
    #     help="pretrained model: " + " | ".join(MODELS) + " (default: {})".format(RESNET),
    # )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run (default: 100)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=4,
        type=int,
        metavar="N",
        help="mini-batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.0001,
        type=float,
        metavar="LR",
        help="initial learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum (default: 0.9)"
    )
    parser.add_argument(
        "--lr-decay-freq",
        default=30,
        type=float,
        metavar="W",
        help="learning rate decay (default: 30)",
    )
    parser.add_argument(
        "--lr-decay",
        default=0.1,
        type=float,
        metavar="W",
        help="learning rate decay (default: 0.1)",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=100,
        type=int,
        metavar="N",
        help="print frequency (default: 100)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        help="Number of local epochs to run at each client before synchronizing",
        default=1,
    )
    parser.add_argument(
        "--quant_level", type=int, help="Number of quantization buckets", default=0
    )
    parser.add_argument(
        "--vecdim", type=int, help="Vector quantization dimension", default=1
    )
    parser.add_argument("--comp", type=str, help="Which compressor", default="")
    parser.add_argument("--seed", type=int, help="Random seed to use", default=0)
    
    parser.add_argument('--dataset', default="radar", type=str) # covtype, higgs,gisette, realsim, epsilon, letter, radar
    parser.add_argument('--splitter', default="corr", type=str) # corr, imp
    parser.add_argument('--weight', default="0.3", type=str)
    parser.add_argument('--dataseed', default="0", type=str)

    # get time
    curtime = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H-%M-%S")
    parser.add_argument('--folder', default=curtime, type=str) # the result saving folder, default using the time

    # Parse input arguments
    args = parser.parse_args()
    num_clients = args.num_clients
    suffix = f"_{args.dataset}_NC{args.num_clients}_LE{args.local_epochs}_quant{args.quant_level}_dim{args.vecdim}_comp{args.comp}_seed{args.seed}_dataseed{args.dataseed}_splitter{args.splitter}_weight{args.weight}_bs{args.batch_size}_TE{args.epochs}_LR{args.lr}"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "Loading data")

    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dset_train = None
    dset_val = None

    # covtype, higgs,gisette, realsim, epsilon, letter, radar, cifar10, mnist
    if args.dataset == "covtype":
        from datasets.covtype import CovType
        dset_train = CovType("train", args.splitter, args.weight, args.dataseed, args.num_clients)
        dset_val = CovType("test", args.splitter, args.weight, args.dataseed, args.num_clients)
    elif args.dataset == "higgs":
        from datasets.higgs import Higgs
        dset_train = Higgs("train", args.splitter, args.weight, args.dataseed, args.num_clients)
        dset_val = Higgs("test", args.splitter, args.weight, args.dataseed, args.num_clients)
    elif args.dataset == "gisette":
        from datasets.gisette import Gisette
        dset_train = Gisette("train", args.splitter, args.weight, args.dataseed, args.num_clients)
        dset_val = Gisette("test", args.splitter, args.weight, args.dataseed, args.num_clients)
    elif args.dataset == "realsim":
        from datasets.realsim import Realsim
        dset_train = Realsim("train", args.splitter, args.weight, args.dataseed, args.num_clients)
        dset_val = Realsim("test", args.splitter, args.weight, args.dataseed, args.num_clients)
    elif args.dataset == "epsilon":
        from datasets.epsilon import Epsilon
        dset_train = Epsilon("train", args.splitter, args.weight, args.dataseed, args.num_clients)
        dset_val = Epsilon("test", args.splitter, args.weight, args.dataseed, args.num_clients)
    elif args.dataset == "letter":
        from datasets.letter import Letter
        dset_train = Letter("train", args.splitter, args.weight, args.dataseed, args.num_clients)
        dset_val = Letter("test", args.splitter, args.weight, args.dataseed, args.num_clients)
    elif args.dataset == "radar":
        from datasets.radar import Radar
        dset_train = Radar("train", args.splitter, args.weight, args.dataseed, args.num_clients)
        dset_val = Radar("test", args.splitter, args.weight, args.dataseed, args.num_clients)
    elif args.dataset == "msd":
        from datasets.msd import MSD
        dset_train = MSD("train", args.splitter, args.weight, args.dataseed, args.num_clients)
        dset_val = MSD("test", args.splitter, args.weight, args.dataseed, args.num_clients)
    elif args.dataset == "cifar10":
        from datasets.cifar10 import CIFAR10
        dset_train = CIFAR10("train", args.splitter, args.weight, args.dataseed, args.num_clients)
        dset_val = CIFAR10("test", args.splitter, args.weight, args.dataseed, args.num_clients)
    elif args.dataset == "mnist":
        from datasets.mnist import MNIST
        dset_train = MNIST("train", args.splitter, args.weight, args.dataseed, args.num_clients)
        dset_val = MNIST("test", args.splitter, args.weight, args.dataseed, args.num_clients)
    elif args.dataset == "wide":
        from datasets.wide import Wide
        dset_train = Wide("train")
        dset_val = Wide("test")
    elif args.dataset == "vehicle":
        from datasets.vehicle import Vehicle
        dset_train = Vehicle("train")
        dset_val = Vehicle("test")
    else:
        assert False, "Unsupported dataset"
    
    DATASET_SPLIT = dset_train.partitions
    
    # Load dataset
    
    train_loader = DataLoader(
        dset_train, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    test_loader = DataLoader(
        dset_val, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    classes = dset_train.classes
    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "Classes:", len(classes), classes)
    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), args)
    losses = []
    accs_train = []
    accs_test = []

    best_acc = 0.0
    best_loss = 0.0
    start_epoch = 0

    models = []
    optimizers = []
    # Make models for each client
    for i in range(num_clients + 1):
        if i == num_clients:
            if args.dataset == "msd":
                model = VertiBench_Cls_top(
                    num_classes=1,
                    num_clients=num_clients,
                    activation = nn.Sigmoid()
                )
            elif args.dataset in ["cifar10", "mnist"]:
                model = resnet_top(
                    pretrained=args.pretrained,
                    num_classes=len(classes),
                    num_clients=num_clients,
                )
            else:
                model = VertiBench_Cls_top(
                    num_classes=len(classes),
                    num_clients=num_clients,
                    activation = None
                )
        else:
            if args.dataset == "cifar10":
                model = resnet18(weights=None)
                model.fc = nn.Identity()
                model.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=3, bias=False)
                # model = ResNet18(input_channels=3, num_classes=len(classes))
            elif args.dataset == "mnist":
                model = resnet18(weights=None)
                model.fc = nn.Identity()
                model.conv1 = nn.Conv2d(1, 64, 9, stride=2, padding=3, bias=False)
                # model = ResNet18(input_channels=1, num_classes=len(classes))
            else:
                model = VertiBench_Cls_bottom(n_features=dset_train.parties[i].X.shape[1])

        model.to(device)
        cudnn.benchmark = True
        print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"),f"Client {i} mode", model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        models.append(model)
        optimizers.append(optimizer) # modified by Junyi

    if args.dataset == "msd":
        server_model_comp = VertiBench_Cls_top( # MSD is a regression dataset
            num_classes=1, num_clients=num_clients
        )
    elif args.dataset in ["cifar10", "mnist"]:
        server_model_comp = resnet_top( # pretrained: False, num_classes: 10, num_clients: 4
            pretrained=args.pretrained, num_classes=len(classes), num_clients=num_clients
        )
    else:
        server_model_comp = VertiBench_Cls_top( # pretrained: False, num_classes: 10, num_clients: 4
            num_classes=len(classes), num_clients=num_clients
        )

    server_model_comp.to(device)
    server_optimizer_comp = torch.optim.Adam(server_model_comp.parameters(), lr=args.lr)
    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "Server model:", server_model_comp)
    # Loss and Optimizer
    n_epochs = args.epochs
    if args.dataset == "msd":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    coords_per = 16

    # Get initial loss/accuracy
    if start_epoch == 0:
        save_eval(
            models,
            train_loader,
            test_loader,
            losses,
            accs_train,
            accs_test,
            0,
            len(train_loader),
        )
    # Training / Eval loop
    train_size = len(train_loader)
    for epoch in range(start_epoch, n_epochs):
        print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "\n-----------------------------------")
        print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "Epoch: [%d/%d]" % (epoch + 1, n_epochs))
        start = time.time()

        train(models, optimizers, epoch)
        print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "Time taken: %.2f sec." % (time.time() - start))
        save_eval(
            models,
            train_loader,
            test_loader,
            losses,
            accs_train,
            accs_test,
            epoch,
            train_size,
        )

        for i in range(num_clients + 1):
            PATH = f"./{args.folder}/checkpoint{i}{suffix}.pt"
            # 禁止保存，加快速度
            # torch.save(
            #     {
            #         "epoch": epoch + 1,
            #         "model_state_dict": models[i].state_dict(),
            #         "optimizer_state_dict": optimizers[i].state_dict(),
            #         "loss": 0,
            #     },
            #     PATH,
            # )
        print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "Time taken: %.2f sec." % (time.time() - start))

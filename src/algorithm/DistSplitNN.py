import os
import sys
import warnings
from datetime import datetime
import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.multiprocessing import Process

from tqdm import tqdm
import pytz

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.VFLDataset import VFLSynAlignedDataset
from dataset.LocalDataset import LocalDataset
from algorithm.SplitNN import SplitMLP
from utils.utils import get_metric_from_str, get_device_from_gpu_id

LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])
MASTER_ADDR = os.environ['MASTER_ADDR']
MASTER_PORT = os.environ['MASTER_PORT']


def init_processes(rank, size, backend='nccl', args=None):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(backend, rank, size, args)


def run(backend, rank, size, args):
    """
    Run the distributed
    :param backend: backend of the distributed environment. Should be in ['nccl', 'gloo']. nccl for GPU, gloo for CPU.
    :param rank: rank of the current process.
    :param size: world size of the distributed environment. Should be equal to args.n_parties.
    :param args: arguments from parser.parse_args()
    :return:
    """
    group = dist.new_group(list(range(args.n_parties)))

    train_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{args.dataset}", f'{args.dataset}', args.n_parties,
                                                     primary_party_id=args.primary_party, splitter=args.splitter,
                                                     weight=args.weights, beta=args.beta, seed=args.seed, type='train')
    test_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{args.dataset}", f'{args.dataset}', args.n_parties,
                                                    primary_party_id=args.primary_party, splitter=args.splitter,
                                                    weight=args.weights, beta=args.beta, seed=args.seed, type='test')

    # create the model
    if args.n_classes == 1:  # regression
        task = 'reg'
        loss_fn = nn.MSELoss()
        out_dim = 1
        out_activation = nn.Sigmoid()
        if args.metric == 'acc':  # if metric is accuracy, change it to rmse
            args.metric = 'rmse'
            warnings.warn("Metric is changed to rmse for regression task")
    elif args.n_classes == 2:  # binary classification
        task = 'bin-cls'
        loss_fn = nn.BCELoss()
        out_dim = 1
        out_activation = nn.Sigmoid()
        # make sure the labels are in [0, 1]
        train_dataset.scale_y_()
        test_dataset.scale_y_()
    else:  # multi-class classification
        task = 'multi-cls'
        loss_fn = nn.CrossEntropyLoss()
        out_dim = args.n_classes
        out_activation = None  # No need for softmax since it is included in CrossEntropyLoss

    model = SplitMLP(train_dataset.local_input_channels, [[100, 100]] * args.n_parties, [200, out_dim],
                         out_activation=out_activation, primary_party=args.primary_party)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    metric_fn = get_metric_from_str(args.metric)

    device = get_device_from_gpu_id(rank + args.gpu)
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        train_pred_y = train_y = torch.zeros([0, 1], device=device)
        total_loss = 0

        for Xs, y in tqdm(train_loader):
            Xs = [Xi.to(device) for Xi in Xs]
            y = y.to(device)
            y = y.long() if task == 'multi-cls' else y

            optimizer.zero_grad()

            cut_output = model.local_mlps[rank](Xs[rank])
            cut_gather = [torch.zeros_like(cut_output) for _ in range(args.n_parties)]
            if rank == args.primary_party:
                dist.gather(cut_output, gather_list=cut_gather, group=group)
            else:
                dist.gather(cut_output, dst=args.primary_party, group=group)
            cut_concat = torch.cat(cut_gather, dim=1)
            if rank == args.primary_party:
                y_pred = model.agg_mlp(cut_concat)

                y_pred = y_pred.flatten() if task in ['reg', 'bin-cls'] else y_pred
                loss = loss_fn(y_pred, y)
                total_loss += loss.item()

                if args.n_classes == 2:
                    y_pred = torch.round(y_pred)
                elif args.n_classes > 2:
                    y_pred = torch.argmax(y_pred, dim=1).reshape(-1, 1)

                y_pred = y_pred.reshape(-1, 1)
                train_pred_y = torch.cat([train_pred_y, y_pred], dim=0)
                train_y = torch.cat([train_y, y.reshape(-1, 1)], dim=0)
                loss.backward(retain_graph=True)
                optimizer.step()
        if rank == args.primary_party:
            train_score = metric_fn(train_y.data.cpu().numpy(), train_pred_y.data.cpu().numpy())
            print(datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), f"Epoch: {epoch}, Train Loss: {total_loss / len(train_loader)}, Train Score: {train_score}")
            if hasattr(model, 'comm_logger') and model.comm_logger is not None:
                print(datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"))
                model.comm_logger.save_log()

            if scheduler is not None:
                scheduler.step()

        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                test_y_pred = test_y = torch.zeros([0, 1], device=device)
                for Xs, y in tqdm(test_loader):
                    Xs = [Xi.to(device) for Xi in Xs]
                    y = y.to(device)
                    y = y.long() if task == 'multi-cls' else y

                    cut_output = model.local_mlps[rank](Xs[rank])
                    cut_gather = [torch.zeros_like(cut_output) for _ in range(args.n_parties)]
                    if rank == args.primary_party:
                        dist.gather(cut_output, gather_list=cut_gather, group=group)
                    else:
                        dist.gather(cut_output, dst=args.primary_party, group=group)
                    cut_concat = torch.cat(cut_gather, dim=1)
                    if rank == args.primary_party:
                        y_pred = model.agg_mlp(cut_concat)
                        y_pred = y_pred.flatten() if task in ['reg', 'bin-cls'] else y_pred
                        test_loss += loss_fn(y_pred, y)

                        if args.n_classes == 2:
                            y_pred = torch.round(y_pred)
                        elif args.n_classes > 2:
                            y_pred = torch.argmax(y_pred, dim=1).reshape(-1, 1)

                        y_pred = y_pred.reshape(-1, 1)
                        test_y_pred = torch.cat([test_y_pred, y_pred], dim=0)
                        test_y = torch.cat([test_y, y.reshape(-1, 1)], dim=0)
                if rank == args.primary_party:
                    test_score = metric_fn(test_y.data.cpu().numpy(), test_y_pred.data.cpu().numpy())
                    print(datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), f"Epoch: {epoch}, Test Loss: {test_loss / len(test_loader)}, Test Score: {test_score}")


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help="GPU ID. Set to None if you want to use CPU")

    # parameters for dataset
    parser.add_argument('--dataset', '-d', type=str, default='covtype',
                        help="dataset to use.")
    parser.add_argument('--n_parties', '-p', type=int, default=4,
                        help="number of parties. Should be >=2")
    parser.add_argument('--primary_party', '-pp', type=int, default=0,
                        help="primary party. Should be in [0, n_parties-1]")
    parser.add_argument('--splitter', '-sp', type=str, default='imp')
    parser.add_argument('--weights', '-w', type=float, default=1, help="weights for the ImportanceSplitter")
    parser.add_argument('--beta', '-b', type=float, default=1, help="beta for the CorrelationSplitter")

    # parameters for model
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--lr', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--n_classes', '-c', type=int, default=7,
                        help="number of classes. 1 for regression, 2 for binary classification,"
                             ">=3 for multi-class classification")
    parser.add_argument('--metric', '-m', type=str, default='acc',
                        help="metric to evaluate the model. Supported metrics: [accuracy, rmse]")
    parser.add_argument('--seed', '-s', type=int, default=0, help="random seed")
    args = parser.parse_args()

    size = WORLD_SIZE
    backend = 'nccl'
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, backend, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


"""
Consider a two-party application on covtype dataset.
On one party:
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 src/algorithm/DistSplitNN.py -d covtype -c 7 -m acc -p 2 -sp corr -b 0.0 -s 0 -g 0

On another party:   (change only the node_rank)
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=127.0.0.1 --master_port=12345 src/algorithm/DistSplitNN.py -d covtype -c 7 -m acc -p 2 -sp corr -b 0.0 -s 0 -g 0

Note that only the parameters on the primary party will be used. The parameters on the other parties are ignored.
"""
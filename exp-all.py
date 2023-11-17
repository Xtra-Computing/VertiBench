import os
import pytz
import random
import socket
import datetime
import subprocess

def get_fedonce_command(
    folder,
    dataset: str,
    n_classes: int,
    metric: str,
    dataseed,
    splitter,
    weight,
    gpu_index,
    num_clients=4,
    global_epochs=100,
    agg_epochs = 100,
    local_lr = 3e-4,
    agg_lr = 1e-4,
    local_batch_size = 128,
    agg_batch_size = 128,
    **other_args,
):
    if dataset in ['mnist', 'cifar10']:
        model_type = 'resnet18'
    else:
        model_type = 'fc'

    # metric: acc, rmse
    command = f"cd /home/junyi/VertiBench && CUDA_VISIBLE_DEVICES={gpu_index} python src/algorithm/fedonce/train_fedonce.py --dataset {dataset} --n_classes {n_classes} -m {metric} --n_parties {num_clients} --seed {dataseed} "
    command += f"--local_lr {local_lr} --agg_lr {agg_lr} --epochs {global_epochs} --agg_epochs {agg_epochs} --local_batch_size {local_batch_size} --agg_batch_size {agg_batch_size} --splitter {splitter} --gpu 0 --model_type {model_type} "

    if splitter == "imp":
        command += f" --weights {weight} "
    elif splitter == "corr":
        command += f" --beta {weight} "
    else:
        raise ValueError(f"Unknown splitter {splitter}")

    command += f" > {folder}/fedonce_{dataset}_c{num_clients}_locallr{local_lr}_agglr{agg_lr}_ds{dataseed}_ge{global_epochs}_ae{agg_epochs}_bs{local_batch_size}_{splitter}_{weight}.log 2>&1"
    return command



def get_cvfl_command(
    folder,
    dataset,
    dataseed,
    lr,
    num_clients,
    local_epochs,
    global_epochs,
    batch_size,
    splitter,
    weight,
    gpu_index,
    **other_args,
):
    # splitter: corr, imp
    # weight(imp): 0.1, 1.0, 10.0, 100.0
    # weight(corr): 0.1, 0.3, 0.6, 1.0
    command = (
        f"cd /home/junyi/VertiBench/src/algorithm/cvfl && CUDA_VISIBLE_DEVICES={gpu_index} python quant_radar.py 10class/classes/ --num_clients {num_clients} --seed {dataseed} --b {batch_size} "
        f"--local_epochs {local_epochs} --epochs {global_epochs} --lr {lr} --quant_level 4 --vecdim 1 --comp topk "
        f"--dataset {dataset} --splitter {splitter} --weight {weight} --dataseed {dataseed} --folder {folder} > "
        f"{folder}/cvfl_{dataset}_c{num_clients}_lr{lr}_ds{dataseed}_le{local_epochs}_ge{global_epochs}_bs{batch_size}_{splitter}_{weight}.log 2>&1"
    )
    return command


def get_splitnn_command(
    folder,
    dataset: str,
    n_classes: int,
    metric: str,
    dataseed,
    splitter,
    weight,
    gpu_index,
    lr: str = "1e-3",
    num_clients=4,
    global_epochs=50,
    batch_size=128,
    **other_args,
):
    # metric: acc, rmse
    command = f"cd /home/junyi/VertiBench && CUDA_VISIBLE_DEVICES={gpu_index} python src/algorithm/SplitNN.py --dataset {dataset} --n_classes {n_classes} -m {metric} --n_parties {num_clients} --seed {dataseed} "
    command += f"--lr {lr} --epochs {global_epochs} --batch_size {batch_size} --splitter {splitter} --gpu 0 "

    if splitter == "imp":
        command += f"--weights {weight} "
    elif splitter == "corr":
        command += f"--beta {weight} "
    else:
        raise ValueError(f"Unknown splitter {splitter}")

    command += f" > {folder}/splitnn_{dataset}_c{num_clients}_lr{lr}_ds{dataseed}_ge{global_epochs}_bs{batch_size}_{splitter}_{weight}.log 2>&1"
    return command


def get_gal_command(
    folder,
    dataset: str,
    dataseed,
    splitter,
    weight,
    gpu_index,
    lr: str = "0.01",
    num_clients=4,
    local_epochs=20,
    global_epochs=200,
    batch_size=512,
    **other_args,
):
    if dataset == "cifar10":
        model = "resnet18_vb"
        dataset = "CIFAR10_VB"
    elif dataset == "mnist":
        model = "resnet18_vb"
        dataset = "MNIST_VB"
    elif dataset == "msd":
        model = "linear"
        dataset = "MSD"
        global_epochs = 5
    else:
        model = "classifier"

    if dataset == 'gisette':
        dataset = "Gisette"
    elif dataset == 'letter':
        dataset = "Letter"
    elif dataset == 'covtype':
        dataset = "CovType"
    elif dataset == 'radar':
        dataset = "Radar"
    elif dataset == 'realsim':
        dataset = "Realsim"
    elif dataset == 'epsilon':
        dataset = "Epsilon"

    control = f"{num_clients}_stack_{local_epochs}_{global_epochs}_search_0"
    
    command = f"cd /home/junyi/VertiBench/src/algorithm/gal && CUDA_VISIBLE_DEVICES={gpu_index} BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 0.0 --dataseed {dataseed}"
    command += f" > {folder}/gal_{dataset}_c{num_clients}_lr{lr}_ds{dataseed}_le{local_epochs}_ge{global_epochs}_bs{batch_size}_{splitter}_{weight}.log 2>&1"
    return command


def get_fedtree_command(
    folder,
    dataset: str,
    n_classes: int,
    dataseed,
    splitter,
    weight,
    gpu_index,
    lr: str = "1e-3",
    num_clients=4,
    global_epochs=50,
    **other_args,
):
    # metric: acc, rmse
    
    command = ""
    for i in range(num_clients):
        if splitter == "imp":
            w = "weight"
        elif splitter == "corr":
            w = "beta"
        
        if dataset in ["gisette", "realsim", "epsilon"]:
            scale_y = " --scale-y"
        else:
            scale_y = ""

        if dataset in ['mnist', 'cifar10']:
            command += f"cd /home/junyi/VertiBench && python src/preprocess/pkl_to_csv.py data/syn/{dataset}/{dataset}_train_party{num_clients}-{i}_{splitter}_{w}{weight}_seed{dataseed}_train.pkl {scale_y} && " # a bug in splitter that cannot rename the mnist and cifar10
            command += f"cd /home/junyi/VertiBench && python src/preprocess/pkl_to_csv.py data/syn/{dataset}/{dataset}_test_party{num_clients}-{i}_{splitter}_{w}{weight}_seed{dataseed}_train.pkl {scale_y} && "
        else:
            command += f"cd /home/junyi/VertiBench && python src/preprocess/pkl_to_csv.py data/syn/{dataset}/{dataset}_party{num_clients}-{i}_{splitter}_{w}{weight}_seed{dataseed}_train.pkl {scale_y} && "
            command += f"cd /home/junyi/VertiBench && python src/preprocess/pkl_to_csv.py data/syn/{dataset}/{dataset}_party{num_clients}-{i}_{splitter}_{w}{weight}_seed{dataseed}_test.pkl {scale_y} && "
    command += f"cd /home/junyi/VertiBench && CUDA_VISIBLE_DEVICES={gpu_index} python src/algorithm/FedTree.py --dataset {dataset} --n_classes {n_classes} --lr {lr} --epochs {global_epochs} --n_parties {num_clients} -sp {splitter} --seed {dataseed} "

    if splitter == "imp":
        command += f"--weight {weight} "
    elif splitter == "corr":
        command += f"--beta {weight} "
    else:
        raise ValueError(f"Unknown splitter {splitter}")

    command += f" > {folder}/fedtree_{dataset}_c{num_clients}_lr{lr}_ds{dataseed}_ge{global_epochs}_{splitter}_{weight}.log 2>&1"
    return command


def main():
    dry_run = os.environ.get("DRYRUN")
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    hostname = socket.gethostname()
    print("Running on", hostname)

    if slurm_task_id is not None:
        slurm_task_id = int(slurm_task_id)  # 转换为整数，如果需要
        print(f"SLURM_ARRAY_TASK_ID: {slurm_task_id}")
    elif dry_run is None:
        raise ValueError("SLURM_ARRAY_TASK_ID not set.")
    # if not "fedtree" in os.environ["CONDA_DEFAULT_ENV"]:
    #     raise ValueError("Not in a valid conda environment")

    folder = f"/home/junyi/rebuttal/iclr2024/experiment"
    os.system("mkdir -p " + folder)

    parties = [
        ("4", "corr", "0.0"),
        ("4", "corr", "0.3"),
        ("4", "corr", "0.6"),
        ("4", "corr", "1.0"),

        ("4", "imp", "0.1"),
        ("4", "imp", "1.0"),
        ("4", "imp", "10.0"),
        ("4", "imp", "100.0"),
    ]

    seeds = ["0", "1", "2", "3", "4"]

    datasets = ["letter", "covtype", "msd", "gisette", "radar", "realsim", "epsilon", "mnist", "cifar10"]
    # datasets = ["gisette", "realsim", "epsilon"]
    
    classes = {
        "gisette": 2,
        "letter": 26,
        "covtype": 7,
        "msd": 1,
        "radar": 7,
        "realsim": 2,
        "epsilon": 2,
        "mnist": 10,
        "cifar10": 10,
    }

    metrics = {
        "gisette": "acc",
        "letter": "acc",
        "covtype": "acc",
        "msd": "rmse",
        "radar": "acc",
        "realsim": "acc",
        "epsilon": "acc",
        "mnist": "acc",
        "cifar10": "acc",
    }

    if "gpu5" in hostname:
        gpu_count = 8
    elif "gpu0" in hostname:
        gpu_count = 8
    elif "gpu4" in hostname:
        gpu_count = 8
    else:
        gpu_count = 4
    gpu_index = 0

    tt = datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M")
    
    cmd = 'echo "NO TASK"'

    print(f"{tt} Running {slurm_task_id}/{len(seeds) * len(datasets) * len(parties) * 4} tasks. (Index starts from 1)")
    
    cnt = 1
    for seed in seeds:
        for dataset in datasets:
            for party in parties: # PartyNumber and splitter settings
                kwargs = {
                    "folder": folder,
                    "dataseed": seed,
                    "num_clients": int(party[0]),
                    "splitter": party[1],
                    "weight": party[2],
                    "dataset": dataset,
                    "n_classes": classes[dataset],
                    "metric": metrics[dataset],
                }

                gpu_index = random.randint(0, gpu_count-1)
                
                if cnt == slurm_task_id or dry_run:
                    cmd = get_fedonce_command(**kwargs, local_lr=3e-4, agg_lr=1e-4, local_batch_size=128, agg_batch_size=128, global_epochs=100, gpu_index=gpu_index)
                    print(cnt, cmd)
                cnt += 1

                if cnt == slurm_task_id or dry_run:
                    cmd = get_fedtree_command(**kwargs, lr="0.1", global_epochs=50, gpu_index=gpu_index)
                    print(cnt, cmd)
                cnt += 1
                
                if cnt == slurm_task_id or dry_run:
                    cmd = get_splitnn_command(**kwargs, lr="0.001", global_epochs=50, batch_size=128, gpu_index=gpu_index)
                    print(cnt, cmd)
                cnt += 1
                
                if cnt == slurm_task_id or dry_run:
                    cmd = get_gal_command(**kwargs, lr="0.01", local_epochs=20, global_epochs=20, batch_size=512, gpu_index=gpu_index)
                    print(cnt, cmd)
                cnt += 1

                if cnt == slurm_task_id or dry_run:
                    cmd = get_cvfl_command(**kwargs, lr="0.0001", local_epochs=10, global_epochs=200, batch_size=512, gpu_index=gpu_index)
                    print(cnt, cmd)
                cnt += 1
            
            # assign gpu to each dataset 
            # gpu_index += 1
            # if gpu_index == gpu_count:
            #     gpu_index = 0
    
    if dry_run is None:
        subprocess.run(cmd, shell=True)
    print("Done.")

if __name__ == "__main__":
    main()

# DRYRUN=1 python exp-all.py > exp-all.sh
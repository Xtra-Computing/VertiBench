import os
import sys
import argparse 
import subprocess
import multiprocessing
from queue import Empty
import datetime, pytz

def process_wrapper(gpuid_queue, command, times):
    while True:
        try:
            gpu_idx = gpuid_queue.get(block=True, timeout=None)
        except Empty:
            continue
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_idx} " + command + f'_final_repeat{times}.txt 2>&1'
        pos1 = cmd.find('> ')
        log_file_name = cmd[pos1 + 2: -5]
        
        # read log_file_name and check wether "Test Epoch: 20" in the log
        already_run = False
        if os.path.exists(log_file_name):
            with open(log_file_name, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'Test Epoch: 20' in line:
                        already_run = True
                        break

        if already_run:
            print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "✅ Already run: ", cmd)
        else:
            print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "🟡 Running command: ", cmd)
            subprocess.call(cmd, shell=True)
        gpuid_queue.put(gpu_idx)
        break
    gpuid_queue.close()


def get_commands(folder, lr, dataseed, dataset, model, control, batch_size):
    master_addr = '192.168.47.111'
    master_port = 12347
    commands = [
        f'tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.0 "mkdir -p {folder}; ifconfig eno1 > {folder}/begin0.txt && BATCH_SIZE={batch_size} LR={lr} GLOO_DEBUG=INFO TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=INFO GLOO_SOCKET_IFNAME=eno1 torchrun --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr={master_addr} --master_port={master_port} train_model_assist_distributed.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 0.1 --dataseed {dataseed} > {folder}/imp_0.1_party0.log && ifconfig eno1 > {folder}/end0.txt" Enter',
        f'tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.1 "mkdir -p {folder}; ifconfig eno1 > {folder}/begin1.txt && BATCH_SIZE={batch_size} LR={lr} GLOO_DEBUG=INFO TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=INFO GLOO_SOCKET_IFNAME=eno1 torchrun --nproc_per_node=1 --nnodes=4 --node_rank=1 --master_addr={master_addr} --master_port={master_port} train_model_assist_distributed.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 0.1 --dataseed {dataseed} > {folder}/imp_0.1_party1.log && ifconfig eno1 > {folder}/end1.txt" Enter',
        f'tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.2 "mkdir -p {folder}; ifconfig eno1 > {folder}/begin2.txt && BATCH_SIZE={batch_size} LR={lr} GLOO_DEBUG=INFO TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=INFO GLOO_SOCKET_IFNAME=eno1 torchrun --nproc_per_node=1 --nnodes=4 --node_rank=2 --master_addr={master_addr} --master_port={master_port} train_model_assist_distributed.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 0.1 --dataseed {dataseed} > {folder}/imp_0.1_party2.log && ifconfig eno1 > {folder}/end2.txt" Enter',
        f'tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.3 "mkdir -p {folder}; ifconfig eno1 > {folder}/begin3.txt && BATCH_SIZE={batch_size} LR={lr} GLOO_DEBUG=INFO TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=INFO GLOO_SOCKET_IFNAME=eno1 torchrun --nproc_per_node=1 --nnodes=4 --node_rank=3 --master_addr={master_addr} --master_port={master_port} train_model_assist_distributed.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 0.1 --dataseed {dataseed} > {folder}/imp_0.1_party3.log && ifconfig eno1 > {folder}/end3.txt" Enter ',
    ]
    
    return commands


def get_args():
    parser = argparse.ArgumentParser(description="GAL running script")  # 脚本描述
    parser.add_argument('-s', '--seeds', help="Random seed for dataset and GAL. You can specify multiple seeds to run. -s 0 1 2 3 4", nargs='+', type=int, required=True)
    parser.add_argument('-c', '--clients', help="Number of clients. -c 4", type=int, required=True)
    
    parser.add_argument('-t', '--ntask', help="How many task you want to run on each gpu", type=int, required=True)
    parser.add_argument('-d', '--datasets', help="What datasets to run. -d MSD Gisette",nargs='+', type=str, required=True)
    parser.add_argument('-lr', '--learning-rates', help="Learning rate. You can specify multiple lr to run. -lr 0.1 0.01 0.001",nargs='+', type=str, required=True)
    parser.add_argument('-l', '--local-epoch', help="Local epochs. -l 20", type=int, required=True)
    parser.add_argument('-e', '--global-epoch', help="Total epochs. -e 20", type=int, required=True)
    parser.add_argument('-b', '--batch-size', help="Batch size. -b 512", type=int, required=True)
   
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    
    num_tasks = args.ntask
    local_epoch = args.local_epoch
    total_epoch = args.global_epoch
    batch_size  = args.batch_size
    num_clients = args.clients

    for lr in args.learning_rates:
        for dataset in args.datasets:
            for seed in args.seeds:
                if dataset in ["CIFAR10_VB", "MNIST_VB"]:
                    model = 'resnet18_vb'
                    control = f'{num_clients}_stack_{local_epoch}_{total_epoch}_search_0'
                    raise Exception("AMD Cluster do not support resnet18, weird...")
                elif dataset == 'MSD':
                    model = 'linear'
                    control = f'{num_clients}_stack_{local_epoch}_5_search_0'
                else:
                    model = 'classifier'
                    control = f'{num_clients}_stack_{local_epoch}_{total_epoch}_search_0'
                
                tt = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S")
                folder = f'./results_real_dist_{tt}/{dataset}_client{num_clients}_lr{lr}_seed{seed}_local{local_epoch}_global{total_epoch}_bs{batch_size}'

                # print(f'mkdir -p {folder}')
                commands = get_commands(folder, lr, seed, dataset, model, control, batch_size)
                for cmd in commands:
                    print(cmd)
            print('')
"""
rebuttal real distributed


PARTITION1=mi210_u250_u55c
NODE1=hacc-gpu1
PARTITION2=mi100
NODE2=hacc-gpu5
PARTITION3=mi210_vck_u55c
NODE3=hacc-gpu3
PARTITION4=mi100
NODE4=hacc-gpu4
TIME=9999
CPUS=4

CMD1="srun -p $PARTITION1 --time=$TIME --cpus-per-task=$CPUS -w $NODE1 --pty zsh -i"
CMD2="srun -p $PARTITION2 --time=$TIME --cpus-per-task=$CPUS -w $NODE2 --pty zsh -i"
CMD3="srun -p $PARTITION3 --time=$TIME --cpus-per-task=$CPUS -w $NODE3 --pty zsh -i"
CMD4="srun -p $PARTITION4 --time=$TIME --cpus-per-task=$CPUS -w $NODE4 --pty zsh -i"

tmux set -g pane-border-status top
tmux set -g pane-border-format "#{session_name}:window#{window_index}.pane#{pane_index}.#{pane_title}"

# Session
SESSION_NAME="Distributed Test"
tmux new-session -d -s $SESSION_NAME

# Window
WINDOW_NAME="ALL"
tmux new-window -t $SESSION_NAME -n $WINDOW_NAME $CMD1
tmux split-window -h -t $SESSION_NAME:$WINDOW_NAME.0 $CMD2
tmux split-window -t $SESSION_NAME:$WINDOW_NAME.0 $CMD3
tmux split-window -t $SESSION_NAME:$WINDOW_NAME.2 $CMD4
sleep 5
tmux select-pane -t $SESSION_NAME:$WINDOW_NAME.0 -T "gpu1"
tmux select-pane -t $SESSION_NAME:$WINDOW_NAME.1 -T "gpu3"
tmux select-pane -t $SESSION_NAME:$WINDOW_NAME.2 -T "gpu5"
tmux select-pane -t $SESSION_NAME:$WINDOW_NAME.3 -T "gpu4"

tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.0 "conda activate gal && cd /home/junyi/data/gal/src" Enter
tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.1 "conda activate gal && cd /home/junyi/data/gal/src" Enter
tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.2 "conda activate gal && cd /home/junyi/data/gal/src" Enter
tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.3 "conda activate gal && cd /home/junyi/data/gal/src" Enter

tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.0 "clear" Enter
tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.1 "clear" Enter
tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.2 "clear" Enter
tmux send-keys -t $SESSION_NAME:$WINDOW_NAME.3 "clear" Enter


python run_distributed.py --seeds 0 --clients 4 --ntask 2 -d CovType MSD Gisette Realsim Epsilon Letter Radar -lr 0.01 -l 1 -e 50 -b 512 > run_distributed.sh

"""


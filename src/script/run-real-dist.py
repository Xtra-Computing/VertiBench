

import time
import argparse


ALGOTIHMS = [
    'cvfl',
    'fedtree',
    'gal',
    'splitnn'
]

CLASSES = {
    'covtype': '7',
    'msd': '1',
    'gisette': '2',
    'realsim': '2',
    'epsilon': '2',
    'letter': '26',
    'radar': '7'
}

METRICS = {
    'covtype': 'acc',
    'msd': 'rmse',
    'gisette': 'acc',
    'realsim': 'acc',
    'epsilon': 'acc',
    'letter': 'acc',
    'radar': 'acc'
}

TT = time.strftime("%H%M%S", time.localtime(time.time()))
TR_ADDR = "192.168.47.111"
TR_PORT = "37596"
TR_ARGS = f"--master_addr={TR_ADDR} --master_port={TR_PORT}"

ENVS = [
    "TORCH_CPP_LOG_LEVEL=INFO",
    "TORCH_DISTRIBUTED_DEBUG=INFO",
    "GLOO_DEBUG=INFO",
    "GLOO_SOCKET_IFNAME=eno1", # binding network interface
    "OMP_NUM_THREADS=20"
]
ENVS = " ".join(ENVS)

SLURM_TIME=9999
SLURM_CPUS=4
SLURM_PART = [
    ('mi210_u250_u55c', 'hacc-gpu1'),
    ('mi100', 'hacc-gpu5'),
    ('mi210_vck_u55c', 'hacc-gpu3'),
    ('mi100', 'hacc-gpu4')
]
SLURM_CMDS = [] # commands for connecting to slurm node
for partition, node in SLURM_PART:
    SLURM_CMDS.append(f"srun -p {partition} --time={SLURM_TIME} --cpus-per-task={SLURM_CPUS} -w {node} --pty zsh -i")


def tmux_kill_session(session_name):
    print(f"tmux kill-session -t {session_name}")

def tmux_create_session(session_name):
    print(f"tmux new-session -d -s {session_name}")
    
def tmux_create_4windows(session_name, window_name = "RealDist"):
    if ' ' in window_name:
        window_name = window_name.replace(' ', '_')
    print("\n# create 4 windows")
    print(f'tmux new-window -t {session_name} -n {window_name} \; split-window -v \; split-window -h \; select-pane -t 0 \; split-window -h')
    print(f'tmux set -g pane-border-status top')
    print('tmux set -g pane-border-format "#{session_name}:window#{window_index}.pane#{pane_index}.#{pane_title}"')
    # print(f'tmux select-layout -t {session_name}:{window_name} tiled')
    print(f'tmux list-panes -t {session_name}:{window_name}')

def tmux_send_cmd(session_name, pane_index: int, cmd: str, window_name = "RealDist", enter = True):
    if ' ' in window_name:
        window_name = window_name.replace(' ', '_')
    if enter:
        print(f'tmux send-keys -t {session_name}:{window_name}.{pane_index} "{cmd}" Enter')
    else:
        print(f'tmux send-keys -t {session_name}:{window_name}.{pane_index} "{cmd}"')

def tmux_send_cmd4(session_name, cmd: str, window_name = "RealDist", enter = True, wait=0):
    for i in range(4):
        tmux_send_cmd(session_name, i, cmd, window_name, enter)
    if wait > 0:
        print(f"sleep {wait};")

def connect_slurm_node(session_name, window_name = "RealDist"):
    assert len(SLURM_CMDS) == 4, "SLURM_CMDS should have 4 commands because we have 4 panes in window"
    print("\n# connect to slurm node")
    print("sleep 3") # wait for tmux to be ready
    for i, cmd in enumerate(SLURM_CMDS):
        tmux_send_cmd(session_name, i, cmd, window_name=window_name)
    print("sleep 5") # wait for slurm node to be ready

def collect_nic(party: int, folder: str, cmd: str) -> str:
    c = f'ifconfig eno1 > {folder}/begin{party}.txt; {cmd}; '
    c += f'ifconfig eno1 > {folder}/end{party}.txt'
    return c

def collect_log(party: int, folder: str, cmd: str) -> str:
    c = f'{cmd} > {folder}/party{party}.txt 2>&1'
    return c

def process_cvfl(session_name, window_name = "RealDist", rounds=0, datasets=[]):
    tmux_send_cmd4(session_name, cmd=f"conda activate cvfl", wait=2)
    tmux_send_cmd4(session_name, cmd=f"cd ~/VertiBenchGH/src/algorithm/cvfl")
    tmux_send_cmd4(session_name, cmd=f"clear; pwd; echo 'C-VFL'")
    print("")
    sleep = {
        'covtype': 35 * 60,
        'msd': 30 * 60,
        'gisette': 5 * 60,
        'realsim': 30 * 60,
        'epsilon': 35 * 60,
        'letter': 5 * 60,
        'radar': 20 * 60
    }
    for i in range(rounds):
        for dataset in datasets:
            folder = f"~/{session_name}/cvfl_{dataset}_round_{i}"
            print(f"# Running C-VFL on {dataset}, round {i}")
            base_cmd = f"quant_radar_dist.py -d {dataset} -p 4 -sp imp -w 0.1 -s 0 -g 0 --b 512 --local_epochs 1 --epochs 50 --lr 0.0001 --quant_level 4 --vecdim 1 --comp topk log/comm"
            host_cmd = f"{ENVS} torchrun --nproc_per_node=1 --nnodes=5 --node_rank=0 {TR_ARGS} {base_cmd}"
            host_cmd = collect_log(99, folder, host_cmd) + " &"
            host_cmd = f'mkdir -p {folder}; {host_cmd}'
            tmux_send_cmd(session_name, 0, host_cmd, window_name=window_name)

            for party_id in range(0, 4):
                party_cmd = f"{ENVS} torchrun --nproc_per_node=1 --nnodes=5 --node_rank={party_id+1} {TR_ARGS} {base_cmd}"
                party_cmd = collect_log(party_id, folder, party_cmd)
                party_cmd = collect_nic(party_id, folder, party_cmd)
                party_cmd = f'mkdir -p {folder}; {party_cmd}'
                tmux_send_cmd(session_name, party_id, party_cmd, window_name=window_name)
            print(f'sleep {sleep[dataset]}')
            print('')

def process_fedtree(session_name, window_name = "RealDist", rounds=0, datasets=[]):
    tmux_send_cmd4(session_name, cmd=f"conda activate cvfl", wait=2)
    tmux_send_cmd4(session_name, cmd=f"cd ~/VertiBenchGH")
    tmux_send_cmd4(session_name, cmd=f"clear; pwd; echo 'FedTree'")
    print("")
    sleep = {
        'covtype': 20 * 60,
        'msd': 10 * 60,
        'gisette': 15 * 60,
        'realsim': 60 * 60, # <<== not sure how many minutes it will take, so set it to 60 minutes
        'epsilon': 60 * 60, # <<== not sure how many minutes it will take, so set it to 60 minutes
        'letter': 60 * 60, # <<== not sure how many minutes it will take, so set it to 60 minutes
        'radar': 20 * 60
    }
    for i in range(rounds):
        for dataset in datasets:
            folder = f"~/{session_name}/fedtree_{dataset}_round_{i}"
            print(f"# Running FedTree on {dataset}, round {i}")

            for party_id in range(0, 4):
                if party_id == 0:
                    wait = ""
                else:
                    wait = "sleep 3; " # sleep because we need to wait for the gRPC server to be ready. (a FedTree bug)

                party_cmd = f"{wait}{ENVS} python src/algorithm/DistFedTree.py -d {dataset} -c {CLASSES[dataset]} -p 4 -sp imp -w 0.1 --ip_addr={TR_ADDR} --party={party_id}"
                party_cmd = collect_log(party_id, folder, party_cmd)
                party_cmd = collect_nic(party_id, folder, party_cmd)
                party_cmd = f'mkdir -p {folder}; {party_cmd}'
                tmux_send_cmd(session_name, party_id, party_cmd, window_name=window_name)
            print(f'sleep {sleep[dataset]}')
            print('')

def process_gal(session_name, window_name = "RealDist", rounds=0, datasets=[]):
    tmux_send_cmd4(session_name, cmd=f"conda activate gal", wait=2)
    tmux_send_cmd4(session_name, cmd=f"cd /data/junyi/gal/src")
    tmux_send_cmd4(session_name, cmd=f"clear; pwd; echo 'GAL'")
    print("")
    dataname = {
        'covtype': 'CovType',
        'msd': 'MSD',
        'gisette': 'Gisette',
        'realsim': 'Realsim',
        'epsilon': 'Epsilon',
        'letter': 'Letter',
        'radar': 'Radar'
    }
    sleep = {
        'covtype': 20 * 60,
        'msd': 10 * 60,
        'gisette': 10 * 60,
        'realsim': 15 * 60,
        'epsilon': 25 * 60,
        'letter': 5 * 60,
        'radar': 10 * 60
    }
    for i in range(rounds):
        for dataset in datasets:
            folder = f"~/{session_name}/gal_{dataset}_round_{i}"
            print(f"# Running GAL on {dataset}, round {i}")

            for party_id in range(0, 4):
                num_clients = 4
                local_epoch = 1
                total_epoch = 50
                if dataset in ["CIFAR10_VB", "MNIST_VB"]:
                    model = 'resnet18_vb'
                    control = f'{num_clients}_stack_{local_epoch}_{total_epoch}_search_0'
                    raise Exception("AMD Cluster does not support resnet18, weird...")
                elif dataset == 'MSD':
                    model = 'linear'
                    control = f'{num_clients}_stack_{local_epoch}_5_search_0'
                else:
                    model = 'classifier'
                    control = f'{num_clients}_stack_{local_epoch}_{total_epoch}_search_0'
                
                party_cmd = f"{ENVS} BATCH_SIZE=512 LR=0.01 torchrun --nproc_per_node=1 --nnodes=4 --node_rank={party_id} {TR_ARGS} train_model_assist_distributed.py --data_name {dataname[dataset]} --model_name {model} --control_name {control} --init_seed 0 --splitter imp --weight 0.1 --dataseed 0"
                party_cmd = collect_log(party_id, folder, party_cmd)
                party_cmd = collect_nic(party_id, folder, party_cmd)
                party_cmd = f'mkdir -p {folder}; {party_cmd}'
                tmux_send_cmd(session_name, party_id, party_cmd, window_name=window_name)
            print(f'sleep {sleep[dataset]}')
            print('')

def process_splitnn(session_name, window_name = "RealDist", rounds=0, datasets=[]):
    tmux_send_cmd4(session_name, cmd=f"conda activate cvfl", wait=2)
    tmux_send_cmd4(session_name, cmd=f"cd ~/VertiBenchGH")
    tmux_send_cmd4(session_name, cmd=f"clear; pwd; echo 'SplitNN'")
    print("")
    sleep = {
        'covtype': 30 * 60,
        'msd': 30 * 60,
        'gisette': 5 * 60,
        'realsim': 15 * 60,
        'epsilon': 30 * 60,
        'letter': 5 * 60,
        'radar': 20 * 60
    }
    for i in range(rounds):
        for dataset in datasets:
            folder = f"~/{session_name}/splitnn_{dataset}_round_{i}"
            print(f"# Running SplitNN on {dataset}, round {i}")
            for party_id in range(0, 4):
                party_cmd = f"{ENVS} torchrun --nproc_per_node=1 --nnodes=4 --node_rank={party_id} {TR_ARGS} src/algorithm/DistSplitNN.py -d {dataset} -c {CLASSES[dataset]} -m {METRICS[dataset]} -p 4 -sp imp -w 0.1 -s 0 -g 0"
                party_cmd = collect_log(party_id, folder, party_cmd)
                party_cmd = collect_nic(party_id, folder, party_cmd)
                party_cmd = f'mkdir -p {folder}; {party_cmd}'
                tmux_send_cmd(session_name, party_id, party_cmd, window_name=window_name)
            print(f'sleep {sleep[dataset]}')
            print('')

def process_measure(session_name, window_name = "RealDist", rounds=0):
    tmux_send_cmd4(session_name, cmd=f"conda activate cvfl", wait=2)
    tmux_send_cmd4(session_name, cmd=f"cd ~/VertiBenchGH")
    tmux_send_cmd4(session_name, cmd=f"clear; pwd; echo 'Measure'")
    print("")
    for i in range(rounds):
        folder = f"~/{session_name}/measure_round_{i}"
        print(f"# Running background traffic measurement round {i}")
        for party_id in range(0, 4):
            party_cmd = f"{ENVS} sleep 60"
            party_cmd = collect_log(party_id, folder, party_cmd)
            party_cmd = collect_nic(party_id, folder, party_cmd)
            party_cmd = f'mkdir -p {folder}; {party_cmd}'
            tmux_send_cmd(session_name, party_id, party_cmd, window_name=window_name)
        print(f'sleep 1')
        print('')


def main():
    parser = argparse.ArgumentParser(description="Real Distributed Script Generator")
    parser.add_argument('-a', '--algorithm', help="Algorithms: cvfl, fedtree, gal, splitnn", type=str, required=True)
    parser.add_argument('-r', '--rounds', help="Number of rounds. -r 1", type=int, required=True)
    parser.add_argument('-d', '--datasets', help="What datasets to run. -d covtype, realsim, epsilon, letter, gisette, radar, msd",nargs='+', type=str, required=True)
    parser.add_argument('-s', '--session-name', help="Tmux and experiment shares the same session name. -s test", type=str, required=True)
    parser.add_argument('-m', '--measure', help="Measure the background network traffic", action='store_true', default=False)
    args = parser.parse_args()
    

    tmux_kill_session(args.session_name)
    tmux_create_session(args.session_name)
    tmux_create_4windows(args.session_name)
    connect_slurm_node(args.session_name)
    
    if args.measure:
        process_measure(session_name=args.session_name, rounds=args.rounds, datasets=args.datasets)
        return
    if args.algorithm == 'cvfl':
        process_cvfl(session_name=args.session_name, rounds=args.rounds, datasets=args.datasets)
    elif args.algorithm == 'fedtree':
        process_fedtree(session_name=args.session_name, rounds=args.rounds, datasets=args.datasets)
    elif args.algorithm == 'gal':
        process_gal(session_name=args.session_name, rounds=args.rounds, datasets=args.datasets)
    elif args.algorithm == 'splitnn':
        process_splitnn(session_name=args.session_name, rounds=args.rounds, datasets=args.datasets)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")


if __name__ == "__main__":
    main()

    # Reminder:
    #   1. This script will print out the commands to run in a *head server*
    #   2. It will create a tmux session and connect to 4 different *client servers*
    #   3. Then it will send the commands to *client servers*, wait for the task finish
    #   4. If the task is finished and you don't want to sleep anymore, you can kill the "sleep" process in the *head server*
    #      the command is "ps -ef | grep "[s]leep" | grep junyi | awk '{print $2}' | xargs kill -9 "

    # Usage Example:
    # `python run-real-dist.py -r 5 -d covtype msd gisette realsim epsilon letter radar -s real-dist -a fedtree > r5_fedtree.sh`
    # `python run-real-dist.py -r 5 -d covtype msd gisette realsim epsilon letter radar -s real-dist -a cvfl > r5_cvfl.sh`
    # `python run-real-dist.py -r 5 -d covtype msd gisette realsim epsilon letter radar -s real-dist -a gal > r5_gal.sh`
    # `python run-real-dist.py -r 5 -d covtype msd gisette realsim epsilon letter radar -s real-dist -a splitnn > r5_splitnn.sh`
    # `python run-real-dist.py -r 5 -a NA -d NA -s background-measurement -m`
    
    # After generating the script, you can run it in the head server
    # `bash r5_fedtree.sh`
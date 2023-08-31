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


def get_commands(folder, lr, dataseed, dataset, model, control, batch_size, scale_test, communication_test, real_dataset):
    if communication_test:
        commands = [
            f'BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 0.1 --dataseed 0 > {folder}/imp_0.1_seed0_communication_test.log',
        ]
        return commands
    
    if scale_test:
        commands = [
            f'BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 1.0 --dataseed {dataseed} > {folder}/imp_1.0.log',
        ]
        return commands
    
    if real_dataset:
        commands = [
            f'BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --dataseed {dataseed} > {folder}/real.log',
        ]
        return commands

    commands = [
        f'BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 0.0 --dataseed {dataseed} > {folder}/corr_0.0.log',
        f'BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 0.3 --dataseed {dataseed} > {folder}/corr_0.3.log',
        f'BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 0.6 --dataseed {dataseed} > {folder}/corr_0.6.log',
        f'BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 1.0 --dataseed {dataseed} > {folder}/corr_1.0.log',

        f'BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 0.1 --dataseed {dataseed} > {folder}/imp_0.1.log',
        f'BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 1.0 --dataseed {dataseed} > {folder}/imp_1.0.log',
        f'BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 10.0 --dataseed {dataseed} > {folder}/imp_10.0.log',
        f'BATCH_SIZE={batch_size} LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 100.0 --dataseed {dataseed} > {folder}/imp_100.0.log',
    ]
    return commands


def get_args():
    parser = argparse.ArgumentParser(description="GAL running script")  # 脚本描述
    parser.add_argument('-s', '--seeds', help="Random seed for dataset and GAL. You can specify multiple seeds to run. -s 0 1 2 3 4", nargs='+', type=int, required=True)
    parser.add_argument('-c', '--clients', help="Number of clients. -c 4", type=int, required=True)
    parser.add_argument('-g', '--gpus', help="How many gpus to use. -g 1 2 3 4", nargs='+', type=int, required=True)
    parser.add_argument('-t', '--ntask', help="How many task you want to run on each gpu", type=int, required=True)
    parser.add_argument('-d', '--datasets', help="What datasets to run. -d MSD Gisette",nargs='+', type=str, required=True)
    parser.add_argument('-lr', '--learning-rates', help="Learning rate. You can specify multiple lr to run. -lr 0.1 0.01 0.001",nargs='+', type=str, required=True)
    parser.add_argument('-l', '--local-epoch', help="Local epochs. -l 20", type=int, required=True)
    parser.add_argument('-e', '--global-epoch', help="Total epochs. -e 20", type=int, required=True)
    parser.add_argument('-b', '--batch-size', help="Batch size. -b 512", type=int, required=True)
    parser.add_argument('-a', '--scale-test', type=bool, default=False, help="Running Scale test")
    parser.add_argument('-m', '--communication-test', type=bool, default=False, help="Running Communication test")
    parser.add_argument('-r', '--real-dataset', type=bool, default=False, help="Running Real dataset")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    gpus = args.gpus # number of gpus
    num_tasks = args.ntask
    local_epoch = args.local_epoch
    total_epoch = args.global_epoch
    batch_size  = args.batch_size
    num_clients = args.clients
    scale_test = args.scale_test
    communication_test = args.communication_test
    real_dataset = args.real_dataset
    gpuid_queue = multiprocessing.Manager().Queue()
    pool = multiprocessing.Pool(processes=len(gpus) * num_tasks)
    for i in gpus:
        for j in range(num_tasks):
            gpuid_queue.put(i) # available gpu ids

    for lr in args.learning_rates:
        for seed in args.seeds:
            for dataset in args.datasets:

                if dataset in ["CIFAR10_VB", "MNIST_VB"]:
                    model = 'resnet18_vb'
                    control = f'{num_clients}_stack_{local_epoch}_{total_epoch}_search_0'
                elif dataset == 'MSD':
                    model = 'linear'
                    control = f'{num_clients}_stack_{local_epoch}_5_search_0'
                else:
                    model = 'classifier'
                    control = f'{num_clients}_stack_{local_epoch}_{total_epoch}_search_0'
                
                tt = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S")
                if scale_test:
                    folder = f'./results_scale_test{tt}/{dataset}_client{num_clients}_lr{lr}_seed{seed}_local{local_epoch}_global{total_epoch}_bs{batch_size}'
                elif communication_test:
                    folder = f'./results_communication_test{tt}/{dataset}_client{num_clients}_lr{lr}_seed{seed}_local{local_epoch}_global{total_epoch}_bs{batch_size}'
                else:
                    folder = f'./results_test_{tt}/{dataset}_client{num_clients}_lr{lr}_seed{seed}_local{local_epoch}_global{total_epoch}_bs{batch_size}'

                os.system(f'mkdir -p {folder}')
                commands = get_commands(folder, lr, seed, dataset, model, control, batch_size, scale_test, communication_test, real_dataset)
                for cmd in commands:
                    pool.apply_async(process_wrapper, (gpuid_queue, cmd, seed))
    
    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "Waiting for all subprocesses done...")
    pool.close()
    pool.join()

# python run.py -s 0 1 2 3 4 -g 0 1 2 3 -t 4 -d CovType MSD Gisette Realsim Epsilon Letter Radar -lr 0.01

"""
rebuttal wide vehicle
python run.py --seeds 0 1 2 3 4 --gpus 0 1 2 3 4 5 6 7 --clients 2 --ntask 2 -d Vehicle -lr 0.01 -l 20 -e 50 -b 512 --real-dataset=True
python run.py --seeds 0 1 2 3 4 --gpus 0 1 2 3 4 5 6 7 --clients 5 --ntask 2 -d Wide -lr 0.01 -l 20 -e 50 -b 512 --real-dataset=True


"""


"""
rebuttal cifar mnist
python run.py --seeds 0 1 2 3 4 --gpus 0 1 2 3 4 5 6 7 --clients 4 --ntask 2 -d MNIST_VB CIFAR10_VB -lr 0.01 -l 20 -e 200 -b 512

"""

"""
rebuttal
python run.py --seeds 0 --gpus 0 1 2 3 --clients 4 --ntask 2 -d CovType MSD Gisette Realsim Epsilon Letter Radar -lr 0.01 -l 20 -e 200 -b 512 --communication-test=True &
🟡 CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name MSD --model_name linear --control_name 4_stack_20_100_search_0 --init_seed 0 --splitter imp --weight 0.1 --dataseed 0 > ./results_communication_test/MSD_client4_lr0.01_seed0_local20_global200_bs512/imp_0.1_seed0_communication_test.log_final_repeat0.txt 2>&1 &
🟡 CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name CovType --model_name classifier --control_name 4_stack_20_200_search_0 --init_seed 0 --splitter imp --weight 0.1 --dataseed 0 > ./results_communication_test/CovType_client4_lr0.01_seed0_local20_global200_bs512/imp_0.1_seed0_communication_test.log_final_repeat0.txt 2>&1 &
🟡 CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 4_stack_20_200_search_0 --init_seed 0 --splitter imp --weight 0.1 --dataseed 0 > ./results_communication_test/Realsim_client4_lr0.01_seed0_local20_global200_bs512/imp_0.1_seed0_communication_test.log_final_repeat0.txt 2>&1 &
🟡 CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Epsilon --model_name classifier --control_name 4_stack_20_200_search_0 --init_seed 0 --splitter imp --weight 0.1 --dataseed 0 > ./results_communication_test/Epsilon_client4_lr0.01_seed0_local20_global200_bs512/imp_0.1_seed0_communication_test.log_final_repeat0.txt 2>&1 &
🟡 CUDA_VISIBLE_DEVICES=3 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Radar --model_name classifier --control_name 4_stack_20_200_search_0 --init_seed 0 --splitter imp --weight 0.1 --dataseed 0 > ./results_communication_test/Radar_client4_lr0.01_seed0_local20_global200_bs512/imp_0.1_seed0_communication_test.log_final_repeat0.txt 2>&1 &
🟢 CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Gisette --model_name classifier --control_name 4_stack_20_200_search_0 --init_seed 0 --splitter imp --weight 0.1 --dataseed 0 > ./results_communication_test/Gisette_client4_lr0.01_seed0_local20_global200_bs512/imp_0.1_seed0_communication_test.log_final_repeat0.txt 2>&1 &
🟢 CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Letter --model_name classifier --control_name 4_stack_20_200_search_0 --init_seed 0 --splitter imp --weight 0.1 --dataseed 0 > ./results_communication_test/Letter_client4_lr0.01_seed0_local20_global200_bs512/imp_0.1_seed0_communication_test.log_final_repeat0.txt 2>&1 &
"""


"""
✅ python run.py --seeds 0 1 2 3 4 --gpus 0 --clients 2 --ntask 1 -d Gisette -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &
✅ python run.py --seeds 0 1 2 3 4 --gpus 0 --clients 8 --ntask 1 -d Gisette -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &
✅ python run.py --seeds 0 1 2 3 4 --gpus 0 --clients 32 --ntask 1 -d Gisette -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &
✅ python run.py --seeds 0 1 2 3 4 --gpus 0 --clients 128 --ntask 2 -d Gisette -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &
✅ python run.py --seeds 0 1 2 3 4 --gpus 1 --clients 512 --ntask 4 -d Gisette -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &

🟣 python run.py --seeds 0 1 2 3 4 --gpus 4 --clients 2048 --ntask 1 -d Gisette -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &

✅ python run.py --seeds 0 1 2 3 4 --gpus 0 --clients 2 --ntask 2 -d Realsim -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &
✅ python run.py --seeds 0 1 2 3 4 --gpus 0 --clients 8 --ntask 2 -d Realsim -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &
🟣 python run.py --seeds 0 1 2 3 4 --gpus 3 --clients 32 --ntask 3 -d Realsim -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &
🟣 python run.py --seeds 0 1 2 3 4 --gpus 0 --clients 128 --ntask 3 -d Realsim -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &
🟡 python run.py --seeds 0 1 2 3 4 --gpus 1 --clients 512 --ntask 5 -d Realsim -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &
🔴 python run.py --seeds 0 1 2 3 4 --gpus 3 --clients 2048 --ntask 5 -d Realsim -lr 0.01 -l 20 -e 20 -b 512 --scale-test=True &
"""

"""
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 512_stack_20_20_search_0 --init_seed 0 --splitter imp --weight 1.0 --dataseed 0 > ./results_scale_test/Realsim_client512_lr0.01_seed0_local20_global20_bs512/imp_1.0.log_final_repeat0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 512_stack_20_20_search_0 --init_seed 1 --splitter imp --weight 1.0 --dataseed 1 > ./results_scale_test/Realsim_client512_lr0.01_seed1_local20_global20_bs512/imp_1.0.log_final_repeat1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 512_stack_20_20_search_0 --init_seed 2 --splitter imp --weight 1.0 --dataseed 2 > ./results_scale_test/Realsim_client512_lr0.01_seed2_local20_global20_bs512/imp_1.0.log_final_repeat2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 512_stack_20_20_search_0 --init_seed 3 --splitter imp --weight 1.0 --dataseed 3 > ./results_scale_test/Realsim_client512_lr0.01_seed3_local20_global20_bs512/imp_1.0.log_final_repeat3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 512_stack_20_20_search_0 --init_seed 4 --splitter imp --weight 1.0 --dataseed 4 > ./results_scale_test/Realsim_client512_lr0.01_seed4_local20_global20_bs512/imp_1.0.log_final_repeat4.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Gisette --model_name classifier --control_name 2048_stack_20_20_search_0 --init_seed 0 --splitter imp --weight 1.0 --dataseed 0 > ./results_scale_test/Gisette_client2048_lr0.01_seed0_local20_global20_bs512/imp_1.0.log_final_repeat0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Gisette --model_name classifier --control_name 2048_stack_20_20_search_0 --init_seed 1 --splitter imp --weight 1.0 --dataseed 1 > ./results_scale_test/Gisette_client2048_lr0.01_seed1_local20_global20_bs512/imp_1.0.log_final_repeat1.txt 2>&1 &
🟡 CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Gisette --model_name classifier --control_name 2048_stack_20_20_search_0 --init_seed 2 --splitter imp --weight 1.0 --dataseed 2 > ./results_scale_test/Gisette_client2048_lr0.01_seed2_local20_global20_bs512/imp_1.0.log_final_repeat2.txt 2>&1 &
✅ CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Gisette --model_name classifier --control_name 2048_stack_20_20_search_0 --init_seed 3 --splitter imp --weight 1.0 --dataseed 3 > ./results_scale_test/Gisette_client2048_lr0.01_seed3_local20_global20_bs512/imp_1.0.log_final_repeat3.txt 2>&1 &
✅ CUDA_VISIBLE_DEVICES=3 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Gisette --model_name classifier --control_name 2048_stack_20_20_search_0 --init_seed 4 --splitter imp --weight 1.0 --dataseed 4 > ./results_scale_test/Gisette_client2048_lr0.01_seed4_local20_global20_bs512/imp_1.0.log_final_repeat4.txt 2>&1 &

正在跑 🏃 ⬆️

✅ CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 32_stack_20_20_search_0 --init_seed 3 --splitter imp --weight 1.0 --dataseed 3 > ./results_scale_test/Realsim_client32_lr0.01_seed3_local20_global20_bs512/imp_1.0.log_final_repeat3.txt 2>&1 &
✅ CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 32_stack_20_20_search_0 --init_seed 2 --splitter imp --weight 1.0 --dataseed 2 > ./results_scale_test/Realsim_client32_lr0.01_seed2_local20_global20_bs512/imp_1.0.log_final_repeat2.txt 2>&1 &
✅ CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 32_stack_20_20_search_0 --init_seed 4 --splitter imp --weight 1.0 --dataseed 4 > ./results_scale_test/Realsim_client32_lr0.01_seed4_local20_global20_bs512/imp_1.0.log_final_repeat4.txt 2>&1 &

✅ CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 128_stack_20_20_search_0 --init_seed 0 --splitter imp --weight 1.0 --dataseed 0 > ./results_scale_test/Realsim_client128_lr0.01_seed0_local20_global20_bs512/imp_1.0.log_final_repeat0.txt 2>&1 &
🏃 CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 128_stack_20_20_search_0 --init_seed 1 --splitter imp --weight 1.0 --dataseed 1 > ./results_scale_test/Realsim_client128_lr0.01_seed1_local20_global20_bs512/imp_1.0.log_final_repeat1_resume.txt 2>&1 &
🏃 CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 128_stack_20_20_search_0 --init_seed 2 --splitter imp --weight 1.0 --dataseed 2 > ./results_scale_test/Realsim_client128_lr0.01_seed2_local20_global20_bs512/imp_1.0.log_final_repeat2_resume.txt 2>&1 &
🏃 CUDA_VISIBLE_DEVICES=3 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 128_stack_20_20_search_0 --init_seed 3 --splitter imp --weight 1.0 --dataseed 3 > ./results_scale_test/Realsim_client128_lr0.01_seed3_local20_global20_bs512/imp_1.0.log_final_repeat3_resume.txt 2>&1 &
🏃 CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 128_stack_20_20_search_0 --init_seed 4 --splitter imp --weight 1.0 --dataseed 4 > ./results_scale_test/Realsim_client128_lr0.01_seed4_local20_global20_bs512/imp_1.0.log_final_repeat4_resume.txt 2>&1 &

🏃 CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 512_stack_20_20_search_0 --init_seed 0 --splitter imp --weight 1.0 --dataseed 0 > ./results_scale_test/Realsim_client512_lr0.01_seed0_local20_global20_bs512/imp_1.0.log_final_repeat0_resume.txt 2>&1 &
🏃 CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 512_stack_20_20_search_0 --init_seed 1 --splitter imp --weight 1.0 --dataseed 1 > ./results_scale_test/Realsim_client512_lr0.01_seed1_local20_global20_bs512/imp_1.0.log_final_repeat1_resume.txt 2>&1 &
🏃 CUDA_VISIBLE_DEVICES=2 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 512_stack_20_20_search_0 --init_seed 2 --splitter imp --weight 1.0 --dataseed 2 > ./results_scale_test/Realsim_client512_lr0.01_seed2_local20_global20_bs512/imp_1.0.log_final_repeat2_resume.txt 2>&1 &
🏃 CUDA_VISIBLE_DEVICES=3 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 512_stack_20_20_search_0 --init_seed 3 --splitter imp --weight 1.0 --dataseed 3 > ./results_scale_test/Realsim_client512_lr0.01_seed3_local20_global20_bs512/imp_1.0.log_final_repeat3_resume.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=512 LR=0.01 python train_model_assist.py --data_name Realsim --model_name classifier --control_name 512_stack_20_20_search_0 --init_seed 4 --splitter imp --weight 1.0 --dataseed 4 > ./results_scale_test/Realsim_client512_lr0.01_seed4_local20_global20_bs512/imp_1.0.log_final_repeat4_resume.txt 2>&1 &
"""

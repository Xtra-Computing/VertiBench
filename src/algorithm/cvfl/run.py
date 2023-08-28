import multiprocessing
import os
import subprocess
import time
from queue import Empty


def process_wrapper(gpuid_queue, command, times):
    while True:
        try:
            gpu_idx = gpuid_queue.get(block=True, timeout=None)
        except Empty:
            break
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        
        # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_idx} " + command + f'_dataseed{times}.txt 2>&1'
        
        subprocess.call(cmd, shell=True)
        print("Finished", cmd)
        gpuid_queue.put(gpu_idx)
        break
    gpuid_queue.close()


def get_commands(dataset, dataseed, lr='0.001'):
    commands = [
        f'python quant_radar.py 10class/classes/ --num_clients 4 --seed {dataseed} --b 512 --local_epochs 10 --epochs 200 --lr {lr} --quant_level 4 --vecdim 1 --comp topk --dataset {dataset} --splitter corr --weight 0.0 --dataseed {dataseed} > ./results/{dataset}_corr_0.0',
        f'python quant_radar.py 10class/classes/ --num_clients 4 --seed {dataseed} --b 512 --local_epochs 10 --epochs 200 --lr {lr} --quant_level 4 --vecdim 1 --comp topk --dataset {dataset} --splitter corr --weight 0.3 --dataseed {dataseed} > ./results/{dataset}_corr_0.3',
        f'python quant_radar.py 10class/classes/ --num_clients 4 --seed {dataseed} --b 512 --local_epochs 10 --epochs 200 --lr {lr} --quant_level 4 --vecdim 1 --comp topk --dataset {dataset} --splitter corr --weight 0.6 --dataseed {dataseed} > ./results/{dataset}_corr_0.6',
        f'python quant_radar.py 10class/classes/ --num_clients 4 --seed {dataseed} --b 512 --local_epochs 10 --epochs 200 --lr {lr} --quant_level 4 --vecdim 1 --comp topk --dataset {dataset} --splitter corr --weight 1.0 --dataseed {dataseed} > ./results/{dataset}_corr_1.0',
        
        f'python quant_radar.py 10class/classes/ --num_clients 4 --seed {dataseed} --b 512 --local_epochs 10 --epochs 200 --lr {lr} --quant_level 4 --vecdim 1 --comp topk --dataset {dataset} --splitter imp --weight 0.1 --dataseed {dataseed} > ./results/{dataset}_imp_0.1',
        f'python quant_radar.py 10class/classes/ --num_clients 4 --seed {dataseed} --b 512 --local_epochs 10 --epochs 200 --lr {lr} --quant_level 4 --vecdim 1 --comp topk --dataset {dataset} --splitter imp --weight 1.0 --dataseed {dataseed} > ./results/{dataset}_imp_1.0',
        f'python quant_radar.py 10class/classes/ --num_clients 4 --seed {dataseed} --b 512 --local_epochs 10 --epochs 200 --lr {lr} --quant_level 4 --vecdim 1 --comp topk --dataset {dataset} --splitter imp --weight 10.0 --dataseed {dataseed} > ./results/{dataset}_imp_10.0',
        f'python quant_radar.py 10class/classes/ --num_clients 4 --seed {dataseed} --b 512 --local_epochs 10 --epochs 200 --lr {lr} --quant_level 4 --vecdim 1 --comp topk --dataset {dataset} --splitter imp --weight 100.0 --dataseed {dataseed} > ./results/{dataset}_imp_100.0',
    ]
    return commands

# datasets = [
#     ('CovType','classifier', '4_stack_50_10_search_0'), multicls
#     ('MSD','linear', '4_stack_50_10_search_0'), 
#     ('Gisette','classifier', '4_stack_50_10_search_0'), 2cls
#     ('Realsim','classifier', '4_stack_50_10_search_0'), 2cls
#     ('Epsilon','classifier', '4_stack_50_10_search_0'), 2cls
#     ('Letter', 'classifier', '4_stack_50_10_search_0'), 26cls
#     ('Radar', 'classifier', '4_stack_50_10_search_0'), 7cls
# ]

if __name__ == "__main__":
    num_gpus = 4
    gpuid_queue = multiprocessing.Manager().Queue()
    pool = multiprocessing.Pool(processes=16)

    for i in range(num_gpus):
        gpuid_queue.put(i) # available gpu ids
        gpuid_queue.put(i)
        gpuid_queue.put(i) 

    for times in range(0, 5): # covtype is in run_issue_fix.py
        for ds in [ 'gisette', 'realsim', 'epsilon', 'radar', 'letter', 'msd']:
            if ds == 'letter':
                commands = get_commands(ds, str(times), '0.003')
            else:
                commands = get_commands(ds, str(times), '0.001')

            for cmd in commands:
                pool.apply_async(process_wrapper, (gpuid_queue, cmd, times))
    
    print("Waiting for all subprocesses done...")
    pool.close()
    pool.join()

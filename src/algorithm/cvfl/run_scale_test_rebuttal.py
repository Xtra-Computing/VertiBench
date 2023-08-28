import multiprocessing
import os
import subprocess
import datetime
import pytz
from queue import Empty


def process_wrapper(gpuid_queue, command, times):
    while True:
        try:
            gpu_idx = gpuid_queue.get(block=True, timeout=None)
        except Empty:
            break
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        
        # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_idx} {command}"
        
        print("🟢 Running command: ", cmd)
        subprocess.call(cmd, shell=True)
        
        gpuid_queue.put(gpu_idx)
        break
    gpuid_queue.close()


def get_commands(folder, dataset, dataseed, lr, num_clients, local_epochs, global_epochs, batch_size, splitter, weight):
    # splitter: corr, imp
    # weight(imp): 0.1, 1.0, 10.0, 100.0
    # weight(corr): 0.1, 0.3, 0.6, 1.0
    commands = [
        f'python quant_radar.py 10class/classes/ --num_clients {num_clients} --seed {dataseed} --b {batch_size} '\
        f'--local_epochs {local_epochs} --epochs {global_epochs} --lr {lr} --quant_level 4 --vecdim 1 --comp topk '\
        f'--dataset {dataset} --splitter {splitter} --weight {weight} --dataseed {dataseed} --folder {folder} > '\
        f'{folder}/{dataset}_c{num_clients}_lr{lr}_ds{dataseed}_le{local_epochs}_ge{global_epochs}_bs{batch_size}_{splitter}_{weight}.log 2>&1',
    ] # '

    return commands

if __name__ == "__main__":
    num_gpus = 8
    gpuid_queue = multiprocessing.Manager().Queue()
    pool = multiprocessing.Pool(processes=16)

    for i in range(num_gpus):
        gpuid_queue.put(i) # available gpu ids
        gpuid_queue.put(i)
        gpuid_queue.put(i) 

    # get current time at Asia/Shanghai
    curtime = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S")
    folder = "./results_scaletest_rebuttal_" + curtime
    os.system(f'mkdir -p {folder}')

    # for dataset in [ 'gisette', 'realsim', 'msd', 'covtype', 'epsilon', 'letter', 'radar' ]:
    for dataset in [ 'covtype' ]:
        for num_clients in [4]:
            for seed in [0]:
                commands = get_commands(folder, dataset, seed, '0.0001', num_clients, 10, 200, 512, "imp", '0.1')
                print(commands)
                for cmd in commands:
                    print(cmd)
                    pool.apply_async(process_wrapper, (gpuid_queue, cmd, seed))

    print("Waiting for all subprocesses done...")
    pool.close()
    pool.join()

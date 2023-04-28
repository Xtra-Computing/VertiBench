import subprocess
import time
import argparse

from nvitop import Device, GpuProcess, NA

def get_num_match_processes(device, pattern, user=None):
    processes = device.processes()
    cnt = 0
    for process in processes:
        if pattern in process.command and (user is None or user == process.user):
            cnt += 1
    return cnt

def get_valid_gpu(devices, pattern, user=None):
    """
    Return a list of valid GPU IDs that does not contain any process that matches `pattern`.
    :param devices: list of nvitop.Device
    :param pattern: str. The pattern to match.
    :param user: str or None. If None, all users are considered.
    :return: List[int]
    """
    valid_gpus = []
    for device in devices:
        cnt = get_num_match_processes(device, pattern)
        if cnt == 0:
            valid_gpus.append(device.index)
    return valid_gpus


class Task:
    def __init__(self, cmd=None, gpu=None, process=None, log_path=None, is_main=False):
        """
        A task is a command executed on a GPU.

        Parameters
        ----------
        cmd : str
            A list of command line arguments to run. The first argument is the executable. The last argument should
            not be '&'.
        gpu : int
            GPU ID
        process: subprocess.Popen
            The process object of the task
        is_main : bool
            Whether this task is the computing-intensive task
        """
        self.cmd = cmd
        self.gpu = gpu
        self.process = process
        self.is_main = is_main
        self.log_path = log_path

    @property
    def executable(self):
        return self.cmd is not None and self.gpu is not None

    @property
    def running(self):
        return self.process is not None

def read_tasks_from_bash(bash_path, cmd_prefix='python'):
    """
    Read tasks from a bash file.
    :param bash_path: str
    :return: List[Task]
    """
    with open(bash_path, 'r') as f:
        lines = f.readlines()

    # tasks is a list of Task objects
    tasks = []
    variables = {}
    for line in lines:
        task = Task()
        if line.strip().startswith('#'):
            continue    # skip comments

        # replace variables
        for var_name, var_value in variables.items():
            line = line.replace(f'"${var_name}"', f'{var_value}')

        params = line.strip().split()

        if len(params) == 0:
            continue    # skip empty lines
        elif params[0] == 'wait':
            continue    # skip wait commands
        elif '=' in params[0]:
            var_name, var_value = params[0].split('=')    # variable assignment
            variables[var_name] = var_value
        elif params[0] == cmd_prefix:
            # create a main task object and append to tasks

            # extract '> <log_path>' from the command
            if '>' in params:
                idx = params.index('>')
                task.log_path = params[idx + 1]
                params.pop(idx)
                params.pop(idx)

            task.is_main = True
            # remove all pre-assigned GPUs
            if '-g' in params or '--gpu' in params:
                idx = params.index('-g') if '-g' in params else params.index('--gpu')
                params.pop(idx)
                params.pop(idx)

            # remove previous '&' if exists
            if params[-1] == '&':
                task.cmd = params[:-1]
            else:
                task.cmd = params
            tasks.append(task)
        else:
            # create an auxiliary task object and append to tasks
            if params[-1] == '&':
                task.cmd = params[:-1]
            else:
                task.cmd = params
            task.is_main = False
            tasks.append(task)
    return tasks

def run_tasks_on_gpus(tasks: list, gpu_ids: list, cmd_prefix='python', check_period=1, user=None, check_valid=False,
                      max_task_per_gpu=1):
    """
    Run tasks on GPUs. The tasks are run in the order of `tasks`. Non-main tasks are also run in the order of `tasks`.
    However, they are expected to be finished in a short time. If a non-main task is not finished, the main task will
    not be run.

    Parameters
    ----------
    tasks : List[Task]
        List of tasks to run
    gpu_ids : List[int]
        List of GPU IDs to run tasks on
    cmd_prefix : str
        Command prefix
    check_period : int
        Check period in seconds
    user : str or None
        If None, all users are considered.
    check_valid : bool
        Whether to check if the GPU is valid. If True, a GPU in `gpu_ids` is valid if it does not contain any process
        that matches `cmd_prefix` and `user`. If False, all GPUs in `gpu_ids` are considered valid.
    max_task_per_gpu : int
        Maximum number of tasks to run on a GPU. If a GPU has more than `max_task_per_gpu` tasks running, the main task
        will not be run on that GPU.
    """
    devices = Device.cuda.all()
    if check_valid:
        valid_gpus = [gpu for gpu in gpu_ids if gpu in get_valid_gpu(devices, pattern=cmd_prefix, user=user)]
        print(f"Usable GPUs: {valid_gpus}")
    else:
        valid_gpus = gpu_ids
        print(f"Usable GPUs (unchecked): {valid_gpus}")

    # run tasks
    available_gpus = {gpu: [None for _ in range(max_task_per_gpu)] for gpu in valid_gpus}
    for task in tasks:
        if not task.is_main:
            # run auxiliary tasks and wait for them to finish
            print(f"Executing: {' '.join(task.cmd)}")
            subprocess.run(' '.join(task.cmd), shell=True)
            continue

        assigned_gpu = None
        while assigned_gpu is None:
            # check if any GPU is available
            if len(available_gpus) == 0:
                print("No GPU is available. Exiting...")
                return
            for gpu, processes in available_gpus.items():
                for pos, process in enumerate(processes):
                    if process is None:
                        assigned_gpu = gpu, pos
                        print(f"GPU {gpu} ({pos}) is empty.")
                        break
                    elif process.poll() is not None:
                        process = None
                        assigned_gpu = gpu, pos
                        print(f"GPU {gpu} ({pos}) is available.")
                        break
                else:
                    continue
                break
            time.sleep(check_period)

        # Found an available GPU, run the main task as a subprocess on the GPU
        task.gpu, pos = assigned_gpu
        # since '>' should be removed before, add '-g' and the GPU ID directly to the end of the command
        task.cmd += ['-g', str(task.gpu)]
        with open(task.log_path, 'w') as log_file:
            task.process = subprocess.Popen(task.cmd, stdout=log_file)
        print(f"Executing on GPU {task.gpu} ({pos}): {' '.join(task.cmd)} > {task.log_path}")
        available_gpus[task.gpu][pos] = task.process




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bash_path', type=str, help='Path to the bash file')
    parser.add_argument('-g', '--gpus', type=int, nargs='+', help='GPU IDs to run tasks on')
    parser.add_argument('-p', '--prefix', type=str, default='python', help='Command prefix')
    parser.add_argument('-c', '--check_period', type=int, default=1, help='Check period in seconds')
    parser.add_argument('-u', '--user', type=str, default=None, help='User name')
    parser.add_argument('-m', '--max_task_per_gpu', type=int, default=1, help='Maximum number of tasks to run on a GPU')
    parser.add_argument('-v', '--check_valid', action='store_true', help='Whether to check if the GPU is valid')
    args = parser.parse_args()

    gpus = [int(gpu) for gpu in args.gpus]
    tasks = read_tasks_from_bash(args.bash_path, cmd_prefix=args.prefix)
    run_tasks_on_gpus(tasks, gpus, cmd_prefix=args.prefix, check_period=args.check_period, user=args.user,
                      max_task_per_gpu=args.max_task_per_gpu, check_valid=args.check_valid)



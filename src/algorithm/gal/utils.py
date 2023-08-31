import collections.abc as container_abcs
import errno
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
import numbers
from itertools import repeat
from torchvision.utils import save_image
from config import cfg
from torch.nn.utils.rnn import pad_sequence
import datetime, pytz

def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=2, pad_value=0, range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif isinstance(input, numbers.Number):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset):
    cfg['target_size'] = dataset['train'].target_size
    cfg['data_size'] = {split: len(dataset[split]) for split in dataset}
    data_shape = {'Blob': [10], 'Iris': [4], 'Diabetes': [10], 'BostonHousing': [13], 'Wine': [13],
                  'BreastCancer': [30], 'QSAR': [41], 'MNIST': [1, 28, 28], 'CIFAR10': [3, 32, 32],
                  'ModelNet40': [3, 32, 32, 12], 'ShapeNet55': [3, 32, 32, 12], 'MIMICL': [22], 'MIMICM': [22], }
    
    if cfg['data_name'] in ['CovType','MSD','Higgs','Gisette','Realsim','Epsilon','Letter','Radar', 'Wide', 'Vehicle']:
        cfg['data_shape'] = [np.sum(dataset['train'].partitions)]
    elif cfg['data_name'] == "MNIST_VB": # vertibench
        cfg['data_shape'] = dataset['train'].X.shape[1:]
    elif cfg['data_name'] == "CIFAR10_VB": # vertibench
        cfg['data_shape'] = dataset['train'].X.shape[1:]
    else:
        cfg['data_shape'] = data_shape[cfg['data_name']]
        
    if cfg['data_name'] in ['MIMICL', 'MIMICM']:
        cfg['data_length'] = {split: dataset[split].length for split in dataset}
    return


def process_control():
    
    cfg['linear'] = {}
    cfg['classifier'] = {}
    cfg['resnet18_vb'] = {}
    cfg['conv'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['lstm'] = {'hidden_size': 128, 'num_layers': 1}
    cfg['gb'] = {}
    cfg['svm'] = {}
    cfg['gb-svm'] = {}
    cfg['num_users'] = int(cfg['control']['num_users'])
    cfg['assist_mode'] = cfg['control']['assist_mode']
    cfg['local_epoch'] = int(cfg['control']['local_epoch']) if cfg['control']['local_epoch'] != 'none' else 'none'
    cfg['global_epoch'] = int(cfg['control']['global_epoch']) if cfg['control'][
                                                                     'global_epoch'] != 'none' else 'none'
    cfg['assist_rate_mode'] = cfg['control']['assist_rate_mode']
    if 'noise' in cfg['control']:
        if cfg['control']['noise'] not in ['none', 'data']:
            cfg['noise'] = float(cfg['control']['noise'])
        else:
            cfg['noise'] = cfg['control']['noise']
    else:
        cfg['noise'] = 'none'
    cfg['active_rate'] = 0.1
    if 'al' in cfg['control']:
        cfg['al'] = cfg['control']['al']
    if 'rl' in cfg['control'] and cfg['control']['rl'] != 'none':
        rl_list = cfg['control']['rl'].split('-')
        num_rl = cfg['num_users'] // len(rl_list)
        rm_rl = cfg['num_users'] - num_rl * len(rl_list)
        cfg['rl'] = []
        for i in range(len(rl_list)):
            cfg['rl'].extend([rl_list[i] for _ in range(num_rl)])
            if i == len(rl_list) - 1:
                cfg['rl'].extend([rl_list[i] for _ in range(rm_rl)])
    else:
        if cfg['data_name'] in ['Diabetes', 'BostonHousing', 'MIMICL']:
            cfg['rl'] = ['l1' for _ in range(cfg['num_users'])]
        else:
            cfg['rl'] = ['l2' for _ in range(cfg['num_users'])]
    if 'dl' in cfg['control']:
        cfg['dl'] = cfg['control']['dl']
    if cfg['model_name'] in ['gb', 'svm', 'gb-svm']:
        cfg['ma'] = '1'
    if 'pl' in cfg['control']:
        cfg['pl'] = cfg['control']['pl']
        if cfg['pl'] != 'none':
            pl_list = cfg['pl'].split('-')
            cfg['pl_mode'], cfg['pl_param'] = pl_list[0], float(pl_list[1])
    cfg['noised_organization_id'] = list(range(cfg['num_users'] // 2, cfg['num_users']))
    cfg['assist'] = {}
    cfg['assist']['batch_size'] = {'train': 1024, 'test': 1024}
    cfg['assist']['optimizer_name'] = 'Adam'
    cfg['assist']['lr'] = 1e-1
    cfg['assist']['momentum'] = 0.9
    cfg['assist']['weight_decay'] = 5e-4
    cfg['assist']['num_epochs'] = 100
    cfg['linesearch'] = {}
    cfg['linesearch']['optimizer_name'] = 'LBFGS'
    cfg['linesearch']['lr'] = 1
    cfg['linesearch']['num_epochs'] = 10
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    if model_name in ['linear']:
        cfg[model_name]['optimizer_name'] = 'SGD'
        cfg[model_name]['momentum'] = 0.9
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['batch_size'] = {'train': int(os.environ.get("BATCH_SIZE", "512")), 'test': int(os.environ.get("BATCH_SIZE", "512"))}
        cfg[model_name]['lr'] = float(os.environ.get("LR", "0.00001"))
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'MultiStepLR'
        cfg[model_name]['factor'] = 0.1
        cfg[model_name]['milestones'] = [2,4,6,8,10,12,14,16,18,20]
    elif model_name in ['classifier']:
        cfg[model_name]['optimizer_name'] = os.environ.get("OPTIMIZER", "Adam")
        cfg[model_name]['momentum'] = float(os.environ.get("MOMENTUM", "0.9"))
        cfg[model_name]['weight_decay'] = float(os.environ.get("WEIGHT_DECAY", "5e-4"))
        cfg[model_name]['batch_size'] = {'train': int(os.environ.get("BATCH_SIZE", "512")), 'test': int(os.environ.get("BATCH_SIZE", "512"))}
        cfg[model_name]['lr'] = float(os.environ.get("LR", "0.01"))
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'MultiStepLR'
        cfg[model_name]['factor'] = 0.1
        cfg[model_name]['milestones'] = [50, 100]
    elif model_name in ['resnet18_vb']:
        if cfg['data_name'] in ['MNIST_VB', 'CIFAR10_VB']:
            cfg[model_name]['optimizer_name'] = 'SGD'
            cfg[model_name]['momentum'] = 0.9
            cfg[model_name]['weight_decay'] = 5e-4
            cfg[model_name]['batch_size'] = {'train': 512, 'test': 512}
            cfg[model_name]['lr'] = 1e-1
        else:
            raise ValueError(f'Not valid data name for {model_name}')
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'MultiStepLR'
        cfg[model_name]['factor'] = 0.1
        cfg[model_name]['milestones'] = [50, 100]
    elif model_name in ['conv']:
        if cfg['data_name'] in ['MNIST', 'CIFAR10', "CovType"]:
            cfg[model_name]['optimizer_name'] = 'SGD'
            cfg[model_name]['momentum'] = 0.9
            cfg[model_name]['weight_decay'] = 5e-4
            cfg[model_name]['batch_size'] = {'train': 512, 'test': 512}
            cfg[model_name]['lr'] = 1e-1
        elif cfg['data_name'] in ['ModelNet40', 'ShapeNet55']:
            cfg[model_name]['optimizer_name'] = 'SGD'
            cfg[model_name]['momentum'] = 0.9
            cfg[model_name]['weight_decay'] = 5e-4
            cfg[model_name]['batch_size'] = {'train': 64, 'test': 128}
            cfg[model_name]['lr'] = 1e-1
            torch.set_num_threads(2)
        else:
            raise ValueError('Not valid data name')
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'MultiStepLR'
        cfg[model_name]['factor'] = 0.1
        cfg[model_name]['milestones'] = [50, 100]
    elif model_name in ['lstm']:
        cfg[model_name]['ICD9_embeddings'] = 5893
        cfg[model_name]['optimizer_name'] = 'Adam'
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['batch_size'] = {'train': 8, 'test': 8}
        cfg[model_name]['lr'] = 1e-3
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'None'
    elif model_name in ['gb', 'svm', 'gb-svm']:
        cfg[model_name]['optimizer_name'] = 'SGD'
        cfg[model_name]['momentum'] = 0.9
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['batch_size'] = {'train': 1024, 'test': 1024}
        cfg[model_name]['lr'] = 1e-1
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'MultiStepLR'
        cfg[model_name]['factor'] = 0.1
        cfg[model_name]['milestones'] = [50, 100]
        if model_name == 'gb-svm':
            cfg['gb'] = cfg[model_name]
            cfg['svm'] = cfg[model_name]
    else:
        raise ValueError('Not valid model name')
    cfg['global'] = {}
    cfg['global']['num_epochs'] = cfg['global_epoch']
    cfg['stats'] = make_stats()
    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "Learning rate:", cfg[model_name]['lr'])
    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "cfg:", cfg)
    return


def make_stats():
    stats = {}
    stats_path = './res/stats'
    makedir_exist_ok(stats_path)
    filenames = os.listdir(stats_path)
    for filename in filenames:
        stats_name = os.path.splitext(filename)[0]
        stats[stats_name] = load(os.path.join(stats_path, filename))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def make_optimizer(model, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg[tag]['lr'], weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]['num_epochs']['global'],
                                                         eta_min=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def resume(model_tag, load_tag='checkpoint', verbose=True):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), 'Resume from {}'.format(load_tag))
        result = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
    else:
        print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), 'Not exists model tag: {}, start from scratch'.format(model_tag))
        # exit(0)
        
        from logger import Logger
        last_epoch = 1
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
        result = {'epoch': last_epoch, 'logger': logger}
    if verbose:
        print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), 'Resume from {}'.format(result['epoch']))
    return result


def collate(input):
    for k in input:
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            if k == 'data':
                input[k] = pad_sequence(input['data'], batch_first=True, padding_value=0)
            elif k == 'target':
                input[k] = pad_sequence(input['target'], batch_first=True, padding_value=-65535)
            else:
                input[k] = torch.stack(input[k], 0)
        else:
            input[k] = torch.stack(input[k], 0)
    return input

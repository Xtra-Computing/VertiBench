import torch
import numpy as np
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def fetch_dataset(data_name, args, verbose=True):
    import datasets
    dataset = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\')'.format(data_name))
    elif data_name in ['MNIST', 'CIFAR10']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
    elif data_name in ['ModelNet40', 'ShapeNet55']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
    elif data_name in ['MIMICL', 'MIMICM']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\')'.format(data_name))
    elif data_name in ['MSD', 'CovType', 'Higgs', 'Gisette', 'Letter', 'Radar', 'Epsilon', 'Realsim', "MNIST_VB", "CIFAR10_VB", "Wide", "Vehicle"]:
        dataset['train'] = eval(f"datasets.{data_name}(split='train', typ='{args['splitter']}', val='{args['weight']}', dataseed='{args['dataseed']}', num_clients={args['num_clients']})")
        dataset['test'] = eval(f"datasets.{data_name}(split='test', typ='{args['splitter']}', val='{args['weight']}', dataseed='{args['dataseed']}', num_clients={args['num_clients']})")
    else:
        raise ValueError('Not a valid dataset name')
    if verbose:
        print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, shuffle=None):
    data_loader = {}
    for k in dataset:
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        data_loader[k] = DataLoader(dataset=dataset[k], shuffle=_shuffle, batch_size=cfg[tag]['batch_size'][k],
                                    pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                    collate_fn=input_collate,
                                    worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


def split_dataset(num_users, dataset):
    if cfg['data_name'] in ['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR', 'ModelNet40',
                            'ShapeNet55']:
        num_features = cfg['data_shape'][-1]
        feature_split = list(torch.randperm(num_features).split(num_features // num_users))
        feature_split = feature_split[:num_users - 1] + [torch.cat(feature_split[num_users - 1:])]
    elif cfg['data_name'] in ['MIMICL', 'MIMICM']:
        if cfg['num_users'] == 1:
            feature_split = [list(range(22))]
        elif cfg['num_users'] == 4:
            feature_split = [None for _ in range(4)]
            feature_split[0] = list(range(16))
            feature_split[1] = list(range(16, 19))
            feature_split[2] = list(range(19, 21))
            feature_split[3] = [21]
        else:
            raise ValueError('Not valid num users')
    elif cfg['data_name'] in ['MNIST', 'CIFAR10']:
        num_features = np.prod(cfg['data_shape']).item()
        idx = torch.arange(num_features).view(*cfg['data_shape'])
        power = np.log2(num_users)
        n_h, n_w = int(2 ** (power // 2)), int(2 ** (power - power // 2))
        feature_split = idx.view(cfg['data_shape'][0], n_h, cfg['data_shape'][1] // n_h, n_w,
                                 cfg['data_shape'][2] // n_w).permute(1, 3, 0, 2, 4).reshape(
            -1, cfg['data_shape'][0], cfg['data_shape'][1] // n_h, cfg['data_shape'][2] // n_w)
        feature_split = list(feature_split.reshape(feature_split.size(0), -1))
    elif cfg['data_name'] in ['MSD', 'CovType', 'Higgs', 'Gisette', 'Letter', 'Radar', 'Epsilon', 'Realsim', "MNIST_VB", "CIFAR10_VB", "Wide", "Vehicle"]:
        assert cfg['num_users'] == len(dataset['train'].partitions)
        p = dataset['train'].partitions
        
        feature_split = [
            torch.arange(sum(p[:i]) , sum(p[: i+1]), dtype=torch.int) for i in range(len(p))
        ]
    else:
        raise ValueError('Not valid data name')
    return feature_split

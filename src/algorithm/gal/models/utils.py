import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg


def init_param(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    return m


def normalize(input):
    if cfg['data_name'] in cfg['stats']:
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = cfg['stats'][cfg['data_name']]
        m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
               torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
        input = input.sub(m).div(s)
    return input


def denormalize(input):
    if cfg['data_name'] in cfg['stats']:
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = cfg['stats'][cfg['data_name']]
        m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
               torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
        input = input.mul(s).add(m)
    return input


def feature_split(input, feature_split):
    if cfg['data_name'] in ['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR', 'MIMICL',
                            'MIMICM', 
                            
                            'MSD', 'CovType', 'Higgs', 'Gisette', 'Letter', 'Radar', 'Epsilon', 'Realsim', 'MNIST_VB', 'CIFAR10_VB', 'Wide', 'Vehicle']:
        mask = torch.zeros(input.size(-1), device=input.device)
        mask[feature_split] = 1
        output = torch.masked_fill(input, mask == 0, 0)
    elif cfg['data_name'] in ['MNIST', 'CIFAR10']:
        num_features = np.prod(cfg['data_shape']).item()
        mask = torch.zeros(num_features, device=input.device)
        mask[feature_split] = 1
        mask = mask.view(cfg['data_shape'])
        output = torch.masked_fill(input, mask == 0, 0)
    elif cfg['data_name'] in ['ModelNet40', 'ShapeNet55']:
        output = torch.index_select(input, -1, feature_split)
        output = output.permute(4, 0, 1, 2, 3).reshape(-1, *output.size()[1:-1])
    else:
        raise ValueError('Not valid data name')
    return output


def loss_fn(output, target, reduction='mean', loss_mode=None):
    if target.dtype == torch.int64:
        if cfg['data_name'] in ['MIMICM']:
            if len(output.size()) == 3:
                output = output.permute(0, 2, 1)
            loss = F.cross_entropy(output, target, reduction=reduction, ignore_index=-65535)
        else:
            loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            if cfg['data_name'] == 'MIMICM':
                reduction = 'sum'
            mask = target != -65535
            output, target = output[mask], target[mask]
            if loss_mode is None:
                if cfg['data_name'] in ['Diabetes', 'BostonHousing', 'MIMICL']:
                    loss = F.l1_loss(output, target, reduction=reduction)
                else:
                    loss = F.mse_loss(output, target, reduction=reduction)
            else:
                if loss_mode == 'l1':
                    loss = F.l1_loss(output, target, reduction=reduction)
                elif loss_mode == 'l1.5':
                    if reduction == 'sum':
                        loss = (output - target).abs().pow(1.5).sum()
                    else:
                        loss = (output - target).abs().pow(1.5).mean()
                elif loss_mode == 'l2':
                    loss = F.mse_loss(output, target, reduction=reduction)
                elif loss_mode == 'l4':
                    if reduction == 'sum':
                        loss = (output - target).abs().pow(4).sum()
                    else:
                        loss = (output - target).abs().pow(4).mean()
                else:
                    raise ValueError('Not valid loss mode')
        else:
            if loss_mode is None:
                if cfg['data_name'] in ['Diabetes', 'BostonHousing', 'MIMICL']:
                    loss = F.l1_loss(output, target, reduction=reduction)
                else:
                    loss = F.mse_loss(output, target, reduction=reduction)
            else:
                if loss_mode == 'l1':
                    loss = F.l1_loss(output, target, reduction=reduction)
                elif loss_mode == 'l1.5':
                    if reduction == 'sum':
                        loss = (output - target).abs().pow(1.5).sum()
                    else:
                        loss = (output - target).abs().pow(1.5).mean()
                elif loss_mode == 'l2':
                    loss = F.mse_loss(output, target, reduction=reduction)
                elif loss_mode == 'l4':
                    if reduction == 'sum':
                        loss = (output - target).abs().pow(4).sum()
                    else:
                        loss = (output - target).abs().pow(4).mean()
                else:
                    raise ValueError('Not valid loss mode')
    return loss


def reset_parameters(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def unpad_sequence(padded_seq, length):
    unpadded_seq = []
    for i in range(len(length)):
        unpadded_seq.append(padded_seq[i, :length[i]])
    return unpadded_seq

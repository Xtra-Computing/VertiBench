import copy
import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split, reset_parameters, unpad_sequence
from torch.nn.utils.rnn import pad_sequence


class VFL(nn.Module):
    def __init__(self, num_users, block, hidden_size, target_size):
        super().__init__()
        self.num_users = num_users
        self.hidden_size = hidden_size
        blocks = []
        for i in range(num_users):
            block = copy.deepcopy(block)
            block.apply(reset_parameters)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.linear = nn.Linear(hidden_size * num_users, target_size)

    def forward(self, input):
        output = {}
        x = []
        for i in range(self.num_users):
            if input['feature_split'][i] is not None:
                if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                    x_i = {'data': input['data'], 'length': input['length'], 'feature_split': input['feature_split'][i]}
                    x_i = self.blocks[i].feature(x_i)
                else:
                    x_i = {'data': input['data'], 'feature_split': input['feature_split'][i]}
                    x_i = self.blocks[i].feature(x_i)
            else:
                if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                    x_i = input['data'].new_zeros((input['data'].size(0), input['data'].size(1), self.hidden_size))
                else:
                    x_i = input['data'].new_zeros((input['data'].size(0), self.hidden_size))
            x.append(x_i)
        x = torch.stack(x, dim=-1)
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            x = x.reshape(x.size(0), x.size(1), -1)
        else:
            x = x.reshape(x.size(0), -1)
        output['target'] = self.linear(x)
        if 'target' in input:
            if cfg['data_name'] in ['ModelNet40', 'ShapeNet55']:
                input['target'] = input['target'].repeat(12 // cfg['num_users'])
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def vfl(block, hidden_size):
    num_users = cfg['num_users']
    target_size = cfg['target_size']
    model = VFL(num_users, block, hidden_size, target_size)
    model.apply(init_param)
    return model

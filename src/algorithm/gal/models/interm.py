import copy
import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split, reset_parameters


class Interm(nn.Module):
    def __init__(self, num_users, block, hidden_size, target_size):
        super().__init__()
        blocks = []
        for i in range(num_users):
            block = copy.deepcopy(block)
            block.apply(reset_parameters)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, input):
        output = {}
        x = []
        for i in range(len(self.blocks)):
            if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                x_i = {'data': input['data'], 'length': input['length'], 'feature_split': input['feature_split'][i]}
            else:
                x_i = {'data': input['data'], 'feature_split': input['feature_split'][i]}
            x_i = self.blocks[i].feature(x_i)
            x.append(x_i)
        x = torch.stack(x, dim=0).mean(dim=0)
        output['target'] = self.linear(x)
        if cfg['data_name'] in ['ModelNet40', 'ShapeNet55']:
            input['target'] = input['target'].repeat(12 // cfg['num_users'])
        # if cfg['data_name'] == 'MIMICM':
        #     output['target'] = output['target'].permute(0, 2, 1)
        output['loss'] = loss_fn(output['target'], input['target'])
        return output


def interm(block, hidden_size):
    num_users = cfg['num_users']
    target_size = cfg['target_size']
    model = Interm(num_users, block, hidden_size, target_size)
    model.apply(init_param)
    return model

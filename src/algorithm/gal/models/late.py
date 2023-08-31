import copy
import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split, reset_parameters


class Late(nn.Module):
    def __init__(self, num_users, block):
        super().__init__()
        blocks = []
        for i in range(num_users):
            block = copy.deepcopy(block)
            block.apply(reset_parameters)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        output = {}
        x = []
        output['loss'] = 0
        if cfg['data_name'] in ['ModelNet40', 'ShapeNet55']:
            input['target'] = input['target'].repeat(12 // cfg['num_users'])
        for i in range(len(self.blocks)):
            if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                x_i = {'data': input['data'], 'length': input['length'], 'feature_split': input['feature_split'][i]}
            else:
                x_i = {'data': input['data'], 'feature_split': input['feature_split'][i]}
            x_i = self.blocks[i](x_i)
            output['loss'] += loss_fn(x_i['target'], input['target'])
            x.append(x_i['target'])
        output['target'] = torch.stack(x, dim=0).mean(dim=0)
        return output


def late(block):
    num_users = cfg['num_users']
    model = Late(num_users, block)
    model.apply(init_param)
    return model

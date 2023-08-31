import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split
from .late import late
from .dl import dl


class Linear(nn.Module):
    def __init__(self, data_shape, target_size):
        super().__init__()
        input_size = np.prod(data_shape).item()
        self.linear1 = nn.Linear(input_size, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, target_size)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def feature(self, input):
        x = input['data']
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        
        return x

    def forward(self, input):
        output = {}
        x = input['data']
        x = normalize(x)

        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        x = x.view(x.size(0), -1)
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)

        output['target'] = x
        if 'target' in input:
            if 'loss_mode' in input:
                output['loss'] = loss_fn(output['target'], input['target'], loss_mode=input['loss_mode'])
            else:
                output['loss'] = loss_fn(output['target'], input['target'])
        return output


def linear():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    if cfg['assist_mode'] == 'late':
        model = late(Linear(data_shape, target_size))
    elif cfg['assist_mode'] in ['none', 'bag', 'stack']:
        if 'dl' in cfg and cfg['dl'] == '1':
            model = dl(Linear(data_shape, target_size), target_size)
        else:
            model = Linear(data_shape, target_size)
    else:
        raise ValueError('Not valid assist mode')
    model.apply(init_param)
    return model

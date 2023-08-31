import torch
import torch.nn as nn
from config import cfg
from .utils import loss_fn


class Stack(nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.stack = nn.Parameter(torch.ones(num_users))

    def forward(self, input):
        output = {}
        x = input['output']
        output['target'] = (x * self.stack.softmax(-1)).sum(-1)
        if 'loss_mode' in input:
            output['loss'] = loss_fn(output['target'], input['target'], loss_mode=input['loss_mode'])
        else:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def stack():
    num_users = cfg['num_users']
    model = Stack(num_users)
    return model

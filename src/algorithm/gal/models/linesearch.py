import torch
import torch.nn as nn
from .utils import loss_fn


class LineSearch(nn.Module):
    def __init__(self):
        super().__init__()
        self.assist_rate = nn.Parameter(torch.ones(1))

    def forward(self, input):
        output = {}
        output['loss'] = loss_fn(input['history'] + self.assist_rate * input['output'], input['target'])
        return output


def linesearch():
    model = LineSearch()
    return model

import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split
from .interm import interm
from .late import late
from .vfl import vfl
from .dl import dl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, data_shape, ICD9_embeddings, hidden_size, num_layers, target_size):
        super().__init__()
        self.embedding = nn.Embedding(ICD9_embeddings + 1, 64)
        self.lstm = nn.LSTM(data_shape[0] - 1 + 64, hidden_size, num_layers, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, target_size)

    def feature(self, input):
        x = input['data']
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        icd9 = x[:, :, -1].long()
        icd9_embedding = self.embedding(icd9)
        x = torch.cat([x[:, :, :-1], icd9_embedding], dim=-1)
        x = pack_padded_sequence(x, input['length'].cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)
        return x

    def forward(self, input):
        output = {}
        x = input['data']
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        icd9 = x[:, :, -1].long()
        icd9_embedding = self.embedding(icd9)
        x = torch.cat([x[:, :, :-1], icd9_embedding], dim=-1)
        x = pack_padded_sequence(x, input['length'].cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)
        x = self.linear(x)
        output['target'] = x
        if 'target' in input:
            if 'loss_mode' in input:
                output['loss'] = loss_fn(output['target'], input['target'], loss_mode=input['loss_mode'])
            else:
                output['loss'] = loss_fn(output['target'], input['target'])
        return output


def lstm():
    data_shape = cfg['data_shape']
    ICD9_embeddings = cfg['lstm']['ICD9_embeddings']
    hidden_size = cfg['lstm']['hidden_size']
    num_layers = cfg['lstm']['num_layers']
    target_size = cfg['target_size']
    if cfg['assist_mode'] == 'interm':
        model = interm(LSTM(data_shape, ICD9_embeddings, hidden_size, num_layers, target_size), hidden_size)
    elif cfg['assist_mode'] == 'late':
        model = late(LSTM(data_shape, ICD9_embeddings, hidden_size, num_layers, target_size))
    elif cfg['assist_mode'] == 'vfl':
        model = vfl(LSTM(data_shape, ICD9_embeddings, hidden_size, num_layers, target_size), hidden_size)
    elif cfg['assist_mode'] in ['none', 'bag', 'stack']:
        if 'dl' in cfg and cfg['dl'] == '1':
            model = dl(LSTM(data_shape, ICD9_embeddings, hidden_size, num_layers, target_size), hidden_size)
        else:
            model = LSTM(data_shape, ICD9_embeddings, hidden_size, num_layers, target_size)
    else:
        raise ValueError('Not valid assist mode')
    model.apply(init_param)
    return model

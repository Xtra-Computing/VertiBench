import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import normalize, loss_fn, feature_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


class SK(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if model_name == 'gb':
            self.model = MultiOutputRegressor(GradientBoostingRegressor())
        elif model_name == 'svm':
            self.model = MultiOutputRegressor(SVR())
        else:
            raise ValueError('Not valid model name')

    def fit(self, input):
        output = {}
        x = normalize(input['data'])
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        self.model.fit(x.numpy(), input['target'].numpy())
        output['target'] = torch.tensor(self.model.predict(x.numpy()), dtype=torch.float)
        if 'target' in input:
            if 'loss_mode' in input:
                output['loss'] = loss_fn(output['target'], input['target'], loss_mode=input['loss_mode'])
            else:
                output['loss'] = loss_fn(output['target'], input['target'])
        return output

    def predict(self, input):
        output = {}
        x = normalize(input['data'])
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        output['target'] = torch.tensor(self.model.predict(x.numpy()), dtype=torch.float)
        if 'target' in input:
            if 'loss_mode' in input:
                output['loss'] = loss_fn(output['target'], input['target'], loss_mode=input['loss_mode'])
            else:
                output['loss'] = loss_fn(output['target'], input['target'])
        return output

    def state_dict(self):
        return self.model.get_params()

    def load_state_dict(self, state_dict):
        self.model.set_params(**state_dict)
        return


def gb():
    if cfg['assist_mode'] in ['none', 'bag', 'stack']:
        model = SK('gb')
    else:
        raise ValueError('Not valid assist mode')
    return model


def svm():
    if cfg['assist_mode'] in ['none', 'bag', 'stack']:
        model = SK('svm')
    else:
        raise ValueError('Not valid assist mode')
    return model

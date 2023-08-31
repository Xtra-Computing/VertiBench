import torch
import torch.nn.functional as F
from config import cfg
from utils import recur
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
# output: (10000, 10)
# target: (10000)

# output: (116203, 7)
# target: (116203)
def Accuracy(output, target, topk=1):
    with torch.no_grad():
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.unsqueeze(1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def AUCROC(output, target):
    auc_roc = roc_auc_score(target, output)
    return auc_roc


def MAD(output, target):
    with torch.no_grad():
        if cfg['data_name'] in ['MIMICL']:
            output = output[target != -65535]
            target = target[target != -65535]
        mad = F.l1_loss(output, target).item()
    return mad

def RMSE(output, target):
    rmse = mean_squared_error(output, target, squared=False)
    return rmse

class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Accuracy': (lambda input, output: recur(Accuracy, output['target'], input['target'])),
                       'MAD': (lambda input, output: recur(MAD, output['target'], input['target'])),
                       'AUCROC': (lambda input, output: recur(AUCROC, output['target'], input['target'])),}
        
        self.metric['RMSE'] = (lambda input, output: recur(RMSE, output['target'], input['target']))

    def make_metric_name(self, metric_name):
        for split in metric_name:
            if split == 'test':
                if cfg['data_name'] in ['Blob', 'Iris', 'Wine', 'BreastCancer', 'QSAR', 'MNIST', 'CIFAR10',
                                        'ModelNet40', 'ShapeNet55', 'CovType', 'Higgs', 'Gisette','Realsim','Epsilon', 'Letter', 'Radar', "MNIST_VB", "CIFAR10_VB", "Wide", "Vehicle"]:
                    metric_name[split] += ['Accuracy']
                elif cfg['data_name'] in ['MIMICM']:
                    metric_name[split] += ['AUCROC']
                elif cfg['data_name'] in ['Diabetes', 'BostonHousing', 'MIMICL']:
                    metric_name[split] += ['MAD']
                elif cfg['data_name'] in ['MSD']:
                    metric_name[split] += ['RMSE']
                else:
                    raise ValueError('Not valid data name')
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['Blob', 'Iris', 'Wine', 'BreastCancer', 'QSAR', 'MNIST', 'CIFAR10', 'ModelNet40',
                                'ShapeNet55', 'CovType', 'Higgs', 'Gisette','Realsim','Epsilon', 'Letter', 'Radar', "MNIST_VB", "CIFAR10_VB", "Wide", "Vehicle"]:
            pivot = -float('inf')
            pivot_name = 'Accuracy'
            pivot_direction = 'up'
        elif cfg['data_name'] in ['MIMICM']:
            pivot = -float('inf')
            pivot_name = 'AUCROC'
            pivot_direction = 'up'
        elif cfg['data_name'] in ['Diabetes', 'BostonHousing', 'MIMICL']:
            pivot = float('inf')
            pivot_name = 'MAD'
            pivot_direction = 'down'
        elif cfg['data_name'] in ['MSD']:
            pivot = float('inf')
            pivot_name = 'RMSE'
            pivot_direction = 'down'
        else:
            raise ValueError('Not valid data name')
        return pivot, pivot_name, pivot_direction

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return

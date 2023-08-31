import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, load, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, \
    collate
from logger import Logger

# if __name__ == "__main__":
#     data_name = 'Wine'
#     subset = 'label'
#     dataset = fetch_dataset(data_name, subset)
#     process_dataset(dataset)
#     data_loader = make_data_loader(dataset)
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(input['feature'].size())
#         print(input[subset].size())
#         break
#     exit()

# import torch.nn.functional as F
#
# if __name__ == "__main__":
#     N = 20
#     C = 5
#     score = torch.rand(N, C)
#     label = torch.randint(0, 2, (N,))
#     for i in range(10):
#         score.requires_grad = True
#         loss = F.cross_entropy(score, label)
#         print(loss)
#         loss.backward()
#         score = (score - 1000* score.grad).detach()
#
#
# if __name__ == "__main__":
#     # p = torch.tensor([0.1, 0.3, 0.2, 0.4]).view(1, -1)
#     # p = torch.tensor([1e-10, 1e-10, 1, 1e-10]).view(1, -1)
#     label = torch.tensor([2])
#     one_hot = torch.tensor([0, 0, 1, 0])
#     odds = one_hot.float()
#     odds[odds == 0] = 1e-4
#     log_odds = torch.log(odds)
#     sm = torch.softmax(log_odds, dim=-1)
#     loss = F.cross_entropy(log_odds.view(1, -1), label)
#     print(log_odds)
#     print(sm)
#     print(loss)

# import math

# if __name__ == "__main__":
#     h, w = 4, 4
#     n = 2
#     power = np.log2(n)
#     n_h = int(2 ** (power // 2))
#     n_w = int(2 ** (power - power // 2))
#     print(n_h, n_w)
#     a = torch.arange(h * w).view(h, w)
#     print(a)
#     print(a.view(n_h, h // n_h, n_w, w // n_w).size())
#     b = a.view(n_h, h // n_h, n_w, w // n_w).transpose(1, 2).reshape(-1, h // n_h, w // n_w)
#     print(b)
#     print(list(b))
#     print(len(b))

# if __name__ == "__main__":
#     torch.manual_seed(1)
#     torch.cuda.manual_seed(1)
#     process_control()
#     cfg['data_name'] = 'MIMIC'
#     dataset = fetch_dataset(cfg['data_name'])
#     process_dataset(dataset)
#     data_loader = make_data_loader(dataset, cfg['model_name'])
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(input['data'].shape)
#         print(input['target'].shape)
#         break
#     exit()
# import numpy as np
# from sklearn.linear_model import LinearRegression

# if __name__ == "__main__":
#     torch.manual_seed(1)
#     torch.cuda.manual_seed(1)
#     process_control()
#     cfg['data_name'] = 'MIMIC'
#     dataset = fetch_dataset(cfg['data_name'])
#     process_dataset(dataset)
#     X_train, y_train = dataset['train'].data, dataset['train'].target
#     X_test, y_test = dataset['test'].data, dataset['test'].target
#     reg = LinearRegression().fit(X_train, y_train)
#     y_pred = reg.predict(X_test)
#     rmse = np.sqrt(np.mean((y_pred-y_test)**2))
#     print(rmse)

# if __name__ == "__main__":
#     cfg['seed'] = 0
#     torch.manual_seed(cfg['seed'])
#     torch.cuda.manual_seed(cfg['seed'])
#     process_control()
#     cfg['data_name'] = 'ModelNet40'
#     dataset = fetch_dataset(cfg['data_name'])
#     process_dataset(dataset)
#     data_loader = make_data_loader(dataset, cfg['model_name'])
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(input['data'].shape)
#         print(input['target'].shape)
#         break
#     exit()


# if __name__ == "__main__":
#     cfg['seed'] = 0
#     torch.manual_seed(cfg['seed'])
#     torch.cuda.manual_seed(cfg['seed'])
#     cfg['data_name'] = 'ModelNet40'
#     cfg['model_name'] = 'conv'
#     process_control()
#     dataset = fetch_dataset(cfg['data_name'])
#     print(len(dataset['train']), len(dataset['test']))
#     process_dataset(dataset)
#     data_loader = make_data_loader(dataset, cfg['model_name'])
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(input['data'].shape)
#         print(input['target'].shape)
#         break
#
#     cfg['seed'] = 0
#     torch.manual_seed(cfg['seed'])
#     torch.cuda.manual_seed(cfg['seed'])
#     cfg['data_name'] = 'MIMIC'
#     cfg['model_name'] = 'lstm'
#     process_control()
#     dataset = fetch_dataset(cfg['data_name'])
#     print(len(dataset['train']), len(dataset['test']))
#     process_dataset(dataset)
#     data_loader = make_data_loader(dataset, cfg['model_name'])
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(input['data'].shape)
#         print(input['target'].shape)
#         break


# if __name__ == "__main__":
#     x = torch.randn(10, 5)
#     y = torch.randn(10, 5)
#     mae = torch.nn.L1Loss()(x, y)
#     mse = torch.nn.MSELoss()(x, y)
#     p1 = torch.norm((x - y).abs(), 1, dim=-1).pow(1).sum().div(x.numel())
#     p1_5 = torch.norm((x - y).abs(), 1.5, dim=-1).pow(1.5).sum().div(x.numel())
#     p2 = torch.norm((x - y).abs(), 2, dim=-1).pow(2).sum().div(x.numel())
#     print(mae, p1, mse, p2, p1_5, (x - y).abs().pow(1.5).mean())


# if __name__ == "__main__":
#     cfg['seed'] = 0
#     torch.manual_seed(cfg['seed'])
#     torch.cuda.manual_seed(cfg['seed'])
#     cfg['data_name'] = 'ModelNet40'
#     cfg['model_name'] = 'conv'
#     process_control()
#     dataset = fetch_dataset(cfg['data_name'])
#     process_dataset(dataset)
#     data_loader = make_data_loader(dataset, cfg['model_name'])
#     print(len(dataset['train']), len(dataset['test']))
#     print(len(data_loader['train']), len(data_loader['test']))
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(i, input['data'].shape, input['target'].shape)
#         break
#     for i, input in enumerate(data_loader['test']):
#         input = collate(input)
#         print(i, input['data'].shape, input['target'].shape)
#         break


# if __name__ == "__main__":
#     cfg['seed'] = 0
#     torch.manual_seed(cfg['seed'])
#     torch.cuda.manual_seed(cfg['seed'])
#     cfg['data_name'] = 'MIMICM'
#     cfg['model_name'] = 'lstm'
#     process_control()
#     dataset = fetch_dataset(cfg['data_name'])
#     process_dataset(dataset)
#     data_loader = make_data_loader(dataset, cfg['model_name'])
#     print(len(dataset['train']), len(dataset['test']))
#     print(len(data_loader['train']), len(data_loader['test']))
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(i, input['data'].shape, input['target'].shape)
#         break
#     for i, input in enumerate(data_loader['test']):
#         input = collate(input)
#         print(i, input['data'].shape, input['target'].shape)
#         break
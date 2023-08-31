import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset
from metrics import Metric
from utils import process_control, save_img
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join(
    [cfg['control'][k] for k in cfg['control'] if cfg['control'][k]]) if 'control' in cfg else ''


def main():
    process_control()
    dataset = fetch_dataset(cfg['data_name'])
    data_loader = make_data_loader(dataset, cfg['model_name'])
    img = next(iter(data_loader['train']))['data'][0]
    M = [1, 2, 4, 8]
    for num_user in M:
        feature_split = split_dataset(num_user, dataset)
        for i in range(num_user):
            num_features = np.prod(cfg['data_shape']).item()
            mask = torch.zeros(num_features, device=img.device)
            mask[feature_split[i]] = 1
            mask = mask.view(cfg['data_shape'])
            img_i = img[mask == 1]
            power = np.log2(num_user)
            n_h, n_w = int(2 ** (power // 2)), int(2 ** (power - power // 2))
            img_i = img_i.view(cfg['data_shape'][0], cfg['data_shape'][1] // n_h, cfg['data_shape'][1] // n_w)
            save_img(img_i, './output/vis/{}_{}_{}.png'.format(cfg['data_name'], num_user, i), nrow=1, padding=0)
    return


if __name__ == "__main__":
    main()

import argparse
import copy
import models
import os
import sys
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset
from metrics import Metric
from assist import Assist
from utils import save, load, process_control, process_dataset, resume
from logger import make_logger
import datetime, pytz

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
parser.add_argument('--splitter', default="corr", type=str) # corr, imp
parser.add_argument('--weight', default="0.3", type=str)
parser.add_argument('--dataseed', default="0", type=str)

args = vars(parser.parse_args())
print("🔵 args: ", args)
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join(
    [cfg['control'][k] for k in cfg['control'] if cfg['control'][k]]) if 'control' in cfg else ''

cfg['splitter'], cfg['weight'], cfg['dataseed'] = args['splitter'], args['weight'], args['dataseed']
args['num_clients'] = cfg['control']['num_users'] # for later dataset.eval use.
cfg['resume_mode'] = 1


LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])
MASTER_ADDR = os.environ['MASTER_ADDR']
MASTER_PORT = os.environ['MASTER_PORT']
# we assume the primary party id == 0
def init_process(rank, size, backend='gloo'):
    dist.init_process_group(backend, rank=rank, world_size=size)
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name'], cfg['splitter'], cfg['weight'], cfg['dataseed']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), 'Experiment: {}'.format(cfg['model_tag']))
        group = dist.new_group(list(range(size)))
        runExperiment(rank, group)
    return

def get_device_from_gpu_id(gpu_id):
    if gpu_id is None:
        return torch.device('cpu')
    return torch.device(f'cuda:{gpu_id}')

def runExperiment(rank, group):
    device = get_device_from_gpu_id(rank)
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'], args)
    process_dataset(dataset)
    feature_split = split_dataset(cfg['num_users'], dataset)
    assist = Assist(feature_split)
    organization = assist.make_organization()
    metric = Metric({'train': ['Loss'], 'test': ['Loss']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        logger = result['logger']
        if last_epoch > 1:
            assist = result['assist']
            organization = result['organization']
        else:
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    if organization is None:
        organization = assist.make_organization()
    if last_epoch == 1:
        initialize(dataset, assist, organization[0], metric, logger, 0)
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        logger.safe(True)

        # Send Party0's label to other parties
        data_loader = assist.broadcast_distributed(dataset, epoch, rank, group)
        
        train(data_loader, organization, metric, logger, epoch, rank)
        
        # receive other parties' intermediate results
        organization_outputs = gather(data_loader, organization, epoch, rank, group)
        
        
        assist.update(organization_outputs, epoch)
        test(assist, metric, logger, epoch)
        logger.safe(False)
        # save_result = {'cfg': cfg, 'epoch': epoch + 1, 'assist': assist, 'organization': organization, 'logger': logger}
        # save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        #if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
        #    metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
        #    shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
        #                './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def initialize(dataset, assist, organization, metric, logger, epoch):
    logger.safe(True)
    initialization = organization.initialize(dataset, metric, logger)
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch: {}'.format(epoch), 'ID: 1']}
    logger.append(info, 'train', mean=False)
    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), logger.write('train', metric.metric_name['train']))
    info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
    logger.append(info, 'test', mean=False)
    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), logger.write('test', metric.metric_name['test']))
    for split in dataset:
        assist.organization_output[0][split] = initialization[split]
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            assist.organization_target[0][split] = torch.tensor(np.concatenate(dataset[split].target, axis=0))
        else:
            assist.organization_target[0][split] = torch.tensor(dataset[split].target)
    logger.safe(False)
    logger.reset()
    return


def train(data_loader, organization, metric, logger, epoch, rank):
    start_time = time.time()
    num_organizations = len(organization)
    for i in range(num_organizations):
        if i != rank:
            continue
        organization[i].train(epoch, data_loader[i]['train'], metric, logger)
        if i % int((num_organizations * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_organizations - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * local_time * num_organizations))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / num_organizations),
                             'ID: {}/{}'.format(i + 1, num_organizations),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), logger.write('train', metric.metric_name['train']))
    return


def gather(data_loader, organization, epoch, rank : int, group : dist.ProcessGroup):
    with torch.no_grad():
        num_organizations = len(organization)

        organization_outputs = [{split: None for split in ['train', 'test']} for _ in range(num_organizations)]
        
        dist_gather = {}
        for split in organization_outputs[rank]:
            organization_outputs[rank][split] = organization[rank].predict(epoch, data_loader[rank][split])['target']

            dist_gather[split] = [torch.zeros_like(organization_outputs[rank][split]) for _ in range(num_organizations)]
            if rank == 0: # we assume the primary party id == 0
                dist.gather(organization_outputs[rank][split], gather_list = dist_gather[split], group=group)
            else:
                dist.gather(organization_outputs[rank][split], dst=0, group=group) # we assume the primary party id == 0
        
        # gather all data to party 0, process to GAL's original format
        for i in range(num_organizations):
            for split in organization_outputs[i]:
                organization_outputs[i][split] = dist_gather[split][i]

    return organization_outputs


def test(assist, metric, logger, epoch):
    with torch.no_grad():
        input_size = assist.organization_target[0]['test'].size(0)
        input = {'target': assist.organization_target[0]['test']}
        output = {'target': assist.organization_output[epoch]['test']}
        output['loss'] = models.loss_fn(output['target'], input['target'])
        if cfg['data_name'] in ['MIMICM']:
            mask = input['target'] != -65535
            output['target'] = output['target'].softmax(dim=-1)[:, 1]
            output['target'], input['target'] = output['target'][mask], input['target'][mask]
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test', n=input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    init_process(WORLD_RANK, WORLD_SIZE, backend='gloo')

import datetime
import numpy as np
import sys
import time
import torch
import models
from config import cfg
from utils import to_device, make_optimizer, make_scheduler, collate
import datetime
import pytz

class Organization:
    def __init__(self, organization_id, feature_split, model_name):
        self.organization_id = organization_id
        self.feature_split = feature_split
        self.model_name = model_name
        self.model_parameters = [None for _ in range(cfg['global']['num_epochs'] + 1)]

    def initialize(self, dataset, metric, logger):
        input, output, initialization = {}, {}, {}
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            train_target = torch.tensor(np.concatenate(dataset['train'].target, axis=0))
            test_target = torch.tensor(np.concatenate(dataset['test'].target, axis=0))
        else:
            train_target = torch.tensor(dataset['train'].target)
            test_target = torch.tensor(dataset['test'].target)
        if train_target.dtype == torch.int64:
            if cfg['data_name'] in ['MIMICM']:
                _, _, counts = torch.unique(train_target[train_target != -65535], sorted=True, return_inverse=True,
                                            return_counts=True)
            else:
                _, _, counts = torch.unique(train_target, sorted=True, return_inverse=True, return_counts=True)
            x = (counts / counts.sum()).log()
            initialization['train'] = x.view(1, -1).repeat(train_target.size(0), 1)
            initialization['test'] = x.view(1, -1).repeat(test_target.size(0), 1)
        else:
            if cfg['data_name'] in ['MIMICL']:
                x = train_target[train_target != -65535].mean()
            else:
                x = train_target.mean()
            initialization['train'] = x.expand_as(train_target).detach().clone()
            initialization['test'] = x.expand_as(test_target).detach().clone()
        if 'train' in metric.metric_name:
            input['target'], output['target'] = train_target, initialization['train']
            output['loss'] = models.loss_fn(output['target'], input['target'])
            evaluation = metric.evaluate(metric.metric_name['train'], input, output)
            logger.append(evaluation, 'train', n=train_target.size(0))
        input['target'], output['target'] = test_target, initialization['test']
        output['loss'] = models.loss_fn(output['target'], input['target'])
        if cfg['data_name'] in ['MIMICM']:
            mask = input['target'] != -65535
            output['target'] = output['target'].softmax(dim=-1)[:, 1]
            output['target'], input['target'] = output['target'][mask], input['target'][mask]
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test', n=test_target.size(0))
        return initialization

    def train(self, iter, data_loader, metric, logger):
        if self.model_name[iter] in ['gb', 'svm']:
            model = eval('models.{}().to(cfg["device"])'.format(self.model_name[iter]))
            data, target = data_loader.dataset.data, data_loader.dataset.target
            input = {'data': torch.tensor(data), 'target': torch.tensor(target), 'feature_split': self.feature_split}
            input_size = len(input['data'])
            output = model.fit(input)
            evaluation = metric.evaluate(metric.metric_name['train'], input, output)
            logger.append(evaluation, 'train', n=input_size)
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'ID: {}'.format(self.organization_id)]}
            logger.append(info, 'train', mean=False)
            print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), logger.write('train', metric.metric_name['train']), end='\r', flush=True)
            self.model_parameters[iter] = model
        else:
            model = eval('models.{}().to(cfg["device"])'.format(self.model_name[iter]))
            if 'dl' in cfg and ['dl'] == '1' and iter > 1:
                model.load_state_dict(self.model_parameters[iter - 1])
            model.train(True)
            optimizer = make_optimizer(model, self.model_name[iter])
            scheduler = make_scheduler(optimizer, self.model_name[iter])
            for local_epoch in range(1, cfg[self.model_name[iter]]['num_epochs'] + 1):
                start_time = time.time()
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['feature_split'] = self.feature_split
                    if cfg['noise'] == 'data' and self.organization_id in cfg['noised_organization_id']:
                        input['data'] = torch.randn(input['data'].size())
                        if 'MIMIC' in cfg['data_name']:
                            input['data'][:, :, -1] = 0
                    input = to_device(input, cfg['device'])
                    input['loss_mode'] = cfg['rl'][self.organization_id]
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                scheduler.step()
                local_time = (time.time() - start_time)
                local_finished_time = datetime.timedelta(
                    seconds=round((cfg[self.model_name[iter]]['num_epochs'] - local_epoch) * local_time))
                info = {'info': ['Model: {}'.format(cfg['model_tag']),
                                 'Train Local Epoch: {}({:.0f}%)'.format(local_epoch, 100. * local_epoch /
                                                                         cfg[self.model_name[iter]]['num_epochs']),
                                 'ID: {}'.format(self.organization_id),
                                 'Local Finished Time: {}'.format(local_finished_time),
                                 f'Learning Rate: {scheduler.get_lr()}']}
                logger.append(info, 'train', mean=False)
                print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), logger.write('train', metric.metric_name['train']), end='\r', flush=True)
            sys.stdout.write('\x1b[2K')
            self.model_parameters[iter] = model.to('cpu').state_dict()
        return

    def predict(self, iter, data_loader):
        if self.model_name[iter] in ['gb', 'svm']:
            model = self.model_parameters[iter]
            data, target = data_loader.dataset.data, data_loader.dataset.target
            input = {'data': torch.tensor(data), 'target': torch.tensor(target), 'feature_split': self.feature_split}
            output = model.predict(input)
            organization_output = {'id': [], 'target': []}
            organization_output['id'] = torch.tensor(data_loader.dataset.id)
            organization_output['target'] = output['target']
            organization_output['id'], indices = torch.sort(organization_output['id'])
            organization_output['target'] = organization_output['target'][indices]
        else:
            with torch.no_grad():
                model = eval('models.{}().to(cfg["device"])'.format(self.model_name[iter]))
                if 'dl' in cfg and cfg['dl'] == '1' and iter > 1:
                    for i in range(len(self.model_parameters)):
                        if self.model_parameters[i] is not None:
                            last_iter = i
                    model.load_state_dict(self.model_parameters[last_iter])
                else:
                    model.load_state_dict(self.model_parameters[iter])
                model.train(False)
                organization_output = {'id': [], 'target': []}
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input['feature_split'] = self.feature_split
                    if cfg['noise'] == 'data' and self.organization_id in cfg['noised_organization_id']:
                        input['data'] = torch.randn(input['data'].size())
                        if 'MIMIC' in cfg['data_name']:
                            input['data'][:, :, -1] = 0
                    input = to_device(input, cfg['device'])
                    output = model(input)
                    organization_output['id'].append(input['id'].cpu())
                    if 'dl' in cfg and cfg['dl'] == '1':
                        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                            output_target = output['target'][:, :, iter - 1].cpu()
                        else:
                            output_target = output['target'][:, iter - 1].cpu()
                    else:
                        output_target = output['target'].cpu()
                    if cfg['noise'] not in ['none', 'data'] and cfg['noise'] > 0 and \
                            self.organization_id in cfg['noised_organization_id']:
                        noise = torch.normal(0, cfg['noise'], size=output_target.size())
                        output_target = output_target + noise
                    if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                        output_target = models.unpad_sequence(output_target, input['length'])
                        organization_output['target'].extend(output_target)
                    else:
                        organization_output['target'].append(output_target)
                organization_output['id'] = torch.cat(organization_output['id'], dim=0)
                organization_output['id'], indices = torch.sort(organization_output['id'])
                if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                    organization_output['target'] = [organization_output['target'][idx] for idx in indices]
                    organization_output['target'] = torch.cat(organization_output['target'], dim=0)
                else:
                    organization_output['target'] = torch.cat(organization_output['target'], dim=0)
                    organization_output['target'] = organization_output['target'][indices]
        return organization_output

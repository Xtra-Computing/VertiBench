import copy
import numpy as np
import torch
import models
from config import cfg
import torch.distributed as dist
from data import make_data_loader
from organization import Organization
from privacy import make_privacy
from utils import make_optimizer, to_device


class Assist:
    def __init__(self, feature_split):
        self.feature_split = feature_split
        self.model_name = self.make_model_name()
        self.assist_parameters = [None for _ in range(cfg['global']['num_epochs'] + 1)]
        self.assist_rates = [None for _ in range(cfg['global']['num_epochs'] + 1)]
        self.reset()

    def reset(self):
        self.organization_output = [{split: None for split in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        self.organization_target = [{split: None for split in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        return

    def make_model_name(self):
        model_name_list = cfg['model_name'].split('-')
        num_split = cfg['num_users'] // len(model_name_list)
        rm_split = cfg['num_users'] - num_split * len(model_name_list)
        model_name = []
        for i in range(len(model_name_list)):
            model_name.extend([model_name_list[i] for _ in range(num_split)])
            if i == len(model_name_list) - 1:
                model_name.extend([model_name_list[i] for _ in range(rm_split)])
        for i in range(len(model_name)):
            model_name[i] = [model_name[i] for _ in range(cfg['global']['num_epochs'] + 1)]
        return model_name

    def make_organization(self):
        feature_split = self.feature_split
        model_name = self.model_name
        organization = [None for _ in range(len(feature_split))]
        for i in range(len(feature_split)):
            model_name_i = model_name[i]
            feature_split_i = feature_split[i]
            organization[i] = Organization(i, feature_split_i, model_name_i)
        return organization
    
    def broadcast_distributed(self, dataset, iter, rank, group):
        for split in dataset: # split: train, test
            self.organization_output[iter - 1][split].requires_grad = True
            loss = models.loss_fn(self.organization_output[iter - 1][split],
                                  self.organization_target[0][split], reduction='sum')
            loss.backward()
            self.organization_target[iter][split] = - copy.deepcopy(self.organization_output[iter - 1][split].grad)
            if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                if 'dl' in cfg and cfg['dl'] == '1':
                    target = self.organization_target[iter][split].unsqueeze(1).numpy()
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    target = np.split(target, np.cumsum(dataset[split].length), axis=0)
                    if iter == 1:
                        dataset[split].target = target
                    else:
                        dataset[split].target = [np.concatenate([dataset[split].target[i], target[i]], axis=1) for i in
                                                 range(len(dataset[split].target))]
                else:
                    target = self.organization_target[iter][split].numpy()
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    dataset[split].target = np.split(target, np.cumsum(dataset[split].length), axis=0)
            else:
                if 'dl' in cfg and cfg['dl'] == '1':
                    target = self.organization_target[iter][split].unsqueeze(1).numpy()
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    if iter == 1:
                        dataset[split].target = target
                    else:
                        dataset[split].target = np.concatenate([dataset[split].target, target], axis=1)
                else:
                    target = self.organization_target[iter][split].numpy()
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    
                    print("Broadcast", rank, target.shape)
                    if rank == 0:
                        tensor = torch.tensor(target)
                    else:
                        tensor = torch.zeros_like(torch.tensor(target))
                    # block until all process received the target from rank 0
                    dist.broadcast(tensor=tensor, src=0)
                    print("Broadcast Received", rank, tensor.shape)
                    dataset[split].target = tensor.numpy()
            self.organization_output[iter - 1][split].detach_()
        data_loader = [None for _ in range(len(self.feature_split))]
        for i in range(len(self.feature_split)):
            if i != rank:
                continue
            data_loader[i] = make_data_loader(dataset, self.model_name[i][iter])
        return data_loader
    
    def broadcast(self, dataset, iter):
        for split in dataset: # split: train, test
            self.organization_output[iter - 1][split].requires_grad = True
            loss = models.loss_fn(self.organization_output[iter - 1][split],
                                  self.organization_target[0][split], reduction='sum')
            loss.backward()
            self.organization_target[iter][split] = - copy.deepcopy(self.organization_output[iter - 1][split].grad)
            if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                if 'dl' in cfg and cfg['dl'] == '1':
                    target = self.organization_target[iter][split].unsqueeze(1).numpy()
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    target = np.split(target, np.cumsum(dataset[split].length), axis=0)
                    if iter == 1:
                        dataset[split].target = target
                    else:
                        dataset[split].target = [np.concatenate([dataset[split].target[i], target[i]], axis=1) for i in
                                                 range(len(dataset[split].target))]
                else:
                    target = self.organization_target[iter][split].numpy()
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    dataset[split].target = np.split(target, np.cumsum(dataset[split].length), axis=0)
            else:
                if 'dl' in cfg and cfg['dl'] == '1':
                    target = self.organization_target[iter][split].unsqueeze(1).numpy()
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    if iter == 1:
                        dataset[split].target = target
                    else:
                        dataset[split].target = np.concatenate([dataset[split].target, target], axis=1)
                else:
                    target = self.organization_target[iter][split].numpy()
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    dataset[split].target = target
            self.organization_output[iter - 1][split].detach_()
        data_loader = [None for _ in range(len(self.feature_split))]
        for i in range(len(self.feature_split)):
            data_loader[i] = make_data_loader(dataset, self.model_name[i][iter])
        return data_loader

    def update(self, organization_outputs, iter):
        if cfg['assist_mode'] == 'none':
            for split in organization_outputs[0]:
                self.organization_output[iter][split] = organization_outputs[0][split]
        elif cfg['assist_mode'] == 'bag':
            _organization_outputs = {split: [] for split in organization_outputs[0]}
            for split in organization_outputs[0]:
                for i in range(len(organization_outputs)):
                    _organization_outputs[split].append(organization_outputs[i][split])
                _organization_outputs[split] = torch.stack(_organization_outputs[split], dim=-1)
            for split in organization_outputs[0]:
                self.organization_output[iter][split] = _organization_outputs[split].mean(dim=-1)
        elif cfg['assist_mode'] == 'stack':
            _organization_outputs = {split: [] for split in organization_outputs[0]}
            for split in organization_outputs[0]:
                for i in range(len(organization_outputs)):
                    _organization_outputs[split].append(organization_outputs[i][split])
                _organization_outputs[split] = torch.stack(_organization_outputs[split], dim=-1)
            if 'train' in organization_outputs[0]:
                input = {'output': _organization_outputs['train'],
                         'target': self.organization_target[iter]['train']}
                input = to_device(input, cfg['device'])
                input['loss_mode'] = cfg['rl'][0]
                model = eval('models.{}().to(cfg["device"])'.format(cfg['assist_mode']))
                model.train(True)
                optimizer = make_optimizer(model, 'assist')
                for assist_epoch in range(1, cfg['assist']['num_epochs'] + 1):
                    output = model(input)
                    optimizer.zero_grad()
                    output['loss'].backward()
                    optimizer.step()
                self.assist_parameters[iter] = model.to('cpu').state_dict()
            with torch.no_grad():
                model = eval('models.{}().to(cfg["device"])'.format(cfg['assist_mode']))
                model.load_state_dict(self.assist_parameters[iter])
                model.train(False)
                for split in organization_outputs[0]:
                    input = {'output': _organization_outputs[split],
                             'target': self.organization_target[iter][split]}
                    input = to_device(input, cfg['device'])
                    self.organization_output[iter][split] = model(input)['target'].cpu()
        else:
            raise ValueError('Not valid assist')
        if 'train' in organization_outputs[0]:
            if cfg['assist_rate_mode'] == 'search':
                input = {'history': self.organization_output[iter - 1]['train'],
                         'output': self.organization_output[iter]['train'],
                         'target': self.organization_target[0]['train']}
                input = to_device(input, cfg['device'])
                model = models.linesearch().to(cfg['device'])
                model.train(True)
                optimizer = make_optimizer(model, 'linesearch')
                for linearsearch_epoch in range(1, cfg['linesearch']['num_epochs'] + 1):
                    def closure():
                        output = model(input)
                        optimizer.zero_grad()
                        output['loss'].backward()
                        return output['loss']

                    optimizer.step(closure)
                self.assist_rates[iter] = min(abs(model.assist_rate.item()), 300)
            else:
                self.assist_rates[iter] = cfg['linesearch']['lr']
        with torch.no_grad():
            for split in organization_outputs[0]:
                self.organization_output[iter][split] = self.organization_output[iter - 1][split] + self.assist_rates[
                    iter] * self.organization_output[iter][split]
        return

    def update_al(self, organization_outputs, iter):
        if cfg['assist_mode'] == 'none':
            for split in organization_outputs[0]:
                self.organization_output[iter][split] = organization_outputs[0][split]
        else:
            raise ValueError('Not valid assist mode')
        if 'train' in organization_outputs[0]:
            if cfg['assist_rate_mode'] == 'search':
                input = {'history': self.organization_output[iter - 1]['train'],
                         'output': self.organization_output[iter]['train'],
                         'target': self.organization_target[0]['train']}
                input = to_device(input, cfg['device'])
                model = models.linesearch().to(cfg['device'])
                model.train(True)
                optimizer = make_optimizer(model, 'linesearch')
                for linearsearch_epoch in range(1, cfg['linesearch']['num_epochs'] + 1):
                    def closure():
                        output = model(input)
                        optimizer.zero_grad()
                        output['loss'].backward()
                        return output['loss']

                    optimizer.step(closure)
                self.assist_rates[iter] = model.assist_rate.item()
            else:
                self.assist_rates[iter] = cfg['linesearch']['lr']
        with torch.no_grad():
            for split in organization_outputs[0]:
                self.organization_output[iter][split] = self.organization_output[iter - 1][split] + self.assist_rates[
                    iter] * self.organization_output[iter][split]
        return

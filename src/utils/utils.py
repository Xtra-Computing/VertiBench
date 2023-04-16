import pathlib
from typing import Callable

import torch
from sklearn.metrics import accuracy_score, mean_squared_error


def party_path(dataset_path, n_parties, party_id, splitter='imp', weight=1, beta=1, seed=None, type='train',
               fmt='pkl') -> str:
    assert type in ['train', 'test']
    path = pathlib.Path(dataset_path)
    if splitter == 'imp':
        # insert meta information before the file extension (extension may not be .csv)
        path = path.with_name(f"{path.stem}_party{n_parties}-{party_id}_{splitter}"
                              f"_weight{weight:.1f}{'_seed' + str(seed) if seed is not None else ''}_{type}.{fmt}")
    elif splitter == 'corr':
        path = path.with_name(f"{path.stem}_party{n_parties}-{party_id}_{splitter}"
                              f"_beta{beta:.1f}{'_seed' + str(seed) if seed is not None else ''}_{type}.{fmt}")
    else:
        raise NotImplementedError(f"Splitter {splitter} is not implemented. splitter should be in ['imp', 'corr']")
    return str(path)


def get_device_from_gpu_id(gpu_id):
    if gpu_id is None:
        return torch.device('cpu')
    else:
        return torch.device(f'cuda:{gpu_id}')


def get_metric_from_str(metric) -> Callable:
    supported_list = ['acc', 'rmse']
    assert metric in supported_list
    if metric == 'acc':
        return accuracy_score
    elif metric == 'rmse':
        return lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented. metric should be in {supported_list}")

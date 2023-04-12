"""
Split a dataset into vertical partitions.
"""

import argparse
import os
import sys
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, hmean, gmean

# add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from preprocess.FeatureEvaluator import ImportanceEvaluator, CorrelationEvaluator
from preprocess.FeatureSplitter import ImportanceSplitter, CorrelationSplitter
from dataset.LocalDataset import LocalDataset
from dataset.GlobalDataset import GlobalDataset


def split_vertical_data(X, num_parties,
                        primary_party_id=0,
                        splitter='imp',
                        weights=1,
                        beta=1,
                        corr_function='spearman',
                        seed=None):
    """
    Split a dataset into vertical partitions.

    Parameters
    ----------
    X: np.ndarray
        the dataset to be split. The last column should be the label.
    num_parties: int
        number of parties
    primary_party_id: int
        the primary party id
    splitter: str
        splitter type, should be in ['imp', 'corr']
        imp: ImportanceSplitter
        corr: CorrelationSplitter
    weights: float or list
        weights for the ImportanceSplitter
    beta: float
        beta for the CorrelationSplitter
    corr_function: str
        correlation function for the CorrelationSplitter, should be in ['pearson']
    seed: int
        random seed

    Returns
    -------
    Xs: list[np.ndarray]
        list of feature matrices
    """

    # check parameters
    assert isinstance(X, np.ndarray), "data should be a numpy array"
    assert splitter in ['imp', 'corr'], "splitter should be in ['imp', 'corr']"
    if corr_function == 'spearman':
        corr_func = lambda X: spearmanr(X).correlation
    else:
        raise NotImplementedError(f"Correlation function {corr_function} is not implemented. corr_function should be in"
                                  f" ['spearman']")
    assert num_parties > 0, "num_parties should be greater than 0"
    assert weights is None or np.all(np.array(weights) > 0), "weights should be positive"

    # split data
    if splitter == 'imp':
        splitter = ImportanceSplitter(num_parties, weights, seed)
        Xs = splitter.split(X)
    elif splitter == 'corr':
        evaluator = CorrelationEvaluator(corr_func=corr_func)
        splitter = CorrelationSplitter(num_parties, evaluator, seed)
        Xs = splitter.fit_split(X, beta=beta)
    else:
        raise NotImplementedError(f"Splitter {splitter} is not implemented. splitter should be in ['imp', 'corr']")

    return Xs


def party_path(dataset_path, n_parties, party_id, primary_party_id=0, splitter='imp', weights=1, beta=1, seed=None,
               fmt='pkl') -> str:
    path = pathlib.Path(dataset_path)
    if splitter == 'imp':
        # insert meta information before the file extension (extension may not be .csv)
        path = path.with_name(f"{path.stem}_party{n_parties}-{party_id}_primary{primary_party_id}_{splitter}"
                              f"_weight{weights:.1f}{'_seed' + str(seed) if seed is not None else ''}.{fmt}")
    elif splitter == 'corr':
        path = path.with_name(f"{path.stem}_party{n_parties}-{party_id}_primary{primary_party_id}_{splitter}"
                              f"_beta{beta:.1f}{'_seed' + str(seed) if seed is not None else ''}.{fmt}")
    else:
        raise NotImplementedError(f"Splitter {splitter} is not implemented. splitter should be in ['imp', 'corr']")
    return str(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('num_parties', type=int)
    parser.add_argument('--primary_party_id', '-p', type=int, default=0)
    parser.add_argument('--splitter', '-sp', type=str, default='imp')
    parser.add_argument('--weights', '-w', type=float)
    parser.add_argument('--beta', '-b', type=float)
    parser.add_argument('--seed', '-s', type=int, default=None)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    X, y = GlobalDataset.from_file(dataset_path).data
    Xs = split_vertical_data(X, num_parties=args.num_parties,
                                primary_party_id=args.primary_party_id,
                                splitter=args.splitter,
                                weights=args.weights,
                                beta=args.beta,
                                seed=args.seed)

    for i, X in enumerate(Xs):
        print(f"Saving party {i}: {X.shape}")
        local_dataset = LocalDataset(X, y)
        local_dataset.save(party_path(dataset_path, args.num_parties, i, args.primary_party_id, args.splitter,
                                      args.weights, args.beta, args.seed, fmt='pkl'))


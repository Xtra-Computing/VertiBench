"""
Split a dataset into vertical partitions.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, hmean, gmean
import torch

# add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.FeatureEvaluator import ImportanceEvaluator, CorrelationEvaluator
from preprocess.FeatureSplitter import ImportanceSplitter, CorrelationSplitter
from dataset.LocalDataset import LocalDataset
from dataset.GlobalDataset import GlobalDataset
from utils import PartyPath


def split_vertical_data(X, num_parties,
                        splitter='imp',
                        weights=1,
                        beta=1,
                        corr_function='spearman',
                        seed=None,
                        gpu_id=None,
                        n_jobs=1,
                        verbose=False):
    """
    Split a dataset into vertical partitions.

    Parameters
    ----------
    X: np.ndarray
        the dataset to be split. The last column should be the label.
    num_parties: int
        number of parties
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
    gpu_id: int
        gpu id for the CorrelationSplitter and CorrelationEvaluator. If None, use cpu.
    n_jobs: int
        number of jobs for the CorrelationSplitter
    verbose: bool
        whether to print verbose information

    Returns
    -------
    Xs: list[np.ndarray]
        list of feature matrices
    """

    # check parameters
    assert isinstance(X, np.ndarray), "data should be a numpy array"
    assert splitter in ['imp', 'corr'], "splitter should be in ['imp', 'corr']"
    if corr_function == 'spearman':
        corr_func = None    # use default spearmanr
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
        evaluator = CorrelationEvaluator(corr_func=corr_func, gpu_id=gpu_id)
        splitter = CorrelationSplitter(num_parties, evaluator, seed, gpu_id=gpu_id, n_jobs=n_jobs)
        Xs = splitter.fit_split(X, beta=beta, verbose=verbose)
    else:
        raise NotImplementedError(f"Splitter {splitter} is not implemented. splitter should be in ['imp', 'corr']")

    return Xs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('num_parties', type=int)
    parser.add_argument('--splitter', '-sp', type=str, default='imp', help="splitter type, should be in ['imp', 'corr']")
    parser.add_argument('--weights', '-w', type=float, default=1, help="weights for the ImportanceSplitter")
    parser.add_argument('--beta', '-b', type=float, default=1, help="beta for the CorrelationSplitter")
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--test', '-t', type=float, default=None, help="test split ratio. If None, no test split.")
    parser.add_argument('--gpu_id', '-g', type=int, default=None)
    parser.add_argument('--jobs', '-j', type=int, default=1)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.n_jobs > 1:
        warnings.warn("Multi-threading has bugs. Set n_jobs=1 instead.")
        args.n_jobs = 1

    if args.verbose:
        print(f"Loading dataset from {args.dataset_path}...")
    dataset_path = args.dataset_path
    X, y = GlobalDataset.from_file(dataset_path).data
    Xs = split_vertical_data(X, num_parties=args.num_parties,
                                splitter=args.splitter,
                                weights=args.weights,
                                beta=args.beta,
                                seed=args.seed,
                                gpu_id=args.gpu_id,
                                n_jobs=args.jobs,
                                verbose=args.verbose)

    # random shuffle Xs
    if args.verbose:
        print("Random shuffle...")
    np.random.seed(args.seed)
    random_indices = np.random.permutation(X.shape[0])
    Xs = [X[random_indices] for X in Xs]
    y = y[random_indices]

    if args.verbose:
        print("Train test splitting...")
    for i, X in enumerate(Xs):
        n_train_samples = int(X.shape[0] * (1 - args.test))
        X_train, y_train = X[:n_train_samples], y[:n_train_samples]
        X_test, y_test = X[n_train_samples:], y[n_train_samples:]
        print(f"Saving party {i}: {X.shape}")
        local_train_dataset = LocalDataset(X_train, y_train)
        path = PartyPath(dataset_path, args.num_parties, i, args.splitter, args.weights, args.beta,
                         args.seed, fmt='pkl')
        local_train_dataset.to_pickle(path.train_data)
        local_test_dataset = LocalDataset(X_test, y_test)
        local_test_dataset.to_pickle(path.test_data)

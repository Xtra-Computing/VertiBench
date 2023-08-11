"""
Split a dataset into vertical partitions.
"""

import argparse
import os
import sys
import warnings
import time

import numpy as np
import pandas as pd
from scipy.stats import hmean, gmean
import torch

# add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.FeatureEvaluator import ImportanceEvaluator, CorrelationEvaluator
from preprocess.FeatureSplitter import ImportanceSplitter, CorrelationSplitter
from dataset.LocalDataset import LocalDataset
from dataset.GlobalDataset import GlobalDataset
from utils.utils import PartyPath


def split_vertical_data(X, num_parties,
                        splitter='imp',
                        weights=1,
                        beta=1,
                        corr_func='spearmanr',
                        seed=None,
                        gpu_id=None,
                        n_jobs=1,
                        verbose=False,
                        split_image=False):
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
    corr_func: str
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
    assert weights is None or np.all(np.array(weights) > 0), "weights should be positive"

    # split data
    if splitter == 'imp':
        splitter = ImportanceSplitter(num_parties, weights, seed)
        Xs = splitter.split(X, allow_empty_party=False, split_image=split_image)     # by default, we do not allow empty parties
    elif splitter == 'corr':
        evaluator = CorrelationEvaluator(corr_func=corr_func, gpu_id=gpu_id)
        splitter = CorrelationSplitter(num_parties, evaluator, seed, gpu_id=gpu_id, n_jobs=n_jobs)
        Xs = splitter.fit_split(X, beta=beta, verbose=verbose, split_image=split_image, 
        n_elites=20*10, n_offsprings=70*10, n_mutants=10*10)


        """
        X, n_elites=20, n_offsprings=70, n_mutants=10, n_gen=100, bias=0.7, verbose=False, beta=1.,
                  term_tol=1e-4, term_period=10"""
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
    parser.add_argument('--test', '-t', type=float, default=0, help="test split ratio. If 0, no test split.")
    parser.add_argument('--gpu_id', '-g', type=int, default=None)
    parser.add_argument('--jobs', '-j', type=int, default=1)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--eval-time', '-et', action='store_true', help="whether to evaluate the time cost. If True, "
                                                                        "saving dataset will be skipped.")
    parser.add_argument('--split-image', '-si', default=False, action='store_true', help="whether to split image dataset")

    args = parser.parse_args()

    if args.jobs > 1:
        warnings.warn("Multi-threading has bugs. Set n_jobs=1 instead.")
        args.jobs = 1

    if args.verbose:
        print(f"Loading dataset from {args.dataset_path}...")
    dataset_path = args.dataset_path
    X, y = GlobalDataset.from_file(dataset_path).data

    start_time = time.time()
    Xs = split_vertical_data(X, num_parties=args.num_parties,
                                splitter=args.splitter,
                                weights=args.weights,
                                beta=args.beta,
                                seed=args.seed,
                                gpu_id=args.gpu_id,
                                n_jobs=args.jobs,
                                verbose=args.verbose,
                                split_image=args.split_image)
    end_time = time.time()
    print(f"Time cost: {end_time - start_time:.2f}s")

    if args.eval_time:
        print("Evaluation time only. Skip saving dataset.")
        sys.exit(0)    # exit without saving

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
        path = PartyPath(dataset_path, args.num_parties, i, args.splitter, args.weights, args.beta, args.seed, fmt='pkl')
        
        n_train_samples = int(X.shape[0] * (1 - args.test))
        
        X_train, y_train = X[:n_train_samples], y[:n_train_samples]
        X_test, y_test = X[n_train_samples:], y[n_train_samples:]
        
        print(f"Saving train party {i}: {X.shape}")
        local_train_dataset = LocalDataset(X_train, y_train)
        local_train_dataset.to_pickle(path.train_data)
        
        if args.test != 0:
            print(f"Saving test party {i}: {X.shape}")
            local_test_dataset = LocalDataset(X_test, y_test)
            local_test_dataset.to_pickle(path.test_data)

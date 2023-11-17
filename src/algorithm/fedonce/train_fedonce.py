

from torch.utils.tensorboard import SummaryWriter
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
import argparse
import torch
import bz2
import shutil
import zipfile
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt


import os.path
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
print(sys.path)
from src.algorithm.fedonce.utils.data_utils import load_data_cross_validation, load_data_train_test
from src.algorithm.fedonce.model.fl_model import VerticalFLModel
from src.algorithm.fedonce.model.models import FC
from src.dataset.SatelliteDataset import SatelliteDataset
from src.dataset.VFLDataset import VFLSynAlignedDataset
from src.dataset.MNISTDataset import MNISTDataset
from src.dataset.CIFAR10Dataset import CIFAR10Dataset
from src.utils import PartyPath




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parameters for dataset
    parser.add_argument('--dataset', '-d', type=str, default='covtype',
                        help="dataset to use.")
    parser.add_argument('--n_parties', '-p', type=int, default=4,
                        help="number of parties. Should be >=2")
    parser.add_argument('--primary_party', '-pp', type=int, default=0,
                        help="primary party. Should be in [0, n_parties-1]")
    parser.add_argument('--splitter', '-sp', type=str, default='imp')
    parser.add_argument('--weights', '-w', type=float, default=1, help="weights for the ImportanceSplitter")
    parser.add_argument('--beta', '-b', type=float, default=1, help="beta for the CorrelationSplitter")
    parser.add_argument('--n_classes', '-c', type=int, default=7,
                        help="number of classes. 1 for regression, 2 for binary classification,"
                             ">=3 for multi-class classification")

    # parameters for model
    parser.add_argument('--metric', '-m', type=str, default='acc',
                        help="metric to evaluate the model. Supported metrics: [acc, rmse]")
    parser.add_argument('--seed', '-s', type=int, default=0, help="random seed")

    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--agg_epochs', '-ae', type=int, default=100)
    parser.add_argument('--local_lr', '-llr', type=float, default=3e-4)
    parser.add_argument('--agg_lr', '-alr', type=float, default=1e-4)
    parser.add_argument('--local_batch_size', '-lbs', type=int, default = 128)
    parser.add_argument('--agg_batch_size', '-abs', type=int, default = 128)
    
    parser.add_argument('--gpu', '-g', type=int, default=0, help="gpu id")
    parser.add_argument('--model_type', '-mt', type=str, default='fc', help="fc or resnet18")
    args = parser.parse_args()

    # load data
    if args.dataset == 'satellite':
        # global_dataset = SatelliteGlobalDataset("data/real/satellite/clean")
        # train_global_dataset, test_global_dataset = global_dataset.split_train_test(test_ratio=0.2, seed=args.seed)
        # train_dataset = SatelliteDataset.from_global(train_global_dataset, n_jobs=20)
        # test_dataset = SatelliteDataset.from_global(test_global_dataset, n_jobs=20)
        # train_dataset.to_pickle("data/real/satellite/cache", 'train')
        # test_dataset.to_pickle("data/real/satellite/cache", 'test')
        train_dataset = SatelliteDataset.from_pickle("data/real/satellite/cache", 'train', n_parties=args.n_parties,
                                                     primary_party_id=args.primary_party, n_jobs=8)
        test_dataset = SatelliteDataset.from_pickle("data/real/satellite/cache", 'test', n_parties=args.n_parties,
                                                    primary_party_id=args.primary_party, n_jobs=8)
        model = 'resnet'
        channel = 13
        kernel_size = 9
        path = PartyPath(f"data/real/{args.dataset}", args.n_parties, 0, fmt='pkl', comm_root="log")
    elif args.dataset == 'vehicle':
        train_dataset = VFLSynAlignedDataset.from_pickle(f"data/real/{args.dataset}/processed", f'{args.dataset}', args.n_parties,
                                                      primary_party_id=args.primary_party, splitter='simple',
                                                      weight=args.weights, beta=args.beta, seed=args.seed, type='train')
        # train_dataset.shift_y_(-1)  # the original labels are in {1,2,3}, shift them to {0,1,2}
        test_dataset = VFLSynAlignedDataset.from_pickle(f"data/real/{args.dataset}/processed", f'{args.dataset}', args.n_parties,
                                                        primary_party_id=args.primary_party, splitter='simple',
                                                        weight=args.weights, beta=args.beta, seed=args.seed, type='test')
        # test_dataset.shift_y_(-1)
        model = 'mlp'
        path = PartyPath(f"data/real/{args.dataset}", args.n_parties, 0, fmt='pkl', comm_root="log")
    elif args.dataset == 'wide':
        train_dataset = VFLSynAlignedDataset.from_pickle(f"data/real/wide/processed", f'{args.dataset}',
                                                         args.n_parties,
                                                         primary_party_id=args.primary_party, splitter='simple',
                                                         weight=args.weights, beta=args.beta, seed=args.seed,
                                                         type='train')
        test_dataset = VFLSynAlignedDataset.from_pickle(f"data/real/wide/processed", f'{args.dataset}',
                                                        args.n_parties,
                                                        primary_party_id=args.primary_party, splitter='simple',
                                                        weight=args.weights, beta=args.beta, seed=args.seed,
                                                        type='test')
        model = 'mlp'
        path = PartyPath(f"data/real/{args.dataset}", args.n_parties, 0, fmt='pkl', comm_root="log")
    elif args.dataset == "mnist":
        train_dataset = MNISTDataset.from_pickle(f"data/syn/mnist", f'mnist', args.n_parties,
                                                      primary_party_id=args.primary_party, splitter=args.splitter,
                                                      weight=args.weights, beta=args.beta, seed=args.seed, type='train')
        test_dataset = MNISTDataset.from_pickle(f"data/syn/mnist", f'mnist', args.n_parties,
                                                      primary_party_id=args.primary_party, splitter=args.splitter,
                                                      weight=args.weights, beta=args.beta, seed=args.seed, type='test')
        model = 'resnet'
        channel = 1
        kernel_size = 9
        path = PartyPath(f"data/syn/mnist", args.n_parties, 0, fmt='pkl', comm_root="log")
    elif args.dataset == "cifar10":
        train_dataset = CIFAR10Dataset.from_pickle(f"data/syn/cifar10", f'cifar10', args.n_parties,
                                                      primary_party_id=args.primary_party, splitter=args.splitter,
                                                      weight=args.weights, beta=args.beta, seed=args.seed, type='train')
        test_dataset = CIFAR10Dataset.from_pickle(f"data/syn/cifar10", f'cifar10', args.n_parties,
                                                      primary_party_id=args.primary_party, splitter=args.splitter,
                                                      weight=args.weights, beta=args.beta, seed=args.seed, type='test')
        model = 'resnet'
        channel = 3
        kernel_size = 3
        path = PartyPath(f"data/syn/cifar10", args.n_parties, 0, fmt='pkl', comm_root="log")
    else:
        # Note: torch.compile() in torch 2.0 significantly harms the accuracy with little speed up
        train_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{args.dataset}", f'{args.dataset}', args.n_parties,
                                                      primary_party_id=args.primary_party, splitter=args.splitter,
                                                      weight=args.weights, beta=args.beta, seed=args.seed, type='train')
        test_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{args.dataset}", f'{args.dataset}', args.n_parties,
                                                     primary_party_id=args.primary_party, splitter=args.splitter,
                                                     weight=args.weights, beta=args.beta, seed=args.seed, type='test')
        model = 'mlp'
        path = PartyPath(f"data/syn/{args.dataset}", args.n_parties, 0, args.splitter, args.weights, args.beta,
                         args.seed, fmt='pkl', comm_root="log")

    
    if args.n_classes == 1:
        task = 'regression'
        out_dim = 1
    elif args.n_classes == 2:
        task = 'binary_classification'
        train_dataset.scale_y_()
        test_dataset.scale_y_()
        out_dim = 1
    else:
        task = 'multi_classification'
        out_dim = args.n_classes
    
    Xs_train, y_train = train_dataset.Xs, train_dataset.y
    Xs_test, y_test = test_dataset.Xs, test_dataset.y

    if args.n_classes >= 2:
        y_train = y_train.astype('long')
        y_test = y_test.astype('long')

    model_name = f"fedonce_{args.dataset}_party_{args.n_parties}_{args.splitter}_w{args.weights:.1f}_seed{args.seed}"
    name = f"{model_name}_active_{0}"
    writer = SummaryWriter("runs/{}".format(name))
    aggregate_model = VerticalFLModel(
        num_parties=args.n_parties,
        active_party_id=0,
        name=model_name,
        num_epochs=args.agg_epochs,
        num_local_rounds=args.epochs,
        local_lr=args.local_lr,
        local_hidden_layers=[100, 100],
        local_batch_size=args.local_batch_size,
        local_weight_decay=1e-5,
        local_output_size=3,
        num_agg_rounds=1,
        agg_lr=args.agg_lr,
        agg_hidden_layers=[200],
        agg_batch_size=args.agg_batch_size,
        agg_weight_decay=1e-4,
        writer=writer,
        device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
        update_target_freq=1,
        task=task,
        n_classes=out_dim,
        test_batch_size=1000,
        test_freq=1,
        cuda_parallel=False,
        n_channels=1,
        model_type='fc',
        optimizer='adam',
        privacy=None
    )
    acc, _, rmse, _ = aggregate_model.train(Xs_train, y_train, Xs_test, y_test, use_cache=False)
    
    if args.n_classes == 1:
        print(f"Final Score: {rmse}")
    else:
        print(f"Final Score: {acc}")

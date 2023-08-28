import os
import sys
import argparse

import numpy as np
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.FeatureEvaluator import ImportanceEvaluator, CorrelationEvaluator
from preprocess.FeatureSplitter import ImportanceSplitter, CorrelationSplitter
from dataset.VFLDataset import VFLSynAlignedDataset


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--seed', '-s', type=int, default=0)
    args.add_argument('--gpu', '-g', type=int, default=0)
    args.add_argument('--log-path', '-lp', type=str, default=None)
    args = args.parse_args()

    datasets = ['covtype', 'gisette', 'radar', 'letter', 'msd', 'realsim', 'epsilon']
    objective_list = ['multi:softmax', 'binary:logistic', 'multi:softmax', 'multi:softmax', 'reg:linear',
                      'binary:logistic', 'binary:logistic']
    n_classes_list = [7, 2, 7, 26, 1, 2, 2]
    n_parties = 4

    if args.log_path is not None:
        file = open(args.log_path, 'w')

    for dataset, n_classes, objective in zip(datasets, n_classes_list, objective_list):
        # get imp dataset
        is_fit = False
        corr_splitter = CorrelationSplitter(n_parties, CorrelationEvaluator(gpu_id=args.gpu), gpu_id=args.gpu)
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            try:
                # load data
                train_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{dataset}", f'{dataset}',
                                                                 n_parties,
                                                                 splitter='imp',
                                                                 weight=alpha, beta=-1, seed=args.seed,
                                                                 type='train')
                Xs = train_dataset.Xs
                X = np.concatenate(Xs, axis=1)

                # evaluate beta
                if not is_fit:
                    corr_splitter.fit(X, n_elites=20, n_offsprings=70, n_mutants=10)
                    is_fit = True

                corr_evaluator = CorrelationEvaluator(gpu_id=args.gpu)
                beta = corr_splitter.evaluate_beta(corr_evaluator.fit_evaluate(Xs))
                print(f"dataset, alpha, beta: {dataset}, {alpha}, {beta}")
                if args.log_path is not None:
                    file.write(f"{dataset}, {alpha}, {beta}\n")
                    file.flush()
            except Exception as e:
                print(e)
                continue

        # get corr dataset
        for beta in [0.0, 0.3, 0.6, 1.0]:
            try:
                # load data
                train_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{dataset}", f'{dataset}',
                                                                 n_parties,
                                                                 splitter='corr',
                                                                 weight=-1, beta=beta, seed=args.seed,
                                                                 type='train')
                Xs = train_dataset.Xs
                X = np.concatenate(Xs, axis=1)

                if n_classes == 2:
                    train_dataset.scale_y_(0, 1)

                assert is_fit

                # train a model
                if n_classes == 1:
                    model = xgb.XGBRegressor(n_estimators=50, max_depth=6, objective=objective,
                                             tree_method='gpu_hist', gpu_id=args.gpu)
                else:
                    model = xgb.XGBClassifier(n_estimators=50, max_depth=6, num_class=n_classes, objective=objective,
                                              tree_method='gpu_hist', gpu_id=args.gpu)
                model.fit(X, train_dataset.y)

                # evaluate alpha
                imp_evaluator = ImportanceEvaluator(sample_rate=0.001)
                alpha = corr_splitter.evaluate_alpha(imp_evaluator.evaluate(Xs, model.predict))
                print(f"dataset, alpha, beta: {dataset}, {alpha}, {beta}")
                if args.log_path is not None:
                    file.write(f"{dataset}, {alpha}, {beta}\n")
                    file.flush()
            except Exception as e:
                print(e)
                continue



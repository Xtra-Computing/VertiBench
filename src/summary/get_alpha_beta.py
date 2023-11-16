import os
import sys
import argparse

import numpy as np
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.FeatureEvaluator import ImportanceEvaluator, CorrelationEvaluator
from preprocess.FeatureSplitter import ImportanceSplitter, CorrelationSplitter
from dataset.VFLDataset import VFLSynAlignedDataset

# uniformly sampled alpha and beta
dataset_alpha = {
    # 46.17 0.22 30.72 0.21 33.17 0.81 0.16 7.26 20.27 3.63
    'covtype': [46.17, 0.22, 30.72, 0.21, 33.17, 0.81, 0.16, 7.26, 20.27, 3.63],
    # 0.34 3.23 1.17 10.71 10.71 4.95 0.17 41.28 0.22 14.93
    'msd': [0.34, 3.23, 1.17, 10.71, 10.71, 4.95, 0.17, 41.28, 0.22, 14.93],
    # 18.04 0.69 2.12 0.65 37.82 74.55 3.72 28.29 3.60 91.28
    'letter': [18.04, 0.69, 2.12, 0.65, 37.82, 74.55, 3.72, 28.29, 3.60, 91.28],
    # 0.21 19.17 10.57 0.97 0.78 0.20 9.56 22.70 10.42 2.41
    'gisette': [0.21, 19.17, 10.57, 0.97, 0.78, 0.20, 9.56, 22.70, 10.42, 2.41],
    # 1.45 0.17 88.73 12.94 0.34 0.17 0.18 0.18 26.50 22.87
    'radar': [1.45, 0.17, 88.73, 12.94, 0.34, 0.17, 0.18, 0.18, 26.50, 22.87],
}


dataset_beta = {
    # 0.25 0.55 1.00 0.93 0.97 0.99 0.02 0.58 0.58 0.67
    'covtype': [0.25, 0.55, 1.00, 0.93, 0.97, 0.99, 0.02, 0.58, 0.58, 0.67],
    # 0.36 0.84 0.59 0.02 0.97 0.07 0.03 0.03 0.65 0.59
    'msd': [0.36, 0.84, 0.59, 0.02, 0.97, 0.07, 0.03, 0.03, 0.65, 0.59],
    # 0.44 0.76 0.77 0.16 0.40 0.21 0.13 0.83 0.77 0.06
    'letter': [0.44, 0.76, 0.77, 0.16, 0.40, 0.21, 0.13, 0.83, 0.77, 0.06],
    # 0.80 0.40 0.62 0.38 0.33 0.27 0.85 0.63 0.20 0.34
    'gisette': [0.80, 0.40, 0.62, 0.38, 0.33, 0.27, 0.85, 0.63, 0.20, 0.34],
    # 0.77 0.48 0.90 0.92 0.59 0.61 0.59 0.28 0.50 0.95
    'radar': [0.77, 0.48, 0.90, 0.92, 0.59, 0.61, 0.59, 0.28, 0.50, 0.95],
}




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--seed', '-s', type=int, default=0)
    args.add_argument('--gpu', '-g', type=int, default=0)
    args.add_argument('--log-path', '-lp', type=str, default=None)
    args.add_argument('--decimal', type=int, default=1)
    args.add_argument('-p', '--n-parties', type=int, default=4)
    args = args.parse_args()

    # datasets = ['gisette', 'radar', 'covtype', 'letter', 'msd', 'realsim', 'epsilon']
    # objective_list = ['binary:logistic', 'multi:softmax', 'multi:softmax', 'multi:softmax', 'reg:linear',
    #                   'binary:logistic', 'binary:logistic']
    # n_classes_list = [2, 7, 7, 26, 1, 2, 2]
    datasets = ['radar', 'covtype', 'letter', 'msd', 'realsim', 'epsilon']
    objective_list = ['multi:softmax', 'multi:softmax', 'multi:softmax', 'reg:linear',
                      'binary:logistic', 'binary:logistic']
    n_classes_list = [7, 7, 26, 1, 2, 2]
    n_parties = args.n_parties

    if args.log_path is not None:
        file = open(args.log_path, 'w')

    for dataset, n_classes, objective in zip(datasets, n_classes_list, objective_list):
        if dataset not in dataset_beta:
            continue
        # get imp dataset
        is_fit = False
        corr_splitter = CorrelationSplitter(n_parties, CorrelationEvaluator(gpu_id=args.gpu), gpu_id=args.gpu)
        for alpha in dataset_alpha[dataset]:
            try:
                # load data
                train_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{dataset}", f'{dataset}',
                                                                 n_parties,
                                                                 splitter='imp',
                                                                 weight=alpha, beta=-1, seed=args.seed,
                                                                 type='train', decimal=args.decimal)
                Xs = train_dataset.Xs
                X = np.concatenate(Xs, axis=1)

                # evaluate beta
                if not is_fit:
                    # corr_splitter.fit(X, n_elites=20, n_offsprings=70, n_mutants=10, n_gen=1)
                    corr_splitter.fit(X)
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
        for beta in dataset_beta[dataset]:
            try:
                # load data
                train_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{dataset}", f'{dataset}',
                                                                 n_parties,
                                                                 splitter='corr',
                                                                 weight=-1, beta=beta, seed=args.seed,
                                                                 type='train', decimal=args.decimal)
                Xs = train_dataset.Xs
                X = np.concatenate(Xs, axis=1)

                if n_classes == 2:
                    train_dataset.scale_y_(0, 1)

                # evaluate beta
                if not is_fit:
                    corr_splitter.fit(X, n_elites=20, n_offsprings=70, n_mutants=10)
                    is_fit = True

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



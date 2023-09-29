import os
import sys
import argparse

import numpy as np
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.FeatureEvaluator import ImportanceEvaluator, CorrelationEvaluator
from preprocess.FeatureSplitter import ImportanceSplitter, CorrelationSplitter
from dataset.VFLDataset import VFLSynAlignedDataset


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





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--seed', '-s', nargs='+', type=int, default=[0])
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

    for seed in args.seed:
        for dataset, n_classes, objective in zip(datasets, n_classes_list, objective_list):
            if dataset not in dataset_alpha:
                continue

            # get imp dataset
            is_fit = False
            corr_splitter = CorrelationSplitter(n_parties, CorrelationEvaluator(gpu_id=args.gpu), gpu_id=args.gpu)

            for alpha in [1.0]:
                try:
                    # load data
                    train_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{dataset}", f'{dataset}',
                                                                     n_parties,
                                                                     splitter='imp',
                                                                     weight=alpha, beta=-1, seed=seed,
                                                                     type='train', decimal=args.decimal)
                    Xs = train_dataset.Xs
                    X = np.concatenate(Xs, axis=1)

                    # evaluate alpha
                    if not is_fit:
                        # corr_splitter.fit(X, n_elites=20, n_offsprings=70, n_mutants=10, n_gen=1)
                        corr_splitter.fit(X)
                        is_fit = True

                    # train a model
                    if n_classes == 1:
                        model = xgb.XGBRegressor(n_estimators=50, max_depth=6, objective=objective,
                                                 tree_method='gpu_hist', gpu_id=args.gpu)
                    else:
                        model = xgb.XGBClassifier(n_estimators=50, max_depth=6, num_class=n_classes,
                                                  objective=objective,
                                                  tree_method='gpu_hist', gpu_id=args.gpu)
                    model.fit(X, train_dataset.y)

                    # evaluate alpha
                    imp_evaluator = ImportanceEvaluator(sample_rate=0.001)
                    alpha = corr_splitter.evaluate_alpha(imp_evaluator.evaluate(Xs, model.predict))

                    # evaluate beta
                    corr_evaluator = CorrelationEvaluator(gpu_id=args.gpu)
                    beta = corr_splitter.evaluate_beta(corr_evaluator.fit_evaluate(Xs))
                    print(f"dataset, alpha, beta: {dataset}, {alpha}, {beta}")
                    if args.log_path is not None:
                        file.write(f"{dataset}, {alpha}, {beta}\n")
                        file.flush()
                except Exception as e:
                    print(e)
                    continue


import matplotlib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from preprocess.FeatureSplitter import CorrelationSplitter, ImportanceSplitter
from preprocess.FeatureEvaluator import CorrelationEvaluator, ImportanceEvaluator


# matplotlib.use('Agg')


def test_corr_splitter():
    # Generate a synthetic dataset using sklearn
    # X1, y1 = make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, n_classes=2,
    #                              random_state=0, shuffle=True)
    # X2, y2 = make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, n_classes=2,
    #                              random_state=1, shuffle=True)
    # X3, y3 = make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, n_classes=2,
    #                              random_state=2, shuffle=True)
    # X = np.concatenate((X1, X2, X3), axis=1)
    X = load_svmlight_file("data/syn/msd/YearPredictionMSD")[0].toarray()
    # X = load_svmlight_file("data/syn/covtype/covtype")[0].toarray()
    # scale each feature to [0, 1]
    X = MinMaxScaler().fit_transform(X)
    print(X.shape)
    corr = spearmanr(X).correlation     # this correlation is only used for plotting

    corr_evaluator = CorrelationEvaluator(gamma=1)
    corr_spliter = CorrelationSplitter(num_parties=3, evaluator=corr_evaluator)

    corr_spliter.fit(X, verbose=False)
    print(f"Min mcor: {corr_spliter.min_mcor}, Max mcor: {corr_spliter.max_mcor}")
    for beta in [0, 0.33, 0.66, 1]:
        Xs = corr_spliter.split(X, beta=beta, verbose=False)
        eval_mcor = corr_evaluator.evaluate(Xs)

        corr_perm = corr[corr_spliter.best_permutation, :][:, corr_spliter.best_permutation]
        print(f"{beta=}: best_mcor={corr_spliter.best_mcor:.4f}, eval_mcor={eval_mcor:.4f}, "
              f"best_error={corr_spliter.best_error:.6f}, features_per_party={corr_spliter.best_feature_per_party}")

        # plot permuted correlation matrix in heatmap
        plt.figure()
        # add sigmoid to make the color more distinguishable
        # corr_perm = np.exp(corr_perm) - 1
        plt.imshow(corr_perm, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'beta={beta}, best_mcor={corr_spliter.best_mcor:.4f}')
        plt.show()


def test_importance_splitter_diff_alpha(n_rounds=100):

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, n_repeated=0,
                               n_classes=2, random_state=0, shuffle=True)
    X = MinMaxScaler().fit_transform(X)
    model = LogisticRegression()
    model.fit(X, y)

    # different weights for each party
    importance_per_w = []
    importance_per_w_std = []
    w_range = np.arange(1, 4, 0.5)
    for w in w_range:
        importance_summary = []
        for i in tqdm(range(n_rounds)):
            splitter = ImportanceSplitter(num_parties=3, weights=[3, 2, w], seed=i)
            Xs = splitter.split(X)
            evaluator = ImportanceEvaluator(model.predict, sample_rate=0.01, seed=i)
            party_importance = evaluator.evaluate(Xs)
            # print(f"Party importance {party_importance}")
            importance_summary.append(party_importance)
        importance_summary = np.array(importance_summary)
        mean_importance = np.mean(importance_summary, axis=0)
        std_importance = np.std(importance_summary, axis=0)
        print(f"Mean importance: {mean_importance}")
        print(f"Std importance: {std_importance}")
        importance_per_w.append(mean_importance)
        importance_per_w_std.append(std_importance)
    importance_per_w = np.array(importance_per_w).T
    importance_per_w_std = np.array(importance_per_w_std).T
    # plot the importance as w changes
    plt.figure()
    plt.plot(w_range, importance_per_w[0], marker='o')
    plt.plot(w_range, importance_per_w[1], marker='x')
    plt.plot(w_range, importance_per_w[2], marker='^')
    plt.legend([r"Party 1 ($\alpha_1=3$)", r"Party 2 ($\alpha_2=2$)", r"Party 3"])
    plt.xlabel(r"Weight of Party 3 ($\alpha_3$)")
    plt.ylabel("Shapley importance")
    plt.title(r"Shapley importance as $\alpha_3$ changes")
    plt.show()


def test_importance_splitter_same_alpha(n_rounds=100):
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, n_repeated=0,
                               n_classes=2, random_state=0, shuffle=True)
    X = MinMaxScaler().fit_transform(X)
    model = LogisticRegression()
    model.fit(X, y)

    # same but varies weights for each party
    mean_importance_per_w = []
    std_importance_per_w = []
    w_range = [0.1, 0.5, 1, 2, 5, 10]
    for w in w_range:
        importance_summary = []
        for i in tqdm(range(n_rounds)):
            splitter = ImportanceSplitter(num_parties=10, weights=w, seed=i)
            Xs = splitter.split(X)
            evaluator = ImportanceEvaluator(model.predict, sample_rate=0.02, seed=i)
            party_importance = evaluator.evaluate(Xs)
            # print(f"Party importance {party_importance}")
            importance_summary.append(party_importance)
        importance_summary = np.array(importance_summary)
        mean_importance = np.mean(importance_summary, axis=0)
        std_importance = np.mean(np.std(importance_summary, axis=1))  # std among parties
        print(f"Mean importance: {mean_importance}")
        print(f"Std importance: {std_importance}")
        mean_importance_per_w.append(mean_importance)
        std_importance_per_w.append(std_importance)
    mean_importance_per_w = np.array(mean_importance_per_w).T
    std_importance_per_w = np.array(std_importance_per_w)
    # plot the mean importance and std as w changes
    plt.figure()
    plt.plot(w_range, std_importance_per_w, marker='o')
    # plt.legend([rf"Party {i} ($\alpha_{i}$=$\alpha$)" for i in range(1, 4)])
    plt.xlabel(r"Weight of each party ($\alpha$)")
    plt.ylabel("Standard variance")
    plt.title(r"Standard variance of Shapley importance across parties")
    plt.show()


def test_weight_different_alpha():
    w_range = np.arange(1, 4, 0.5)
    w_party1_base = 3
    w_party2_base = 2
    scale_ws = []
    for w in w_range:
        w_party1_scale = w_party1_base / (w_party1_base + w_party2_base + w)
        w_party2_scale = w_party2_base / (w_party1_base + w_party2_base + w)
        w_party3_scale = w / (w_party1_base + w_party2_base + w)
        scale_ws.append([w_party1_scale, w_party2_scale, w_party3_scale])
    scale_ws = np.array(scale_ws)
    plt.figure()
    plt.plot(w_range, scale_ws[:, 0], marker='o')
    plt.plot(w_range, scale_ws[:, 1], marker='x')
    plt.plot(w_range, scale_ws[:, 2], marker='^')
    plt.legend([r"Party 1 ($\alpha_1$)", r"Party 2 ($\alpha_2$)", r"Party 3 ($\alpha_3$)"])
    plt.xlabel(r"Weight of Party 3 ($\alpha_3$)")
    plt.ylabel("Scaled weight")
    plt.title(r"Scaled weight of each party as $\alpha_3$ changes")
    plt.show()


if __name__ == '__main__':
    # test_importance_splitter_same_alpha(2000)
    # test_importance_splitter_diff_alpha(2000)
    # test_weight_different_alpha()
    test_corr_splitter()

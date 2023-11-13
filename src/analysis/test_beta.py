import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import make_classification, make_regression
from scipy.stats import spearmanr
from xgboost import XGBClassifier, XGBRegressor

# import the path of the project
sys.path.append(os.path.abspath("src"))

from preprocess.FeatureSplitter import CorrelationSplitter, ImportanceSplitter
from preprocess.FeatureEvaluator import CorrelationEvaluator, ImportanceEvaluator
from dataset.WideDataset import WideDataset
from dataset.SatelliteDataset import SatelliteDataset


# Generate two binary classification datasets with correlated features and a target

def generate_independent_Xs(n_informative, n_features1, n_features2):

    X1, y1 = make_classification(n_samples=10000, n_features=n_features1, n_informative=n_informative,
                                 n_redundant=n_features1 - n_informative,
                                 n_repeated=0, n_classes=2, random_state=0, shuffle=True)
    X2, y2 = make_classification(n_samples=10000, n_features=n_features2, n_informative=n_informative,
                                 n_redundant=n_features2 - n_informative,
                                 n_repeated=0, n_classes=2, random_state=1, shuffle=True)
    X = np.concatenate([X1, X2], axis=1)
    return X

def test_correlation_by_moving_features(X, n_features1, n_features2, pcor_func, save_path=None, label='Pcor'):

    # set a larger font size for plots
    plt.rcParams.update({'font.size': 16})

    n_features = n_features1 + n_features2
    corr = spearmanr(X).correlation

    X1_ids = np.arange(n_features)[:n_features1]
    X2_ids = np.arange(n_features)[n_features1:]
    n_X1s = np.arange(n_features1 + 1)
    pcor1s = []
    pcor2s = []
    pcor12s = []
    for n_X1 in n_X1s:
        assert n_X1 <= n_features1
        X1_ids_1 = X1_ids[:n_X1]
        X2_ids_1 = X2_ids[:n_features1-n_X1]
        ids_1 = np.concatenate([X1_ids_1, X2_ids_1])
        X1_ids_2 = X1_ids[n_X1:]
        X2_ids_2 = X2_ids[n_features1-n_X1:]
        ids_2 = np.concatenate([X1_ids_2, X2_ids_2])

        corr1 = corr[ids_1, :][:, ids_1]
        corr2 = corr[ids_2, :][:, ids_2]
        corr_12 = corr[ids_1, :][:, ids_2]

        # # initialize a new correlation matrix with an identity matrix
        # corr_masked = np.zeros(corr.shape)
        # corr_masked[np.ix_(ids_1, ids_2)] = corr[np.ix_(ids_1, ids_2)]
        # corr_masked[np.ix_(ids_1, ids_2)] = corr[np.ix_(ids_1, ids_2)]

        pcor1 = pcor_func(corr1)
        pcor2 = pcor_func(corr2)
        pcor12 = pcor_func(corr_12)
        # print(f"n_X1: {n_X1}, pcor1: {pcor1:.4f}, pcor2: {pcor2:.4f}, pcor12: {pcor12:.8f}")
        pcor1s.append(pcor1)
        pcor2s.append(pcor2)
        pcor12s.append(pcor12)


    # plot the trend of pcor1, pcor2, and pcor12 w.r.t. n_X1
    ax = plt.gca()
    ax.invert_xaxis()
    fig = plt.gcf()
    plt.xlabel("Party $P_1$: number of features from $X_1$")
    plt.ylabel(f"Correlation Index ({label})")
    ax.plot(n_X1s, pcor1s, label=f"{label}1-1", marker="o")
    ax.plot(n_X1s, pcor2s, label=f"{label}2-2", marker="^")
    ax.plot(n_X1s, pcor12s, label=f"{label}1-2", marker="x")
    # add an x-axis to show the number of informative features from X2

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax.set_xticks(n_X1s)
    ax.set_xticklabels(n_X1s)
    ax2.set_xticks(n_X1s)
    ax2.set_xticklabels(n_features1 - n_X1s)
    ax2.set_xlabel("Party $P_2$: number of features from $X_1$")

    # plt.title(f"(Party 1: {n_features1} features, Party 2: {n_features2} features)\n(X1: {n_features1} features, X2: {n_features2} features)")

    ax.legend()
    plt.show()

    if save_path is not None:
        fig.savefig(save_path)
    plt.close()


def pcor_eigen(corr):
    """Summarize the correlation matrix corr"""
    assert corr.shape[0] == corr.shape[1]   # eigenvalues are only defined for square matrices
    d = corr.shape[0]
    eigen_values = np.linalg.eigvals(corr)
    score = np.std(eigen_values, ddof=1) / np.sqrt(d)
    return score

def pcor_singular(corr):
    """Summarize the correlation matrix corr"""
    d = min(corr.shape[0], corr.shape[1])
    singular_values = np.linalg.svd(corr)[1]
    score = np.std(singular_values, ddof=1) / np.sqrt(d)
    return score

def test_corrlation_by_splitting_synthetic():
    # generate three independent datasets
    X1 = \
    make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, random_state=0, shuffle=True)[0]
    X2 = \
    make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, random_state=1, shuffle=True)[0]
    X3 = \
    make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, random_state=2, shuffle=True)[0]

    # concatenate them to a global dataset
    X = np.concatenate([X1, X2, X3], axis=1)
    np.random.shuffle(X.T)

    # original three parties
    corr_evaluator_original = CorrelationEvaluator(gpu_id=0)
    score_original = corr_evaluator_original.fit_evaluate([X1, X2, X3])
    corr_evaluator_original.visualize("fig/pcor-original.png", value=score_original)

    # fit the correlation splitter to get the lowest and highest correlation scores
    corr_splitter = CorrelationSplitter(num_parties=3, evaluator=CorrelationEvaluator(gpu_id=0), gpu_id=0)
    corr_splitter.fit(X, n_elites=200, n_offsprings=700, n_mutants=100, verbose=True)

    # split the correlation matrix with different beta
    Xs_00 = corr_splitter.split(X, beta=0.0, n_elites=200, n_offsprings=700, n_mutants=100, verbose=True)
    corr_evaluator_00 = CorrelationEvaluator(gpu_id=0)
    score_00 = corr_evaluator_00.fit_evaluate(Xs_00)
    corr_evaluator_00.visualize("fig/pcor-split-beta0.0.png", value=score_00, fontsize=28)

    Xs_05 = corr_splitter.split(X, beta=0.5, n_elites=200, n_offsprings=700, n_mutants=100, verbose=True)
    corr_evaluator_05 = CorrelationEvaluator(gpu_id=0)
    score_05 = corr_evaluator_05.fit_evaluate(Xs_05)
    corr_evaluator_05.visualize("fig/pcor-split-beta0.5.png", value=score_05, fontsize=28)

    X_10 = corr_splitter.split(X, beta=1.0, n_elites=200, n_offsprings=700, n_mutants=100, verbose=True)
    corr_evaluator_10 = CorrelationEvaluator(gpu_id=0)
    score_10 = corr_evaluator_10.fit_evaluate(X_10)
    corr_evaluator_10.visualize("fig/pcor-split-beta1.0.png", value=score_10, fontsize=28)

    # correlation score of randomly shuffled the features
    X_shuffle = X.copy()
    np.random.shuffle(X_shuffle.T)
    Xs_shuffle = np.split(X_shuffle, 3, axis=1)
    corr_evaluator_shuffle = CorrelationEvaluator(gpu_id=0)
    score_shuffle = corr_evaluator_shuffle.fit_evaluate(Xs_shuffle)
    corr_evaluator_shuffle.visualize("fig/pcor-shuffle.png", value=score_shuffle, fontsize=28)



if __name__ == '__main__':
    np.random.seed(0)
    # test two functions
    X = generate_independent_Xs(n_informative=6, n_features1=10, n_features2=10)
    test_correlation_by_moving_features(X, n_features1=10, n_features2=10, pcor_func=pcor_eigen,
                                        save_path="fig/pcor-eigen-10-10.png", label='mcor')
    test_correlation_by_moving_features(X, n_features1=10, n_features2=10, pcor_func=pcor_singular,
                                        save_path="fig/pcor-singular-10-10.png")
    X2 = generate_independent_Xs(n_informative=6, n_features1=10, n_features2=20)
    test_correlation_by_moving_features(X2, n_features1=10, n_features2=20, pcor_func=pcor_singular,
                                    save_path="fig/pcor-singular-10-20.png")
    # test_corrlation_by_splitting_synthetic()


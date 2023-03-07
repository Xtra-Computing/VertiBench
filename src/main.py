import matplotlib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

import torch

from preprocess.FeatureSplitter import CorrelationSplitter, ImportanceSplitter


# matplotlib.use('Agg')


def test_corr_splitter():
    # Generate a synthetic dataset using sklearn
    X1, y1 = make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, n_classes=2,
                                 random_state=0, shuffle=True)
    X2, y2 = make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, n_classes=2,
                                 random_state=1, shuffle=True)
    X3, y3 = make_classification(n_samples=10000, n_features=10, n_informative=3, n_redundant=7, n_classes=2,
                                 random_state=2, shuffle=True)
    X = np.concatenate((X1, X2, X3), axis=1)
    corr = spearmanr(X).correlation

    corr_spliter = CorrelationSplitter(num_parties=3)
    corr_spliter.fit(X)
    print(f"Min mcor: {corr_spliter.min_mcor}, Max mcor: {corr_spliter.max_mcor}")
    for beta in [0, 0.2, 0.5, 0.8, 1]:
        Xs = corr_spliter.split(X, beta=beta)

        corr_perm = corr[corr_spliter.best_permutation, :][:, corr_spliter.best_permutation]
        print(f"{beta=}: best_mcor={corr_spliter.best_mcor:.4f}, best_error={corr_spliter.best_error},"
              f"features_per_party={corr_spliter.best_feature_per_party}")

        # plot permuted correlation matrix in heatmap
        plt.figure()
        # add sigmoid to make the color more distinguishable
        corr_perm = 1 / (1 + np.exp(-corr_perm))
        plt.imshow(corr_perm, cmap='viridis')
        plt.colorbar()
        plt.title(f'beta={beta}, best_mcor={corr_spliter.best_mcor:.4f}')
        plt.show()


if __name__ == '__main__':
    test_corr_splitter()


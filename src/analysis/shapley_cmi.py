from collections import Counter
import sys
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from scipy.stats import entropy


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from alpha_importance import plot_alpha_vs_mean_imp, plot_alpha_vs_std

########### Functions used to calculate the Shapley-CMI ###########

def discretize(X, category=5):
    """
    Discretize the features into categories.
    :param X: Features.
    :param category: Number of categories.
    :return:
    """
    X_category = np.zeros(X.shape)
    for i in range(X.shape[1]):
        X_category[:, i] = pd.cut(X[:, i], category, labels=list(range(category))).to_numpy()
    return X_category.astype(int)


def our_entropy(labels):     # H(A)
    pro_dict = Counter(labels)
    s = sum(pro_dict.values())
    probs = np.array([i/s for i in pro_dict.values()])
    return - probs.dot(np.log(probs))


def MI_(s1,s2):     # Mutual Information
    s_s_1=["%s%s"%(i,j) for i,j in zip(s1,s2)]
    MI_1=our_entropy(s1)+our_entropy(s2)-our_entropy(s_s_1)
    return MI_1


def N_MI(s1,s2):    # Normalized Mutual Information
    MI_1 = MI_(s1,s2)
    NMI_1 = MI_1/(our_entropy(s1)*our_entropy(s2))**0.5
    return NMI_1

########################################

def shapley_CMI(X, y, n_rounds=10000, seed=None):
    """
    Calculate the Shapley-CMI for each feature
    :param X: Discretized features.
    :param y: Discretized labels. (only works for classification)
    :param n_rounds: Number of rounds to calculate the Shapley-CMI.
    :param seed: Random seed.
    :return:
    """
    feature_ids = np.arange(X.shape[1])
    rng = np.random.default_rng(seed)
    perms = rng.permuted(np.tile(feature_ids, n_rounds).reshape(n_rounds, feature_ids.size), axis=1)

    contributions = np.zeros((n_rounds, feature_ids.size))
    for i in range(n_rounds):
        perm = perms[i]
        current_feature_set = []
        current_MI = 0
        for feature_id in perm:
            current_feature_set.append(feature_id)
            X_new = X[:, current_feature_set]
            X_new_tuple = [tuple(x) for x in X_new]
            new_MI = MI_(list(y), list(X_new_tuple))
            contribution = new_MI - current_MI
            contributions[i, feature_id] = contribution
            current_MI = new_MI
        i += 1

        if i % 100 == 0:
            shapley_cmi = np.mean(contributions[:i], axis=0)
            print(f"Round {i}: {shapley_cmi}")

    shapley_cmi = np.mean(contributions, axis=0)
    print(f"feature_importance = {shapley_cmi}")

    return shapley_cmi


if __name__ == '__main__':
    # # load data
    # X, y = load_svmlight_file("data/syn/letter/letter.libsvm")
    # X = X.toarray()
    # y = y.astype(int)
    # print(X.shape, y.shape)
    #
    # # discretize data
    # X_category = discretize(X, category=5)
    #
    # feature_importance = shapley_CMI(X_category, y)

    """
feature_importance = [0.08420452 0.10850604 0.10432603 0.0842274  0.11213454 0.18339879
 0.24893719 0.28429343 0.30225421 0.2485479  0.28416086 0.28622037
 0.19588112 0.19678296 0.28227749 0.18349692]
    """

    feature_importance = [0.08420452, 0.10850604, 0.10432603, 0.0842274,  0.11213454, 0.18339879,
 0.24893719 ,0.28429343, 0.30225421, 0.2485479 , 0.28416086, 0.28622037,
 0.19588112, 0.19678296, 0.28227749, 0.18349692]
    feature_importance = np.array(feature_importance)

    # plot alpha vs mean importance
    # plot_alpha_vs_mean_imp(feature_importance, metric="Shapley-CMI")
    plot_alpha_vs_std(feature_importance, metric="Shapley-CMI")

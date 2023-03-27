from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, hmean, gmean
import shap


class ImportanceEvaluator:
    """
    Importance evaluator for VFL datasets
    """
    def __init__(self, sample_rate=0.01, seed=0):
        """
        :param sample_rate: [float] sample rate of the dataset to calculate Shaplley values
        :param seed: [int] random seed of sampling
        """
        self.sample_rate = sample_rate
        self.seed = seed
        np.random.seed(seed)  # This is for reproducibility of Permutation explainer

    @staticmethod
    def check_data(Xs):
        """
        Check if the data is valid for correlation evaluation.
        :param Xs: [List|Tuple] list of feature matrices of each party
        :return: [bool] True if the data is valid
        """
        n_features_on_party = [X.shape[1] for X in Xs]
        n_parties = len(n_features_on_party)
        assert n_parties > 1, "ImportanceEvaluator only works for multi-party VFL datasets"
        assert all([X.shape[0] == Xs[0].shape[0] for X in Xs]), \
            "The number of samples should be the same for all parties"
        return n_features_on_party

    def evaluate_feature(self, X, model: callable):
        """
        Evaluate the importance of features in a dataset
        :param X: feature matrix
        :param model: [callable] model to be evaluated
        :return: [np.ndarray] sum of importance on each party
        """
        # calculate Shapley values for each feature
        explainer = shap.explainers.Permutation(model, X)
        sample_size = int(self.sample_rate * X.shape[0])
        X_sample = shap.sample(X, sample_size, random_state=self.seed)
        shap_values = explainer(X_sample).values

        importance_by_feature = np.sum(np.abs(shap_values), axis=0)
        assert importance_by_feature.shape[0] == X.shape[1], "The number of features should be the same"
        return importance_by_feature

    def evaluate(self, Xs, model: callable):
        """
        Evaluate the importance of features in VFL datasets
        :param Xs: [list] list of feature matrices
        :param model: [callable] model to be evaluated
        :return: [np.ndarray] sum of importance on each party
        """
        n_features_on_party = self.check_data(Xs)
        X = np.concatenate(Xs, axis=1)

        # calculate Shapley values for each feature
        explainer = shap.explainers.Permutation(model, X)
        sample_size = int(self.sample_rate * X.shape[0])
        X_sample = shap.sample(X, sample_size, random_state=self.seed)
        shap_values = explainer(X_sample).values

        # sum up the importance of each feature on each party
        party_cutoff = np.cumsum(n_features_on_party)
        party_cutoff = np.insert(party_cutoff, 0, 0)
        importance = np.zeros(len(n_features_on_party))
        for i in range(len(n_features_on_party)):
            party_shapley = np.sum(np.abs(shap_values[:, party_cutoff[i]:party_cutoff[i + 1]]), axis=1)
            importance[i] = np.mean(np.abs(party_shapley))
        assert len(importance) == len(n_features_on_party), \
            "The length of importance should be the same as the number of parties"
        return importance


class CorrelationEvaluator:
    """
    Correlation evaluator for VFL datasets
    """
    def __init__(self, corr_func: callable = lambda X: spearmanr(X).correlation, gamma=1.0):
        """
        :param corr_func: [callable] function to calculate the correlation between two features
        :param gamma: [float] weight of the inner-party correlation score
        """
        self.corr_func = corr_func
        self.gamma = gamma

    @staticmethod
    def mcor_singular(corr):
        """
        Calculate the overall correlation score of a correlation matrix using the variance of singular values.
        This is an example of corr
        [ v1 v1 ]
        [ v1 v1 ]
        :param corr:
        :return:
        """
        # replace all NaN values with 0 (NaN values are caused by constant features, the covariance of which is 0)
        corr = np.nan_to_num(corr, nan=0)

        # d = corr.shape[0]
        singular_values = np.linalg.svd(corr)[1]
        assert (singular_values > 0).any()  # singular values should be positive by definition
        score = np.std(singular_values)
        return score

    def _get_inner_and_inter_corr(self, corr, n_features_on_party, mcor_func):
        """
        Calculate the inner-party and inter-party correlation matrices.
        :param corr: [np.ndarray] correlation matrix
        :param n_features_on_party: [list] number of features on each party
        :param mcor_func: [callable] function to calculate the overall correlation score of a correlation matrix
        :return: [1D np.ndarray] inner-party correlation scores (size: number of parties)
                 [2D np.ndarray] inter-party correlation scores (size: number of parties x (number of parties - 1))
        """
        n_parties = len(n_features_on_party)
        assert sum(n_features_on_party) == corr.shape[0] == corr.shape[1], \
            f"The number of features on each party should be the same as the size of the correlation matrix," \
            "but got {sum(n_features_on_party)} != {corr.shape[0]} != {corr.shape[1]}"
        corr_cut_points = np.cumsum(n_features_on_party)
        corr_cut_points = np.insert(corr_cut_points, 0, 0)

        inner_mcors = []
        inter_mcors = [[] for _ in range(n_parties)]
        for i in range(n_parties):
            for j in range(n_parties):
                start_i = corr_cut_points[i]
                end_i = corr_cut_points[i + 1]
                start_j = corr_cut_points[j]
                end_j = corr_cut_points[j + 1]
                if i == j:
                    inner_mcors.append(mcor_func(corr[start_i:end_i, start_j:end_j]))
                else:
                    inter_mcors[i].append(mcor_func(corr[start_i:end_i, start_j:end_j]))

        return np.array(inner_mcors), np.array(inter_mcors)

    def overall_corr_score_diff(self, corr, n_features_on_party, mcor_func: callable = mcor_singular):
        """
        Calculate the correlation score of a correlation matrix. This is a three-party example of corr:
        [ v1 v1 .  .  .  .  ]
        [ v1 v1 .  .  .  .  ]
        [ .  .  v2 v2 .  .  ]
        [ .  .  v2 v2 .  .  ]
        [ .  .  .  .  v3 v3 ]
        [ .  .  .  .  v3 v3 ]
        v1, v2, v3 represent inner-party correlation matrices of each party. This function only evaluates the
        arithmetic mean of differences between the inner-party correlation scores and the inter-party correlation
        scores for each party.
        :param corr: [np.ndarray] correlation matrix
        :param n_features_on_party: [list] number of features on each party
        :param mcor_func: [callable] function to calculate the correlation score of a correlation matrix
        :return: [float] correlation score
        """

        inner_mcors, inter_mcors = self._get_inner_and_inter_corr(corr, n_features_on_party, mcor_func)

        inter_mean_mcor = np.mean(np.array(inter_mcors).flatten())
        inner_mean_mcor = np.mean(np.array(inner_mcors).flatten())
        assert inter_mean_mcor >= 0, f"inter_mean_mcor: {inter_mean_mcor} should be non-negative"
        return inter_mean_mcor - self.gamma * inner_mean_mcor

    def overall_corr_score_ratio(self, corr, n_features_on_party, mcor_func: callable = mcor_singular):
        """
        Calculate the correlation score of a correlation matrix. This is a three-party example of corr:
        [ v1 v1 .  .  .  .  ]
        [ v1 v1 .  .  .  .  ]
        [ .  .  v2 v2 .  .  ]
        [ .  .  v2 v2 .  .  ]
        [ .  .  .  .  v3 v3 ]
        [ .  .  .  .  v3 v3 ]
        v1, v2, v3 represent inner-party correlation matrices of each party. This function only evaluates the
        mean of ratios between the inner-party correlation scores and the inter-party correlation
        scores for each party.
        :param corr: [np.ndarray] correlation matrix
        :param n_features_on_party: [list] number of features on each party
        :param mcor_func: [callable] function to calculate the correlation score of a correlation matrix
        :return: [float] correlation score
        """

        inner_mcors, inter_mcors = self._get_inner_and_inter_corr(corr, n_features_on_party, mcor_func)

        inter_inner_ratio_summary = []
        for inner_mcor_i, inter_mcors_i in zip(inner_mcors, inter_mcors):
            inter_inner_ratio = inter_mcors_i / (inner_mcor_i + 1e-8)
            inter_inner_ratio_summary.append(np.mean(inter_inner_ratio))
        return np.mean(inter_inner_ratio_summary)

    @staticmethod
    def check_data(Xs):
        """
        Check if the data is valid for correlation evaluation.
        :param Xs: [List|Tuple] list of feature matrices of each party
        :return: [bool] True if the data is valid
        """
        n_features_on_party = [X.shape[1] for X in Xs]
        n_parties = len(n_features_on_party)
        assert n_parties > 1, "CorrelationEvaluator only works for multi-party VFL datasets"
        assert all([X.shape[0] == Xs[0].shape[0] for X in Xs]), \
            "The number of samples should be the same for all parties"
        return n_features_on_party

    def evaluate(self, Xs):
        """
        Evaluate the correlation score of a vertical federated learning dataset.
        :param Xs: [List|Tuple] list of feature matrices of each party
        :return: [float] correlation score
        """
        n_features_on_party = self.check_data(Xs)
        corr = self.corr_func(np.concatenate(Xs, axis=1))
        return self.overall_corr_score_diff(corr, n_features_on_party)

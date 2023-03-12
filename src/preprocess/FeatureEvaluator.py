import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap


class ImportanceEvaluator:
    """
    Importance evaluator for VFL datasets
    """
    def __init__(self, model, sample_rate=0.01, seed=0):
        """
        :param model: [callable] model to be evaluated
        :param sample_rate: [float] sample rate of the dataset to calculate Shaplley values
        :param seed: [int] random seed of sampling
        """
        self.model = model
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

    def evaluate(self, Xs):
        """
        Evaluate the importance of features in VFL datasets
        :param Xs: [list] list of feature matrices
        :return: [np.ndarray] sum of importance on each party
        """
        n_features_on_party = self.check_data(Xs)
        X = np.concatenate(Xs, axis=1)

        # calculate Shapley values for each feature
        explainer = shap.explainers.Permutation(self.model, X)
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
    def __init__(self, gamma=0.,
                 corr_func: callable = lambda X: spearmanr(X).correlation):
        """
        :param gamma: [float] the weight of inter-party correlation
        :param corr_func: [callable] function to calculate the correlation between two features
        :param corr: [np.ndarray] correlation matrix
        :param n_features_on_party: [list] number of features on each party
        """
        self.gamma = gamma
        self.corr_func = corr_func

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
        d = corr.shape[0]
        singular_values = np.linalg.svd(corr)[1]
        assert (singular_values > 0).any()  # singular values should be positive by definition
        score = 1 / np.sqrt(d) * np.sqrt((1 / (d - 1)) * np.sum((singular_values - np.average(singular_values)) ** 2))
        return score

    def overall_corr_score(self, corr, n_features_on_party, mcor_func: callable = mcor_singular):
        """
        Calculate the correlation score of a correlation matrix. This is a three-party example of corr:
        [ v1 v1 .  .  .  .  ]
        [ v1 v1 .  .  .  .  ]
        [ .  .  v2 v2 .  .  ]
        [ .  .  v2 v2 .  .  ]
        [ .  .  .  .  v3 v3 ]
        [ .  .  .  .  v3 v3 ]
        v1, v2, v3 represent inner-party correlation matrices of each party.
        Apart from inner-party correlation, this function also evaluates inter-party correlation.
        The overall score is the mean of the inner-party minus the inter-party correlation.
        :param corr: [np.ndarray] correlation matrix
        :param n_features_on_party: [list] number of features on each party
        :param mcor_func: [callable] function to calculate the correlation score of a correlation matrix
        :param gamma: [float] weight of inter-party correlation
        :return: [float] correlation score
        """

        inner_only = np.isclose(self.gamma, 0)

        assert self.gamma > 0, f"gamma {self.gamma} should be non-negative"

        n_parties = len(n_features_on_party)
        assert sum(n_features_on_party) == corr.shape[0] == corr.shape[1], \
            f"The number of features on each party should be the same as the size of the correlation matrix," \
            "but got {sum(n_features_on_party)} != {corr.shape[0]} != {corr.shape[1]}"
        corr_cut_points = np.cumsum(n_features_on_party)
        corr_cut_points = np.insert(corr_cut_points, 0, 0)
        inner_mcors = []
        inter_mcors = []
        for i in range(n_parties):
            for j in range(n_parties):
                if inner_only and i != j:
                    inter_mcors.append(0)
                    continue        # skip inter-party correlation if gamma is 0

                start_i = corr_cut_points[i]
                end_i = corr_cut_points[i + 1]
                start_j = corr_cut_points[j]
                end_j = corr_cut_points[j + 1]
                corr_ij = corr[start_i:end_i, start_j:end_j]
                mcor_score = mcor_func(corr_ij)
                if i == j:
                    inner_mcors.append(mcor_score)
                else:
                    inter_mcors.append(mcor_score)
        inner_mean_mcor = np.mean(inner_mcors)
        inter_mean_mcor = np.mean(inter_mcors)
        assert inner_mean_mcor >= 0 and inter_mean_mcor >= 0, f"inner_mean_mcor: {inner_mean_mcor}, " \
                                                              f"inter_mean_mcor: {inter_mean_mcor}"
        # print("inner_mean_mcor: {}, inter_mean_mcor: {}".format(inner_mean_mcor, inter_mean_mcor))
        return inner_mean_mcor - inter_mean_mcor * self.gamma

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
        return self.overall_corr_score(corr, n_features_on_party)

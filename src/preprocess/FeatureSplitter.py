import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from dataset.VFLDataset import VFLAlignedDataset, VFLRawDataset


class ImportanceSplitter:
    def __init__(self, num_parties, primary_party_id=0, weights=1):
        """
        Split a 2D dataset by feature importance under dirichlet distribution (assuming the features are independent).
        :param num_parties: [int] number of parties
        :param primary_party_id: [int] primary party (the party with labels) id, should be in range of [0, num_parties)
        :param weights: [int | list with size num_parties]
                        If weights is an int, the weight of each party is the same.
                        If weights is an array, the weight of each party is the corresponding element in the array.
                        The weights indicate the expected sum of feature importance of each party.
                        Meanwhile, larger weights mean less bias on the feature importance.
        """
        self.num_parties = num_parties
        self.primary_party_id = primary_party_id
        self.weights = weights
        if isinstance(self.weights, int):
            self.weights = [self.weights for _ in range(self.num_parties)]

        self.check_params()

    def check_params(self):
        """
        Check if the parameters are valid
        """
        assert 0 <= self.primary_party_id < self.num_parties, "primary_party_id should be in range of [0, num_parties)"
        assert len(self.weights) == self.num_parties, "The length of weights should equal to the number of parties"

    def split(self, X):
        """
        Split X by feature importance.
        :param X: [np.ndarray] 2D dataset
        :return: (X1, X2, ..., Xn) [np.ndarray, ...] where n is the number of parties
        """
        # Generate the probabilities of being assigned to each party for each feature under dirichlet distribution
        probs = np.random.dirichlet(self.weights, X.shape[1]).T

        # Assign each feature to a party
        party_to_feature = {}
        for feature_id in range(X.shape[1]):
            party_id = np.random.choice(self.num_parties, p=probs[feature_id])
            if party_id not in party_to_feature:
                party_to_feature[party_id] = [feature_id]
            else:
                party_to_feature[party_id].append(feature_id)

        # Split the dataset according to party_to_feature
        Xs = []
        for party_id in range(self.num_parties):
            if party_id in party_to_feature:
                Xs.append(X[:, party_to_feature[party_id]])
            else:
                Xs.append(np.empty((X.shape[0], 0)))

        return tuple(Xs)


class CorrelationSplitter:
    def __init__(self, num_parties, primary_party_id=0, weights=1):
        """
        Split a 2D dataset by feature correlation (assuming the features are equally important).
        :param num_parties: [int] number of parties
        :param primary_party_id: [int] primary party (the party with labels) id, should be in range of [0, num_parties)
        :param weights: [int | list with size num_parties]
                        If weights is an int, the weight of each party is the same.
                        If weights is an array, the weight of each party is the corresponding element in the array.
                        The weights indicate the expected sum of feature correlation of each party.
                        Meanwhile, larger weights mean less bias on the feature correlation.
        """
        self.num_parties = num_parties
        self.primary_party_id = primary_party_id
        self.weights = weights
        if isinstance(self.weights, int):
            self.weights = [self.weights for _ in range(self.num_parties)]

        self.check_params()

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
        assert (singular_values > 0).any()      # singular values should be positive by definition
        score = 1 / np.sqrt(d) * np.sqrt((1 / (d - 1)) * np.sum((singular_values - np.average(singular_values)) ** 2))
        return score

    @staticmethod
    def mean_mcor(corr, n_features_on_party, mcor_func: callable = mcor_singular):
        """
        Calculate the mean of mcor inside all parties. This is a three-party example of corr:
        [ v1 v1 .  .  .  .  ]
        [ v1 v1 .  .  .  .  ]
        [ .  .  v2 v2 .  .  ]
        [ .  .  v2 v2 .  .  ]
        [ .  .  .  .  v3 v3 ]
        [ .  .  .  .  v3 v3 ]
        v1, v2, v3 represent inner-party correlation matrices of each party.
        "." represents inter-party correlation matrices.
        :param corr: global correlation matrix of all parties
        :param n_features_on_party: List. Number of features on each party. The sum should equal to the size of corr.
        :param mcor_func: callable. Function to calculate mcor of a correlation matrix.
        :return: (float) mean of mcor.
        """
        n_parties = len(n_features_on_party)
        assert sum(n_features_on_party) == corr.shape[0]
        corr_cut_points = np.cumsum(n_features_on_party)
        corr_cut_points = np.insert(corr_cut_points, 0, 0)
        mcors = []
        for i in range(n_parties):
            start = corr_cut_points[i]
            end = corr_cut_points[i + 1]
            corr_party_i = corr[start:end, start:end]
            mcor_party_i = mcor_func(corr_party_i)
            mcors.append(mcor_party_i)
        return np.mean(mcors)

    def check_params(self):
        """
        Check if the parameters are valid
        """
        assert 0 <= self.primary_party_id < self.num_parties, "primary_party_id should be in range of [0, num_parties)"
        assert len(self.weights) == self.num_parties, "The length of weights should equal to the number of parties"

    def split(self, X):
        """
        Split X by feature correlation.
        :param X: [np.ndarray] 2D dataset
        :return: (X1, X2, ..., Xn) [np.ndarray, ...] where n is the number of parties
        """






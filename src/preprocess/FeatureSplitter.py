import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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

    def __init__(self, num_parties: int):
        """
        Split a 2D dataset by feature correlation (assuming the features are equally important).
        :param num_parties: [int] number of parties
        """
        self.num_parties = num_parties

        # split result of the last call of split()
        self.corr = None
        self.n_features_on_party = None
        self.min_mcor = None
        self.max_mcor = None
        self.best_mcor = None
        self.best_error = None
        self.best_feature_per_party = None
        self.best_permutation = None

    # Nested class for BRKGA solver: duplicate elimination of permutations
    class DuplicationElimination(ElementwiseDuplicateElimination):
        def is_equal(self, a, b):
            return a.get("hash") == b.get("hash")

    # Nested class for BRKGA solver: max-mcor problem definition
    class CorrMaxProblem(ElementwiseProblem):
        def __init__(self, corr, n_features_on_party):
            super().__init__(n_var=corr.shape[1], n_obj=1, n_constr=0, xl=-1, xu=1)
            self.corr = corr
            self.n_features_on_party = n_features_on_party

        def _evaluate(self, x, out, *args, **kwargs):
            order = np.argsort(x)
            corr_perm = self.corr[:, order]
            out['F'] = -CorrelationSplitter.mean_mcor(corr_perm, self.n_features_on_party)
            out['order'] = order
            out['hash'] = hash(tuple(order))

    # Nested class for BRKGA solver: min-mcor problem definition
    class CorrMinProblem(ElementwiseProblem):
        def __init__(self, corr, n_features_on_party):
            super().__init__(n_var=corr.shape[1], n_obj=1, n_constr=0, xl=-1, xu=1)
            self.corr = corr
            self.n_features_on_party = n_features_on_party

        def _evaluate(self, x, out, *args, **kwargs):
            order = np.argsort(x)
            corr_perm = self.corr[:, order]
            out['F'] = CorrelationSplitter.mean_mcor(corr_perm, self.n_features_on_party)
            out['order'] = order
            out['hash'] = hash(tuple(order))

    # Nested class for BRKGA solver: best-matched-mcor problem definition
    class CorrBestMatchProblem(ElementwiseProblem):
        def __init__(self, corr, n_features_on_party, beta, min_mcor, max_mcor):
            super().__init__(n_var=corr.shape[1], n_obj=1, n_constr=0, xl=-1, xu=1)
            assert min_mcor < max_mcor, f"min_mcor {min_mcor} should be smaller than max_mcor {max_mcor}"
            self.corr = corr
            self.n_features_on_party = n_features_on_party
            self.beta = beta
            self.max_mcor = max_mcor
            self.min_mcor = min_mcor

        def _evaluate(self, x, out, *args, **kwargs):
            order = np.argsort(x)
            corr_perm = self.corr[:, order]
            mcor = CorrelationSplitter.mean_mcor(corr_perm, self.n_features_on_party)
            target_mcor = self.beta * self.max_mcor + (1 - self.beta) * self.min_mcor

            out['F'] = np.abs(mcor - target_mcor)
            out['mcor'] = mcor
            out['order'] = order
            out['hash'] = hash(tuple(order))

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

    @staticmethod
    def split_num_features_equal(n_features, n_parties):
        """
        Split n_features into n_parties equally. The first party may have more features.
        :param n_features: (int) number of features
        :param n_parties: (int) number of parties
        :return: (list) number of features on each party
        """
        n_features_on_party = [n_features // n_parties] * n_parties
        n_features_on_party[0] += n_features % n_parties
        assert sum(n_features_on_party) == n_features
        return n_features_on_party

    def check_fit_data(self):
        """
        Check if the required members are calculated by fit()
        """
        assert self.corr is not None, "self.corr is None. Please call fit() first."
        assert self.n_features_on_party is not None, "self.n_features_on_party is None. Please call fit() first."
        assert self.min_mcor is not None, "self.min_mcor is None. Please call fit() first."
        assert self.max_mcor is not None, "self.max_mcor is None. Please call fit() first."

    def fit(self, X, n_elites=200, n_offsprings=700, n_mutants=100, n_gen=10, bias=0.8, seed=0, verbose=False):
        """
        Calculate the min and max mcor of the overall correlation score.
        Required parameters:
        :param X: [np.ndarray] 2D dataset

        Optional parameters: (BRKGA parameters)
        :param n_elites: (int) number of elites in BRKGA
        :param n_offsprings: (int) number of offsprings in BRKGA
        :param n_mutants: (int) number of mutants in BRKGA
        :param n_gen: (int) number of generations in BRKGA
        :param bias: (float) bias of BRKGA
        :param seed: (int) random seed
        :param verbose: (bool) whether to print the progress
        """
        self.corr = spearmanr(X).correlation
        self.n_features_on_party = self.split_num_features_equal(X.shape[1], self.num_parties)

        algorithm = BRKGA(
            n_elites=n_elites,
            n_offsprings=n_offsprings,
            n_mutants=n_mutants,
            bias=bias,
            eliminate_duplicates=self.DuplicationElimination(),
        )

        # calculate the min and max mcor
        res_min = minimize(
            self.CorrMinProblem(self.corr, self.n_features_on_party),
            algorithm,
            ('n_gen', n_gen),
            seed=seed,
            verbose=verbose,
        )
        self.min_mcor = res_min.F[0]
        res_max = minimize(
            self.CorrMaxProblem(self.corr, self.n_features_on_party),
            algorithm,
            ('n_gen', n_gen),
            seed=seed,
            verbose=verbose,
        )
        self.max_mcor = -res_max.F[0]

    def split(self, X, n_elites=200, n_offsprings=700, n_mutants=100, n_gen=10, bias=0.8, seed=0, verbose=False,
              beta=0.5):
        """
        Use BRKGA to find the best order of features that minimizes the difference between the mean of mcor and the
        target. split() assumes that the min and max mcor have been calculated by fit().
        Required parameters:
        :param X: [np.ndarray] 2D dataset

        Optional parameters: (BRKGA parameters)
        :param n_elites: (int) number of elites in BRKGA
        :param n_offsprings: (int) number of offsprings in BRKGA
        :param n_mutants: (int) number of mutants in BRKGA
        :param n_gen: (int) number of generations in BRKGA
        :param bias: (float) bias of BRKGA
        :param seed: (int) random seed
        :param verbose: (bool) whether to print the progress
        :param beta: [float] the tightness of inner-party correlation. Larger beta means more inner-party correlation
                             and less inter-party correlation.

        :return: (np.ndarray) indices of features in the order of importance
        """
        self.check_fit_data()

        algorithm = BRKGA(
            n_elites=n_elites,
            n_offsprings=n_offsprings,
            n_mutants=n_mutants,
            bias=bias,
            eliminate_duplicates=self.DuplicationElimination(),
        )

        # find the best permutation order that makes the mcor closest to the target mcor
        # target_mcor = beta * max_mcor + (1 - beta) * min_mcor
        res_beta = minimize(
            self.CorrBestMatchProblem(self.corr, self.n_features_on_party, beta, self.min_mcor, self.max_mcor),
            algorithm,
            ('n_gen', n_gen),
            seed=seed,
            verbose=verbose,
        )
        self.best_permutation = res_beta.opt.get('order')[0].astype(int)
        self.best_mcor = res_beta.opt.get('mcor')[0]
        self.best_error = res_beta.F[0]
        # print(f"Best permutation order: {permute_order}")
        # print(f"Beta {self.beta}, Best match mcor: {best_match_mcor}")

        # summarize the feature ids on each party
        party_cut_points = np.cumsum(self.n_features_on_party)
        party_cut_points = np.insert(party_cut_points, 0, 0)
        self.best_feature_per_party = []
        for i in range(len(party_cut_points) - 1):
            start = party_cut_points[i]
            end = party_cut_points[i + 1]
            self.best_feature_per_party.append(np.sort(self.best_permutation[start:end]))
        assert (np.sort(np.concatenate(self.best_feature_per_party)) == np.arange(X.shape[1])).all()
        # print(f"Feature ids on each party: {feature_ids_on_party}")

        # split X according to the permutation order
        X_split = []
        for feature_ids in self.best_feature_per_party:
            X_split.append(X[:, feature_ids])
        return tuple(X_split)

    def fit_split(self, X, n_elites=200, n_offsprings=700, n_mutants=100, n_gen=10, bias=0.8, seed=0, verbose=False):
        """
        Calculate the min and max mcor of the overall correlation score. Then use BRKGA to find the best order of
        features that minimizes the difference between the mean of mcor and the target mcor.
        Required parameters:
        :param X: [np.ndarray] 2D dataset

        Optional parameters: (BRKGA parameters)
        :param n_elites: (int) number of elites in BRKGA
        :param n_offsprings: (int) number of offsprings in BRKGA
        :param n_mutants: (int) number of mutants in BRKGA
        :param n_gen: (int) number of generations in BRKGA
        :param bias: (float) bias of BRKGA
        :param seed: (int) seed of BRKGA
        :param verbose: (bool) whether to print the progress of BRKGA optimization

        :return: (X1, X2, ..., Xn) [np.ndarray, ...] where n is the number of parties
        """
        self.fit(X, n_elites, n_offsprings, n_mutants, n_gen, bias, seed, verbose)
        return self.split(X, n_elites, n_offsprings, n_mutants, n_gen, bias, seed, verbose)












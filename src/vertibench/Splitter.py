from numbers import Real
import warnings
import abc

import numpy as np
import torch
import torch.linalg
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.problem import StarmapParallelization
from multiprocessing.pool import ThreadPool

from .Evaluator import CorrelationEvaluator


class Splitter(abc.ABC):
    def __init__(self, num_parties):
        self.num_parties = num_parties

    @abc.abstractmethod
    def split_indices(self, *args, **kwargs):
        """
        Split the indices of X
        :param X: [np.ndarray] 2D dataset
        :return: [list] list of indices of each party
        """
        pass

    def split(self, *Xs, indices=None, allow_empty_party=False, fill=None):
        assert len(Xs) > 0, "At least one dataset should be given"
        ans = []

        # calculate the indices for each party for all datasets
        if indices is None:
            allX = np.concatenate(Xs, axis=0)
            party_to_feature = self.split_indices(allX, allow_empty_party=allow_empty_party)
        else:
            party_to_feature = indices

        # split each dataset
        for X in Xs:
            X_split = []
            for i in range(self.num_parties):
                selected = party_to_feature[i]  # selected column_ids
                if fill is not None:
                    X_party_i = np.full_like(X, fill)
                    X_party_i[:, selected] = X[:, selected]
                else:
                    X_party_i = X[:, selected]
                X_split.append(X_party_i)
            ans.append(X_split)

        if len(ans) == 1:
            return ans[0]
        else:
            return ans


class ImportanceSplitter(Splitter):
    def __init__(self, num_parties, weights=1, seed=None):
        """
        Split a 2D dataset by feature importance under dirichlet distribution (assuming the features are independent).
        :param num_parties: [int] number of parties
        :param weights: [int | list with size num_parties]
                        If weights is an int, the weight of each party is the same.
                        If weights is an array, the weight of each party is the corresponding element in the array.
                        The weights indicate the expected sum of feature importance of each party.
                        Meanwhile, larger weights mean less bias on the feature importance.
        :param seed: [int] random seed
        """
        super().__init__(num_parties)
        self.weights = weights
        self.seed = seed
        np.random.seed(seed)
        if isinstance(self.weights, Real):
            self.weights = [self.weights for _ in range(self.num_parties)]

        self.check_params()

    def check_params(self):
        """
        Check if the parameters are valid
        """
        if len(self.weights) != self.num_parties:
            raise ValueError("The length of weights should equal to the number of parties")
        if not all([weight > 0 for weight in self.weights]):
            raise ValueError("The weights should be positive")
        if self.num_parties < 2:
            raise ValueError("The number of parties should be greater than 1")

    @staticmethod
    def dirichlet(alpha):
        """
        Generate a random sample from a Dirichlet distribution using beta distribution. This function can
        work with small alpha values.
        :param alpha: [float] the parameter of the symmetric Dirichlet distribution
        :return: [np.ndarray] the generated sample
        """
        xs = [np.random.beta(alpha[0], sum(alpha[1:]))]
        for i in range(1, len(alpha) - 1):
            phi = np.random.beta(alpha[i], sum(alpha[i + 1:]))
            xs.append((1 - sum(xs)) * phi)
        xs.append(1 - sum(xs))
        return np.array(xs)

    def split_indices(self, X, allow_empty_party=False):
        """
        Split the indices of X by feature importance.
        :param allow_empty_party: [bool] whether to allow parties with zero features
        :param X: [np.ndarray] 2D dataset
        :return: [list] list of indices of each party
        """
        # Generate the probabilities of being assigned to each party
        # All the features share the same ratio
        # probs = np.random.dirichlet(self.weights)     # has bug with small weights
        # if np.isnan(probs).any():
        #     probs = self.dirichlet(self.weights)      # use beta distribution instead, slower but more accurate
        probs = self.dirichlet(self.weights)

        if np.sum(probs) < 1 - 1e-6:
            # very small weights may cause all probs to be 0
            # in this case, assign all features to a random party
            assert np.isclose(sum(probs), 0)
            target_party = np.random.randint(self.num_parties)
            probs[target_party] = 1

        if allow_empty_party:
            # Assign each feature to a party
            party_to_feature = [[] for _ in range(self.num_parties)]
            party_ids = np.random.choice(self.num_parties, size=X.shape[1], p=probs)
            for feature_id in range(X.shape[1]):
                party_id = party_ids[feature_id]
                party_to_feature[party_id].append(feature_id)
        else:
            # uniform-randomly select one feature for each party
            preassigned_feature_ids = np.random.choice(X.shape[1], size=self.num_parties, replace=False)
            party_to_feature = [[feature_id] for feature_id in preassigned_feature_ids]

            # Assign the remaining features to the parties
            preassigned_feature_id_set = set(preassigned_feature_ids)
            if not np.isclose(sum(probs), 1):
                raise ValueError(f"The sum of probs should be equal to 1, got {probs} from weights {self.weights}")
            party_ids = np.random.choice(self.num_parties, size=X.shape[1], p=probs)
            for feature_id in range(X.shape[1]):
                if feature_id not in preassigned_feature_id_set:
                    party_id = party_ids[feature_id]
                    party_to_feature[party_id].append(feature_id)

            # no empty party
            assert np.count_nonzero([len(party) for party in party_to_feature]) == self.num_parties, \
                "There should be no empty party"
            assert np.sum([len(party) for party in party_to_feature]) == X.shape[1], \
                "The number of features should be the same as the number of columns of X"

        return party_to_feature


class CorrelationSplitter(Splitter):

    def __init__(self, num_parties: int, evaluator: CorrelationEvaluator = None, seed=None, gpu_id=None, n_jobs=1):
        """
        Split a 2D dataset by feature correlation (assuming the features are equally important).
        :param num_parties: [int] number of parties
        :param evaluator: [CorrelationEvaluator] the evaluator to evaluate the correlation
        :param seed: [int] random seed
        :param gpu_id: [int] GPU id
        """
        super().__init__(num_parties)
        self.evaluator = evaluator
        if evaluator is None:
            self.evaluator = CorrelationEvaluator(gpu_id=gpu_id)
        self.seed = seed
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            assert self.evaluator.gpu_id == self.gpu_id, \
                "The gpu_id of the evaluator should be the same as the gpu_id of the splitter"
            self.device = torch.device("cuda:{}".format(self.gpu_id))
        else:
            self.device = torch.device("cpu")
            warnings.warn("The device is set to CPU. This may cause the performance to be slow.")

        self.pool = ThreadPool(n_jobs)
        self.runner = StarmapParallelization(self.pool.starmap)

        # split result of the last call of fit()
        # self.corr = None      # use evaluator.corr instead
        # self.n_features_on_party = None   # use evaluator.n_features_on_party instead
        self.min_icor = None
        self.max_icor = None

        # best split result of all calls of split()
        self.best_icor = None
        self.best_error = None
        self.best_feature_per_party = None
        self.best_permutation = None

    @staticmethod
    def sort_order_by_party(order, n_features_on_party):
        order_cut_points = np.cumsum(n_features_on_party)
        order_cut_points = np.insert(order_cut_points, 0, 0)
        sorted_order_by_party = []
        for i in range(1, len(order_cut_points)):
            sorted_order_by_party.append(tuple(sorted(order[order_cut_points[i - 1]:order_cut_points[i]])))
        return sorted_order_by_party

    # Nested class for BRKGA solver: duplicate elimination of permutations
    class DuplicationElimination(ElementwiseDuplicateElimination):
        def is_equal(self, perm_a, perm_b):
            return perm_a.get("hash") == perm_b.get("hash")

    # Nested class for BRKGA solver: best-matched-icor problem definition
    class CorrBestMatchProblem(ElementwiseProblem):
        def __init__(self, corr, n_features_on_party, beta, min_icor, max_icor, evaluator: CorrelationEvaluator = None,
                     runner=None):
            super().__init__(n_var=corr.shape[1], n_obj=1, n_constr=0, xl=-1, xu=1, elementwise_runner=runner)
            assert min_icor < max_icor, f"min_icor {min_icor} should be smaller than max_icor {max_icor}"
            self.corr = corr
            self.n_features_on_party = n_features_on_party
            self.beta = beta
            self.max_icor = max_icor
            self.min_icor = min_icor
            self.evaluator = evaluator

        # @cachetools.cached(cache=cachetools.TTLCache(maxsize=10000, ttl=60),
        #                    key=lambda self, corr, order: hash(tuple(
        #                        CorrelationSplitter.sort_order_by_party(order, self.n_features_on_party))))
        def corr_score(self, corr, order):
            corr_perm = corr[order, :][:, order]
            return self.evaluator.overall_corr_score(corr_perm, self.n_features_on_party)

        def _evaluate(self, x, out, *args, **kwargs):
            order = np.argsort(x)
            icor = self.corr_score(self.corr, order)
            target_icor = self.beta * self.max_icor + (1 - self.beta) * self.min_icor

            out['F'] = np.abs(icor - target_icor)
            out['icor'] = icor
            out['order'] = order
            # sorted_order_by_party = CorrelationSplitter.sort_order_by_party(order, self.n_features_on_party)
            out['hash'] = hash(tuple(order))

    def check_fit_data(self):
        """
        Check if the required members are calculated by fit()
        """
        assert self.evaluator.corr is not None, "self.evaluator.corr is None. Please call fit() first."
        assert self.evaluator.n_features_on_party is not None, "self.evaluator.n_features_on_party is None. Please call fit() first."
        assert self.min_icor is not None, "self.min_icor is None. Please call fit() first."
        assert self.max_icor is not None, "self.max_icor is None. Please call fit() first."

    def fit(self, X, **kwargs):
        """
        Calculate the min and max icor of the overall correlation score.
        Required parameters:
        :param X: [np.ndarray] 2D dataset

        Optional parameters: (BRKGA parameters)
        :param n_elites: (int) number of elites in BRKGA
        :param n_offsprings: (int) number of offsprings in BRKGA
        :param n_mutants: (int) number of mutants in BRKGA
        :param n_gen: (int) number of generations in BRKGA
        :param bias: (float) bias of BRKGA
        :param verbose: (bool) whether to print the progress
        :param kwargs: other unused args
        """
        Xs = np.array_split(X, self.num_parties, axis=1)
        self.evaluator.fit(Xs)
        self.evaluator.fit_min_max(**kwargs)
        self.min_icor = self.evaluator.min_icor
        self.max_icor = self.evaluator.max_icor

    def split_indices(self, X, n_elites=20, n_offsprings=70, n_mutants=10, n_gen=100, bias=0.7, verbose=False,
              beta=0.5, term_tol=1e-4, term_period=10):
        """
        Use BRKGA to find the best order of features that minimizes the difference between the mean of icor and the
        target. split() assumes that the min and max icor have been calculated by fit().
        Required parameters:
        :param X: [np.ndarray] 2D dataset

        Optional parameters: (BRKGA parameters)
        :param n_elites: (int) number of elites in BRKGA
        :param n_offsprings: (int) number of offsprings in BRKGA
        :param n_mutants: (int) number of mutants in BRKGA
        :param n_gen: (int) number of generations in BRKGA
        :param bias: (float) bias of BRKGA
        :param verbose: (bool) whether to print the progress
        :param beta: [float] the tightness of inner-party correlation. Larger beta means more inner-party correlation
                             and less inter-party correlation.
        :param term_tol: (float) If out['F'] < term_tol after term_period generations, the algorithm terminates.
        :param term_period: (int) Check the termination condition every term_period generations

        :return: (np.ndarray) indices of features in the order of importance
        """
        self.check_fit_data()
        if not (0 <= beta <= 1):
            raise ValueError(f"beta should be in [0, 1], got {beta}")

        # termination by number of generations or the error is less than 1e-6
        termination = DefaultSingleObjectiveTermination(ftol=term_tol, n_max_gen=n_gen, period=term_period)
        algorithm = BRKGA(
            n_elites=n_elites,
            n_offsprings=n_offsprings,
            n_mutants=n_mutants,
            bias=bias,
            eliminate_duplicates=self.DuplicationElimination(),
        )

        # find the best permutation order that makes the icor closest to the target icor
        # target_icor = beta * max_icor + (1 - beta) * min_icor
        res_beta = minimize(
            self.CorrBestMatchProblem(self.evaluator.corr, self.evaluator.n_features_on_party, beta, self.min_icor,
                                      self.max_icor,
                                      evaluator=self.evaluator, runner=self.runner),
            algorithm,
            termination,
            seed=self.seed,
            verbose=verbose,
        )
        self.best_permutation = res_beta.opt.get('order')[0].astype(int)
        self.best_icor = res_beta.opt.get('icor')[0]
        self.best_error = res_beta.F[0]
        # print(f"Best permutation order: {permute_order}")
        # print(f"Beta {self.beta}, Best match icor: {best_match_icor}")

        # summarize the feature ids on each party
        party_cut_points = np.cumsum(self.evaluator.n_features_on_party)
        party_cut_points = np.insert(party_cut_points, 0, 0)
        self.best_feature_per_party = []
        for i in range(len(party_cut_points) - 1):
            start = party_cut_points[i]
            end = party_cut_points[i + 1]
            self.best_feature_per_party.append(np.sort(self.best_permutation[start:end]))
        assert (np.sort(np.concatenate(self.best_feature_per_party)) == np.arange(X.shape[1])).all()
        return self.best_feature_per_party

    def fit_split(self, X, **kwargs):
        self.fit(X, **kwargs)
        return self.split(X, **kwargs)

    def visualize(self, *args, **kwargs):
        return self.evaluator.visualize(*args, **kwargs)


class SimpleSplitter(Splitter):
    def __init__(self, num_parties):
        """
        Split a 2D dataset by equally dividing the features.
        :param num_parties: [int] number of parties
        """
        super().__init__(num_parties)

    def split_indices(self, n_features):
        """
        Split the indices of features into num_parties parts
        :param n_features: [int] number of features
        :return: [list of np.ndarray] indices of features for each party
        """
        indices = np.arange(n_features)
        return np.array_split(indices, self.num_parties)




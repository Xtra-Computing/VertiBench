import warnings
from typing import Iterable
import time
import os

import numpy as np
import pandas as pd
import torch
import torch.linalg
from scipy.stats import spearmanr, hmean, gmean
from sklearn.utils.extmath import randomized_svd
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.datasets import load_svmlight_file
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.problem import StarmapParallelization
from multiprocessing.pool import ThreadPool


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
        assert n_parties >= 1, "ImportanceEvaluator only works for at least one party"
        assert all([X.shape[0] == Xs[0].shape[0] for X in Xs]), \
            "The number of samples should be the same for all parties"
        return n_features_on_party

    def evaluate_feature(self, X, model: callable, **kwargs):
        """
        Evaluate the importance of features in a dataset
        :param X: feature matrix
        :param model: [callable] model to be evaluated
        :param kwargs: [dict] other parameters for shap.explainers.Permutation
        :return: [np.ndarray] sum of importance on each party
        """
        # max_eval is 500 by default in sha.explainers.Permutation explainer
        # if the number of features is large, we need to increase max_evals
        if 'max_evals' not in kwargs and 2 * X.shape[1] + 1 > 500:
            kwargs['max_evals'] = 2 * X.shape[1] + 1

        # calculate Shapley values for each feature
        explainer = shap.explainers.Permutation(model, X, **kwargs)
        sample_size = int(self.sample_rate * X.shape[0])
        if sample_size <= 1:
            raise ValueError(f"Sample size 0 cannot produce meaningful Shapley values."
                             f" Please increase the sample rate.")
        X_sample = shap.sample(X, sample_size, random_state=self.seed)
        shap_values = explainer(X_sample).values

        importance_by_feature = np.sum(np.abs(shap_values), axis=0)
        assert importance_by_feature.shape[0] == X.shape[1], "The number of features should be the same"
        return importance_by_feature

    def evaluate(self, Xs, model: callable, **kwargs):
        """
        Evaluate the importance of features in VFL datasets
        :param Xs: [list] list of feature matrices
        :param model: [callable] model to be evaluated
        :param kwargs: [dict] other parameters for shap.explainers.Permutation
        :return: [np.ndarray] sum of importance on each party
        """
        n_features_on_party = self.check_data(Xs)
        X = np.concatenate(Xs, axis=1)

        # max_eval is 500 by default in sha.explainers.Permutation explainer
        # if the number of features is large, we need to increase max_evals
        if 'max_evals' not in kwargs and 2 * X.shape[1] + 1 > 500:
            kwargs['max_evals'] = 2 * X.shape[1] + 1

        # calculate Shapley values for each feature
        explainer = shap.explainers.Permutation(model, X, **kwargs)
        sample_size = int(self.sample_rate * X.shape[0])
        if sample_size <= 1:
            raise ValueError(f"Sample size 0 cannot produce meaningful Shapley values."
                             f" Please increase the sample rate.")
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

    def evaluate_alpha(self, Xs=None, model: callable=None, scores=None, **kwargs):
        if scores is None:
            if Xs is None or model is None:
                raise ValueError(f"Since {scores=}, either scores or Xs and model should be provided to get scores.")
            scores = self.evaluate(Xs, model, **kwargs)
        num_parties = len(scores)
        scores = scores / np.sum(scores)  # normalize the scores
        score_var = np.var(scores)
        est_alpha = (num_parties - 1 - num_parties ** 2 * score_var) / (num_parties ** 3 * score_var)
        return est_alpha


class CorrelationEvaluator:
    """
    Correlation evaluator for VFL datasets
    """

    def __init__(self, corr_func='spearmanr', gpu_id=None, svd_algo='auto', n_jobs=1, **kwargs):
        """
        :param corr_func: [str] function to calculate the correlation between two features
        :param gamma: [float] weight of the inner-party correlation score
        :param gpu_id: [int] GPU id to use. If None, use CPU
        :param svd_algo: [str] algorithm to use for SVD. Should be one of {'auto', 'approx', 'exact'}
        :param kwargs: [dict] other parameters for pcor_singular
        """

        self.corr = None
        self.n_features_on_party = None
        self.gpu_id = gpu_id
        self.svd_algo = svd_algo

        assert self.svd_algo in ['auto', 'approx', 'exact'], "svd_algo should be auto, approx or exact"
        if self.gpu_id is not None:
            self.device = torch.device(f"cuda:{self.gpu_id}")

            if corr_func == "spearmanr":
                self.corr_func = self.spearmanr_pandas  # use CPU for now, a bug in GPU version
            elif corr_func == "spearmanr_pandas":
                self.corr_func = self.spearmanr_pandas
            elif corr_func == "pearson":
                self.corr_func = self.pearson_gpu
            else:
                raise NotImplementedError("corr_func should be in spearmanr, spearmanr_pandas or pearson")
        else:
            self.device = torch.device("cpu")
            if corr_func == "spearmanr":
                self.corr_func = self.spearmanr_pandas
            elif corr_func == "spearmanr_pandas":
                self.corr_func = self.spearmanr_pandas
            elif corr_func == "pearson":
                self.corr_func = self.pearson
            else:
                raise NotImplementedError("corr_func should be in spearmanr, spearmanr_pandas or pearson")

        print(f"CorrelationEvaluator uses {self.device}")
        self.pcor_kwargs = kwargs

        # parallel evaluator
        self.pool = ThreadPool(n_jobs)
        self.runner = StarmapParallelization(self.pool.starmap)

        self.min_icor = None
        self.max_icor = None

    def spearmanr_pandas(self, X):
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = pd.DataFrame(X).corr(method='spearman').values
            # corr = spearmanr(X).correlation <=== BUG: this cannot calculate the correlation of constant features
        if np.isnan(corr).all():  # in case all features are constant
            corr = np.zeros((X.shape[1], X.shape[1]))
        else:
            corr = np.nan_to_num(corr, nan=0)
        if self.gpu_id is not None:
            corr = torch.from_numpy(corr).float().to(self.device)
        return corr

    def spearmanr(self, X):
        """
        Calculate the correlation matrix of X
        :param X: [np.ndarray] 2D data matrix. Size: n_samples * n_features
        :return: [np.ndarray] correlation matrix. Size: n_features * n_features
        """
        # When there are constant features in X. The correlation may be NaN, raise a warning "numpy ignore divide by
        # zero warning". We ignore this warning and replace NaN in corr with 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = spearmanr(X).correlation  # <=== BUG: this cannot calculate the correlation of constant features
        if np.isnan(corr).all():  # in case all features are constant
            raise ValueError("All features are constant, scipy have a BUG that cannot calculate the correlation.")
        else:
            corr = np.nan_to_num(corr, nan=0)
        if self.gpu_id is not None:
            corr = torch.from_numpy(corr).float().to(self.device)
        return corr

    def pearson(self, X):
        """
        Calculate the correlation matrix of X
        :param X: [np.ndarray] 2D data matrix. Size: n_samples * n_features
        :return: [np.ndarray] correlation matrix. Size: n_features * n_features
        """
        # When there are constant features in X. The correlation may be NaN, raise a warning "numpy ignore divide by
        # zero warning". We ignore this warning and replace NaN in corr with 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(X, rowvar=False)
        if np.isnan(corr).all():  # in case all features are constant
            corr = np.zeros((X.shape[1], X.shape[1]))
        else:
            corr = np.nan_to_num(corr, nan=0)
        if self.gpu_id is not None:
            corr = torch.from_numpy(corr).float().to(self.device)
        return corr

    def pearson_gpu(self, X):
        """
        Calculate the correlation matrix of X
        :param X: [np.ndarray] 2D data matrix. Size: n_samples * n_features
        :return: [np.ndarray] correlation matrix. Size: n_features * n_features
        """
        # When there are constant features in X. The correlation may be NaN, raise a warning "numpy ignore divide by
        # zero warning". We ignore this warning and replace NaN in corr with 0.
        X = torch.from_numpy(X).float().to(self.device)
        corr = torch.corrcoef(X.T)  # columns are variables
        corr = torch.nan_to_num(corr, nan=0)
        return corr

    @staticmethod
    def pcor_singular_naive(corr):
        """
        Calculate the overall correlation score of a correlation matrix using the variance of singular values.
        This is a naive implementation of the method that calculates all singular values. This is usually 2 to
        3 times slower than pcor_singular_exact().
        This is an example of corr
        [ v1 v1 ]
        [ v1 v1 ]
        :param corr:
        :return:
        """
        # start_time = time.time()
        assert np.isnan(corr).any() == False, "NaN values should be replaced with 0"

        # d = corr.shape[0]
        singular_values = np.linalg.svd(corr)[1]
        score = np.std(singular_values, ddof=1) / np.sqrt(min(corr.shape))

        # end_time = time.time()
        # print(f"Time for calculating the correlation score: {end_time - start_time}")
        return float(score.item())

    @staticmethod
    def pcor_singular_exact(corr):
        """
        [Exact] Calculate the overall correlation score of a correlation matrix using the variance of singular values.
        Using the definition of std: std = sqrt(E[X^2] - E[X]^2). E[X^2] can be calculated by trace(X^TX), which is
        square of Frobenius norm. E[X] is known as nuclear norm.
        This is an example of corr
        [ v1 v1 ]
        [ v1 v1 ]
        :param corr:
        :return:
        """
        assert np.isnan(corr).any() == False, "NaN values should be replaced with 0"

        d = min(corr.shape[0], corr.shape[1])
        vals = np.linalg.svd(corr, compute_uv=False)
        if len(vals) <= 1:
            score = np.array([0])  # less than one feature, we define std as 0
        else:
            score = np.std(vals, ddof=1) / np.sqrt(d)

        return float(score.item())

    def pcor_singular_exact_gpu(self, corr: torch.Tensor):
        """
        [Exact] Calculate the overall correlation score of a correlation matrix using the variance of singular values.
        Using the definition of std: std = sqrt(E[X^2] - E[X]^2). E[X^2] can be calculated by trace(X^TX), which is
        square of Frobenius norm. E[X] is known as nuclear norm.
        This is an example of corr
        [ v1 v1 ]
        [ v1 v1 ]
        :param corr:
        :return:
        """
        singular_values = torch.linalg.svdvals(corr)
        d = min(corr.shape)
        if len(singular_values) <= 1:
            score = torch.tensor([0])  # less than one feature, we define std as 0
        else:
            score = torch.std(singular_values) / np.sqrt(d)

        return float(score.item())

    @staticmethod
    def pcor_singular_approx(corr, n_components=400, n_oversamples=10, n_iter=4, random_state=0):
        """
        [Approximate] Calculate the overall correlation score of a correlation matrix using the variance of singular
        values. This function uses randomized SVD to approximate the singular values. It is much faster than the exact
        in high-dimensional data at the cost of some accuracy.
        This is an example of corr
        [ v1 v1 ]
        [ v1 v1 ]
        :param corr:
        :param n_components: [int] number of singular values to return
        :param n_oversamples: [int] usually set to 2*k-n_components when k is the effective rank of the matrix
        :param n_iter: [int] number of power iterations
        :param random_state: [int] random seed
        :param gpu_id: [int] GPU id. If None, use CPU
        :return:
        """
        # start_time = time.time()
        if n_components > min(corr.shape):
            n_components = min(corr.shape)

        assert np.isnan(corr).any() == False, "NaN values should be replaced with 0"
        _, singular_values, _ = randomized_svd(corr, n_components=n_components, n_oversamples=n_oversamples,
                                               n_iter=n_iter,
                                               random_state=random_state)
        singular_shape = min(corr.shape)
        s_append_zero = np.concatenate((singular_values, np.zeros(singular_shape - singular_values.shape[0])))

        if len(s_append_zero) <= 1:
            score = np.array([0])
        else:
            score = np.std(s_append_zero, ddof=1) / np.sqrt(singular_shape)

        # end_time = time.time()
        # print(f"Time for calculating the correlation score: {end_time - start_time}")
        return float(score.item())

    @staticmethod
    def pcor_singular_approx_gpu(corr: torch.Tensor, n_components=400, n_iter=4):
        """
        [Approximate] Calculate the overall correlation score of a correlation matrix using the variance of singular
        values. This function uses randomized SVD to approximate the singular values. It invokes torch.svd_lowrank() to
        calculate the singular values on GPU.
        This is an example of corr
        [ v1 v1 ]
        [ v1 v1 ]
        :param corr: [torch.Tensor] 2D data matrix. Size: n_features * n_features. This matrix should be already on GPU.
        :param n_components: [int] number of singular values to return
        :param n_iter: [int] number of power iterations
        :return: [float] correlation score
        """

        if n_components > min(corr.shape):
            n_components = min(corr.shape)

        _, singular_values, _ = torch.svd_lowrank(corr, q=n_components, niter=n_iter)
        singular_shape = min(corr.shape)
        s_append_zero = torch.concatenate((singular_values,
                                           torch.zeros(singular_shape - singular_values.shape[0]).to(corr.device)))

        if len(s_append_zero) <= 1:
            score = torch.tensor([0])
        else:
            score = torch.std(s_append_zero) / np.sqrt(singular_shape)

        return float(score.item())

    def pcor_singular(self, corr, **kwargs):
        """
        Calculat the std of the singular values of corr matrix.
        :param corr: [np.ndarray] correlation matrix
        :param kwargs:

        :param self.svd_algo: [str] algorithm to calculate the overall correlation score of a correlation matrix
                        - 'auto': automatically choose the algorithm based on the size of the correlation matrix
                                  if the size is smaller than 200, use 'exact', otherwise use 'approx'
                        - 'exact': calculate the exact singular values
                        - 'approx': calculate the approximate singular values
        :return:
        """
        # merge kwargs with self.kwargs, and overwrite self.kwargs if there is a conflict
        kwargs = self.pcor_kwargs | kwargs
        kwargs.pop('method', None)  # remove the unused self.kwargs['method']

        if self.svd_algo == 'auto':
            if min(corr.shape) < 100:
                if self.gpu_id is not None:
                    return self.pcor_singular_exact_gpu(corr)
                else:
                    return CorrelationEvaluator.pcor_singular_exact(corr)
            else:
                if self.gpu_id is not None:
                    return self.pcor_singular_approx_gpu(corr, **kwargs)
                else:
                    return CorrelationEvaluator.pcor_singular_approx(corr, **kwargs)
        elif self.svd_algo == 'exact':
            if self.gpu_id is not None:
                return self.pcor_singular_exact_gpu(corr)
            else:
                return CorrelationEvaluator.pcor_singular_exact(corr)
        elif self.svd_algo == 'approx':
            if self.gpu_id is not None:
                return self.pcor_singular_approx_gpu(corr, **kwargs)
            else:
                return CorrelationEvaluator.pcor_singular_approx(corr, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.svd_algo}")

    def _get_inner_and_inter_corr(self, corr, n_features_on_party, symmetric=True):
        """
        Calculate the inner-party and inter-party correlation matrices.
        :param corr: [np.ndarray] correlation matrix
        :param n_features_on_party: [list] number of features on each party
        :return: [2D np.ndarray] correlation scores (size: number of parties x number of parties)
        """
        n_parties = len(n_features_on_party)
        assert sum(n_features_on_party) == corr.shape[0] == corr.shape[1], \
            f"The number of features on each party should be the same as the size of the correlation matrix," \
            "but got {sum(n_features_on_party)} != {corr.shape[0]} != {corr.shape[1]}"
        corr_cut_points = np.cumsum(n_features_on_party)
        corr_cut_points = np.insert(corr_cut_points, 0, 0)

        pcors = np.zeros((n_parties, n_parties))
        for i in range(n_parties):
            for j in range(n_parties):
                start_i = corr_cut_points[i].item()
                end_i = corr_cut_points[i + 1].item()
                start_j = corr_cut_points[j].item()
                end_j = corr_cut_points[j + 1].item()

                if symmetric and j > i:
                    continue

                pcors[i][j] = self.pcor_singular(corr[start_i:end_i, start_j:end_j])

        if symmetric:
            for i in range(n_parties):
                for j in range(i + 1, n_parties):
                    pcors[i][j] = pcors[j][i]
        return pcors

    def _get_inter_corr(self, corr, n_features_on_party, symmetric=True):
        """
        Calculate the inter-party correlation matrices.
        :param corr: [np.ndarray|torch.Tensor] correlation matrix
        :param n_features_on_party: [list] number of features on each party
        :param symmetric: [bool] whether the inter-party correlation matrix is symmetric. If True, only the upper
                            triangular part of the matrix is calculated.
        :return: [2D np.ndarray] inter-party correlation scores (size: number of parties x (number of parties - 1))
        """
        n_parties = len(n_features_on_party)
        assert sum(n_features_on_party) == corr.shape[0] == corr.shape[1], \
            f"The number of features on each party should be the same as the size of the correlation matrix," \
            "but got {sum(n_features_on_party)} != {corr.shape[0]} != {corr.shape[1]}"
        corr_cut_points = np.cumsum(n_features_on_party)
        corr_cut_points = np.insert(corr_cut_points, 0, 0)

        inter_pcors = []
        for i in range(n_parties):
            for j in range(n_parties):
                start_i = corr_cut_points[i].item()
                end_i = corr_cut_points[i + 1].item()
                start_j = corr_cut_points[j].item()
                end_j = corr_cut_points[j + 1].item()
                if symmetric:
                    save = i < j
                else:
                    save = i != j
                if save:
                    inter_pcors.append(self.pcor_singular(corr[start_i:end_i, start_j:end_j]))

        return np.array(inter_pcors)

    def overall_corr_score(self, corr, n_features_on_party):
        """
        Calculate the correlation score of a correlation matrix. This is a three-party example of corr:
        [ v1 v1 .  .  .  .  ]
        [ v1 v1 .  .  .  .  ]
        [ .  .  v2 v2 .  .  ]
        [ .  .  v2 v2 .  .  ]
        [ .  .  .  .  v3 v3 ]
        [ .  .  .  .  v3 v3 ]
        v1, v2, v3 represent inner-party correlation matrices of each party. This function evaluates the
        arithmetic mean of differences between the inter-party correlation scores and the inner-party correlation
        scores for each party.
        :param corr: [np.ndarray|torch.Tensor] correlation matrix
        :param n_features_on_party: [list] number of features on each party
        :return: [float] correlation score
        """
        # inter_pcors = self._get_inter_corr(corr, n_features_on_party, pcor_func)
        pcors = self._get_inner_and_inter_corr(corr, n_features_on_party)
        N = pcors.shape[0]

        # calculate the absolute difference between the inner-party correlation scores and the inter-party correlation
        # for each party
        diffs = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    diffs[i] += (pcors[j][i] - pcors[i][i])  # in range [-1, 1]
            diffs[i] /= (N - 1)
        return np.mean(diffs)

    @staticmethod
    def check_data(Xs):
        """
        Check if the data is valid for correlation evaluation.
        :param Xs: [List|Tuple] list of feature matrices of each party
        :return: [bool] True if the data is valid
        """
        n_features_on_party = [X.shape[1] for X in Xs]
        n_parties = len(n_features_on_party)
        assert n_parties >= 1, "The number of parties should be at least 1"
        assert all([X.shape[0] == Xs[0].shape[0] for X in Xs]), \
            "The number of samples should be the same for all parties"
        return n_features_on_party

    def fit_evaluate(self, Xs):
        """
        Evaluate the correlation score of a vertical federated learning dataset.
        :param Xs: [List|Tuple] list of feature matrices of each party
        :return: [float] correlation score
        """
        self.fit(Xs)
        return self.evaluate(Xs)

    def fit(self, Xs):
        """
        Fit the correlation matrix of a vertical federated learning dataset.
        :param Xs: [List|Tuple] list of feature matrices of each party
        """
        self.n_features_on_party = self.check_data(Xs)
        if Xs[0].shape[0] <= 1:
            raise ValueError("The number of samples should be greater than 1")
        Xs = list(Xs)
        if torch.is_tensor(Xs[0]):
            self.corr = self.corr_func(torch.cat(Xs, dim=1))
        elif isinstance(Xs[0], np.ndarray):
            self.corr = self.corr_func(np.concatenate(Xs, axis=1))
        else:
            raise ValueError(f"Xs should be either np.ndarray or torch.Tensor, but got {type(Xs[0])}")

        if torch.is_tensor(self.corr):
            self.corr = torch.nan_to_num(self.corr, nan=0)
        else:
            self.corr = np.nan_to_num(self.corr, nan=0)

    class DuplicationElimination(ElementwiseDuplicateElimination):
        def is_equal(self, perm_a, perm_b):
            return perm_a.get("hash") == perm_b.get("hash")

    # Nested class for BRKGA solver: max-pcor problem definition
    class CorrMaxProblem(ElementwiseProblem):
        def __init__(self, corr, n_features_on_party, evaluator=None, runner=None):
            super().__init__(n_var=corr.shape[1], n_obj=1, n_constr=0, xl=-1, xu=1, elementwise_runner=runner)
            self.corr = corr
            self.n_features_on_party = n_features_on_party
            self.evaluator = evaluator

        # @cachetools.cached(cache=cachetools.TTLCache(maxsize=1000, ttl=60),
        #                    key=lambda self, corr, order: hash(tuple(
        #                        CorrelationSplitter.sort_order_by_party(order, self.n_features_on_party))))
        def corr_score(self, corr, order):
            corr_perm = corr[order, :][:, order]
            return self.evaluator.overall_corr_score(corr_perm, self.n_features_on_party)

        def _evaluate(self, x, out, *args, **kwargs):
            order = np.argsort(x)
            out['F'] = -self.corr_score(self.corr, order)
            out['order'] = order
            # sorted_order_by_party = CorrelationSplitter.sort_order_by_party(order, self.n_features_on_party)
            out['hash'] = hash(tuple(order))

    # Nested class for BRKGA solver: min-pcor problem definition
    class CorrMinProblem(ElementwiseProblem):
        def __init__(self, corr, n_features_on_party, evaluator=None, runner=None):
            super().__init__(n_var=corr.shape[1], n_obj=1, n_constr=0, xl=-1, xu=1, elementwise_runner=runner)
            self.corr = corr
            self.n_features_on_party = n_features_on_party
            self.evaluator = evaluator

        def corr_score(self, corr, order):
            corr_perm = corr[order, :][:, order]
            return self.evaluator.overall_corr_score(corr_perm, self.n_features_on_party)

        def _evaluate(self, x, out, *args, **kwargs):
            order = np.argsort(x)
            out['F'] = self.corr_score(self.corr, order)
            out['order'] = order
            # sorted_order_by_party = CorrelationSplitter.sort_order_by_party(order, self.n_features_on_party)
            out['hash'] = hash(tuple(order))

    def fit_min_max(self, n_elites=200, n_offsprings=700, n_mutants=100, n_gen=100, bias=0.7, verbose=False,
                    seed=None, **kwargs):
        """
        Find the min-max value of Icor for all the possible feature split
        :param X: [np.ndarray] global data set
        :return: [tuple(float, float)] min-max value of Icor
        """
        # self.evaluator.corr = self.evaluator.corr_func(X)
        # self.evaluator.n_features_on_party = self.split_num_features_equal(X.shape[1], self.num_parties)
        if self.corr is None:
            raise ValueError("Data matrix X is not provided, and self.corr is not calculated yet."
                             "Please either provide X or call fit() first.")

        algorithm = BRKGA(
            n_elites=n_elites,
            n_offsprings=n_offsprings,
            n_mutants=n_mutants,
            bias=bias,
            eliminate_duplicates=self.DuplicationElimination(),
        )

        # calculate the min and max pcor
        if verbose:
            print("Calculating the min pcor of the overall correlation score...")
        res_min = minimize(
            self.CorrMinProblem(self.corr, self.n_features_on_party, evaluator=self,
                                runner=self.runner),
            algorithm,
            ('n_gen', n_gen),
            seed=seed,
            verbose=verbose,
        )
        self.min_icor = res_min.F[0]
        print(f"min_icor: {self.min_icor}")
        if verbose:
            print("Calculating the max icor of the overall correlation score...")
        res_max = minimize(
            self.CorrMaxProblem(self.corr, self.n_features_on_party, evaluator=self,
                                runner=self.runner),
            algorithm,
            ('n_gen', n_gen),
            seed=seed,
            verbose=verbose,
        )
        self.max_icor = -res_max.F[0]
        print(f"max_icor: {self.max_icor}")

    def evaluate(self, Xs=None):
        """
        Evaluate the correlation score of a vertical federated learning dataset with self.corr.
        :return: [float] correlation score
        """
        if self.corr is None:
            raise ValueError("Please call fit() or fit_evaluate() first to calculate the correlation matrix.")
        if Xs is not None:
            # If Xs is provided, update self.n_features_on_party
            self.n_features_on_party = self.check_data(Xs)
        return self.overall_corr_score(self.corr, self.n_features_on_party)

    def evaluate_beta(self, Xs=None, clip=True) -> float:
        if self.corr is None:
            raise ValueError("Please call fit() or fit_evaluate() first to calculate the correlation matrix.")

        if self.min_icor is None or self.max_icor is None:
            # evaluate the min and max icor
            self.fit_min_max()

        score = self.evaluate(Xs)  # when Xs is None, self.corr and self.n_features_on_party should be already set
        beta = (score - self.min_icor) / (self.max_icor - self.min_icor)
        if clip:
            return np.clip(beta, 0, 1).item()
        else:
            return beta

    def visualize(self, save_path=None, value=None, cmap='cividis', map_func=np.abs, fontsize=16, title_size=None):
        """
        Visualize the correlation matrix with the latest self.n_features_on_party and self.corr.
        :param map_func: [callable|None|str] function to map the correlation matrix. If None, the correlation matrix will
        be used directly.
        :param cmap: [str] color map for the figure
        :param save_path: [str|None] path to save the figure. If None, the figure will be shown.
        :param value: [float|None] The overall correlation score to be shown on the figure. If None, the score will not
        be shown.
        """
        if save_path and not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        if self.corr is None:
            raise ValueError("Please call fit() or fit_evaluate() first to calculate the correlation matrix.")
        if type(self.corr) == torch.Tensor:
            corr = self.corr.cpu().numpy()
        else:
            corr = self.corr

        if title_size is None:
            title_size = fontsize

        if map_func is None:
            map_corr = corr
        elif isinstance(map_func, str):
            # corr is in [-1, 1]
            if map_func == 'sigmoid':
                map_corr = 1 / (1 + np.exp(-corr))
            elif map_func == 'tanh':
                map_corr = np.tanh(corr)
            elif map_func == 'relu':
                map_corr = np.maximum(0, corr)
            elif map_func == 'log':
                map_corr = np.log(corr + 1)
            elif map_func == 'exp':
                map_corr = np.exp(corr)
            elif map_func == 'abs':
                map_corr = np.abs(corr)
            else:
                raise ValueError(f"Unknown map function {map_func} (str)")
        elif callable(map_func):
            map_corr = map_func(corr)
        else:
            raise ValueError(f"Unknown map function {map_func} (callable)")

        plt.rc('font', size=fontsize)

        fig, ax = plt.subplots(figsize=(10, 9))
        plt.imshow(map_corr, cmap=cmap)
        plt.colorbar()

        # add the axis to indicate the party
        n_features_on_party = np.asarray(self.n_features_on_party)
        n_parties = len(n_features_on_party)
        xtick_major = np.cumsum([0] + self.n_features_on_party)
        xtick_minor = (xtick_major[1:] + xtick_major[:-1]) / 2
        xtick_labels = [f"Party {i + 1}" for i in range(n_parties)]

        # add lines to separate the parties
        for i in range(n_parties - 1):
            plt.axvline(x=xtick_major[i + 1] - 0.5, color='red', linewidth=2)
            plt.axhline(y=xtick_major[i + 1] - 0.5, color='red', linewidth=2)

        ax.xaxis.set_major_locator(ticker.FixedLocator(xtick_major))
        ax.yaxis.set_major_locator(ticker.FixedLocator(xtick_major))
        ax.xaxis.set_minor_locator(ticker.FixedLocator(xtick_minor))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(xtick_minor))
        ax.set_xticklabels(xtick_labels, minor=True)
        ax.set_yticklabels(xtick_labels, minor=True)
        ax.tick_params(axis='both', which='minor', length=0)
        ax.tick_params(axis='x', which='minor', rotation=90)
        ax.tick_params(axis='x', which='major', rotation=90)

        fig.tight_layout()

        # move title up
        if value is not None:
            plt.title(f"Correlation matrix (Icor={value:.2f})", y=1.05, fontsize=title_size)
        else:
            plt.title("Correlation matrix", y=1.05, fontsize=title_size)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    X, y = load_svmlight_file("data/real/vehicle/processed/vehicle.libsvm")
    X = X.toarray()

    Xs = [X[:, :50], X[:, 50:]]
    evaluator = CorrelationEvaluator(gpu_id=0)
    score = evaluator.fit_evaluate(Xs)
    evaluator.visualize(value=score, cmap="afmhot")

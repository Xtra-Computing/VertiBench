import warnings
from typing import Iterable
import time

import deprecated
import numpy as np
import pandas as pd
import torch
import torch.linalg
from scipy.stats import spearmanr, hmean, gmean
from sklearn.utils.extmath import randomized_svd
from torchmetrics.functional import spearman_corrcoef
import shap
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

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


@deprecated.deprecated(reason="This joblib parallel is much more slower than a single thread")
def parallel_spearmanr(X):
    """
    Calculate the correlation matrix of X in parallel
    :param X: [np.ndarray] 2D data matrix. Size: n_samples * n_features
    :return: [np.ndarray] correlation matrix. Size: n_features * n_features
    """

    # replace all NaN values with 0 (NaN values are caused by constant features, the covariance of which is 0)
    X = np.nan_to_num(X, nan=0)
    n_features = X.shape[1]
    corr = np.zeros((n_features, n_features))

    def spearmanr_ij(i, j):
        return spearmanr(X[:, i], X[:, j]).statistic

    def spearmanr_i(i):
        return [spearmanr_ij(i, j) for j in range(n_features)]

    # calculate the correlation matrix in parallel
    corr = Parallel(n_jobs=-1)(delayed(spearmanr_i)(i) for i in range(n_features))
    return np.array(corr).reshape(n_features, n_features)


class CorrelationEvaluator:
    """
    Correlation evaluator for VFL datasets
    """
    def __init__(self, corr_func='spearmanr', gamma=1.0, gpu_id=None):
        """
        :param corr_func: [str] function to calculate the correlation between two features
        :param gamma: [float] weight of the inner-party correlation score
        :param gpu_id: [int] GPU id to use. If None, use CPU
        """
        assert corr_func in ["spearmanr"], "corr_func should be spearmanr"
        self.gamma = gamma
        self.corr = None
        self.n_features_on_party = None
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            self.device = torch.device(f"cuda:{self.gpu_id}")
            if corr_func == "spearmanr":
                self.corr_func = self.spearmanr     # use CPU for now, a bug in GPU version
        else:
            self.device = torch.device("cpu")
            if corr_func == "spearmanr":
                self.corr_func = self.spearmanr
        print(f"CorrelationEvaluator uses {self.device}")

    def spearmanr_gpu(self, X):
        """
        Calculate the correlation matrix of X using GPU
        :param X: [np.ndarray] 2D data matrix. Size: n_samples * n_features
        :return: [np.ndarray] correlation matrix. Size: n_features * n_features
        """
        # When there are constant features in X. The correlation may be NaN, raise a warning "numpy ignore divide by
        # zero warning". We ignore this warning and replace NaN in corr with 0.
        X = torch.from_numpy(X).float().to(self.device)
        corr = spearman_corrcoef(X, X)
        corr = torch.nan_to_num(corr, nan=0)
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
            corr = spearmanr(X).correlation
        corr = np.nan_to_num(corr, nan=0)
        if self.gpu_id is not None:
            corr = torch.from_numpy(corr).float().to(self.device)
        return corr

    @staticmethod
    def mcor_singular_naive(corr):
        """
        Calculate the overall correlation score of a correlation matrix using the variance of singular values.
        This is a naive implementation of the method that calculates all singular values. This is usually 2 to
        3 times slower than mcor_singular_exact().
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
        score = np.std(singular_values)

        # end_time = time.time()
        # print(f"Time for calculating the correlation score: {end_time - start_time}")
        return score

    @staticmethod
    def mcor_singular_exact(corr):
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
        # start_time = time.time()
        assert np.isnan(corr).any() == False, "NaN values should be replaced with 0"
        EX2 = np.linalg.norm(corr, ord='fro') ** 2 / min(corr.shape)
        EX = np.linalg.norm(corr, ord='nuc') / min(corr.shape)
        score = np.sqrt(EX2 - EX ** 2)

        # end_time = time.time()
        # print(f"Time for calculating the correlation score: {end_time - start_time}")
        return score

    def mcor_singular_exact_gpu(self, corr: torch.Tensor):
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
        # start_time = time.time()
        EX2 = torch.norm(corr, p='fro') ** 2 / min(corr.shape)
        EX = torch.norm(corr, p='nuc') / min(corr.shape)
        score = torch.sqrt(EX2 - EX ** 2)

        # end_time = time.time()
        # print(f"Time for calculating the correlation score: {end_time - start_time}")
        return float(score.item())

    @staticmethod
    def mcor_singular_approx(corr, n_components=400, n_oversamples=10, n_iter=4, random_state=0):
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

        assert np.isnan(corr).any() == False, "NaN values should be replaced with 0"
        _, singular_values, _ = randomized_svd(corr, n_components=n_components, n_oversamples=n_oversamples, n_iter=n_iter,
                                 random_state=random_state)
        singular_shape = min(corr.shape)
        s_append_zero = np.concatenate((singular_values, np.zeros(singular_shape - singular_values.shape[0])))
        score = np.std(s_append_zero)

        # end_time = time.time()
        # print(f"Time for calculating the correlation score: {end_time - start_time}")
        return score

    @staticmethod
    def mcor_singular_approx_gpu(corr: torch.Tensor, n_components=400, n_iter=4):
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
        # start_time = time.time()

        # assert torch.isnan(corr).any() == False, "NaN values should be replaced with 0"

        _, singular_values, _ = torch.svd_lowrank(corr, q=n_components, niter=n_iter)
        singular_shape = min(corr.shape)
        s_append_zero = torch.concatenate((singular_values,
                                           torch.zeros(singular_shape - singular_values.shape[0]).to(corr.device)))
        score = torch.std(s_append_zero)

        # end_time = time.time()
        # print(f"Time for calculating the correlation score: {end_time - start_time}")
        return float(score.item())

    def mcor_singular(self, corr, algo='auto', **kwargs):
        """
        Calculat the std of the singular values of corr matrix.
        :param corr: [np.ndarray] correlation matrix
        :param algo: [str] algorithm to calculate the overall correlation score of a correlation matrix
                    - 'auto': automatically choose the algorithm based on the size of the correlation matrix
                              if the size is smaller than 200, use 'exact', otherwise use 'approx'
                    - 'exact': calculate the exact singular values
                    - 'approx': calculate the approximate singular values
        :param kwargs:
        :return:
        """
        if algo == 'auto':
            if min(corr.shape) < 1000:
                if self.gpu_id is not None:
                    return self.mcor_singular_exact_gpu(corr)
                else:
                    return CorrelationEvaluator.mcor_singular_exact(corr)
            else:
                if self.gpu_id is not None:
                    return self.mcor_singular_approx_gpu(corr, **kwargs)
                else:
                    return CorrelationEvaluator.mcor_singular_approx(corr, **kwargs)
        elif algo == 'exact':
            if self.gpu_id is not None:
                return self.mcor_singular_exact_gpu(corr)
            else:
                return CorrelationEvaluator.mcor_singular_exact(corr)
        elif algo == 'approx':
            if self.gpu_id is not None:
                return self.mcor_singular_approx_gpu(corr, **kwargs)
            else:
                return CorrelationEvaluator.mcor_singular_approx(corr, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    def _get_inner_and_inter_corr(self, corr, n_features_on_party):
        """
        Calculate the inner-party and inter-party correlation matrices.
        :param corr: [np.ndarray] correlation matrix
        :param n_features_on_party: [list] number of features on each party
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
        inter_mcors = []
        for i in range(n_parties):
            for j in range(n_parties):
                start_i = corr_cut_points[i]
                end_i = corr_cut_points[i + 1]
                start_j = corr_cut_points[j]
                end_j = corr_cut_points[j + 1]
                if i == j:
                    inner_mcors.append(self.mcor_singular(corr[start_i:end_i, start_j:end_j]))
                else:
                    inter_mcors.append(self.mcor_singular(corr[start_i:end_i, start_j:end_j]))

        return np.array(inner_mcors), np.array(inter_mcors)

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

        inter_mcors = []
        for i in range(n_parties):
            for j in range(n_parties):
                start_i = corr_cut_points[i]
                end_i = corr_cut_points[i + 1]
                start_j = corr_cut_points[j]
                end_j = corr_cut_points[j + 1]
                if symmetric:
                    save = i < j
                else:
                    save = i != j
                if save:
                    inter_mcors.append(self.mcor_singular(corr[start_i:end_i, start_j:end_j]))

        return np.array(inter_mcors)

    def overall_corr_score(self, corr, n_features_on_party, mcor_func: callable = mcor_singular):
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
        :param mcor_func: [callable] function to calculate the overall correlation score of a correlation matrix
        :return: [float] correlation score
        """
        inter_mcors = self._get_inter_corr(corr, n_features_on_party, mcor_func)
        return np.mean(inter_mcors)

    @deprecated.deprecated(reason="This function is deprecated. Please use overall_corr_score instead.")
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
        arithmetic mean of differences between the inter-party correlation scores and the inner-party correlation
        scores for each party.
        :param corr: [np.ndarray] correlation matrix
        :param n_features_on_party: [list] number of features on each party
        :param mcor_func: [callable] function to calculate the correlation score of a correlation matrix
        :return: [float] correlation score
        """
        start_time = time.time()
        inner_mcors, inter_mcors = self._get_inner_and_inter_corr(corr, n_features_on_party, mcor_func)
        end_time = time.time()
        print(f"Time to calculate variance of singular values: {end_time - start_time:.2f} seconds")


        start_time = time.time()
        inter_mean_mcor = np.mean(np.array(inter_mcors).flatten())
        inner_mean_mcor = np.mean(np.array(inner_mcors).flatten())
        assert inter_mean_mcor >= 0, f"inter_mean_mcor: {inter_mean_mcor} should be non-negative"
        end_time = time.time()
        print(f"Time to summarize correlation scores: {end_time - start_time:.2f} seconds")

        return inter_mean_mcor - self.gamma * inner_mean_mcor

    @deprecated.deprecated(reason="This function is deprecated. Please use overall_corr_score instead.")
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
        self.n_features_on_party = self.check_data(Xs)
        start_time = time.time()
        self.corr = self.corr_func(np.concatenate(Xs, axis=1))
        self.corr = torch.nan_to_num(self.corr, nan=0)
        end_time = time.time()
        print(f"Correlation calculation time: {end_time - start_time:.2f}s")
        return self.overall_corr_score(self.corr, self.n_features_on_party)

    def fit(self, Xs):
        """
        Fit the correlation matrix of a vertical federated learning dataset.
        :param Xs: [List|Tuple] list of feature matrices of each party
        """
        self.n_features_on_party = self.check_data(Xs)
        self.corr = self.corr_func(np.concatenate(Xs, axis=1))
        self.corr = np.nan_to_num(self.corr, nan=0)

    def evaluate(self):
        """
        Evaluate the correlation score of a vertical federated learning dataset with self.corr.
        :return: [float] correlation score
        """
        return self.overall_corr_score(self.corr, self.n_features_on_party)

    def visualize(self, save_path=None, value=None):
        """
        Visualize the correlation matrix.
        :param save_path: [str|None] path to save the figure. If None, the figure will be shown.
        :param value: [float|None] The overall correlation score to be shown on the figure. If None, the score will not
        be shown.
        """
        if self.corr is None:
            raise ValueError("Please call fit() or fit_evaluate() first to calculate the correlation matrix.")
        if type(self.corr) == torch.Tensor:
            corr = self.corr.cpu().numpy()
        else:
            corr = self.corr
        plt.figure(figsize=(10, 10))
        plt.imshow(corr, cmap='plasma')
        plt.colorbar()
        if value is not None:
            plt.title(f"Correlation matrix (inter-mcor={value:.2f})")
        else:
            plt.title("Correlation matrix")
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

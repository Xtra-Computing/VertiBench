import time

import numpy as np
from src.preprocess.FeatureSplitter import ImportanceSplitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification, load_svmlight_file
from sklearn.utils.extmath import randomized_svd
import xgboost as xgb
import pandas as pd

import torch

from dataset.LocalDataset import LocalDataset
from dataset.VFLDataset import VFLDataset, VFLAlignedDataset
from preprocess.FeatureSplitter import ImportanceSplitter, CorrelationSplitter
from preprocess.FeatureEvaluator import ImportanceEvaluator, CorrelationEvaluator, parallel_spearmanr

# X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, n_repeated=0,
#                                n_classes=2, random_state=0, shuffle=True)
# X = MinMaxScaler().fit_transform(X)
# model = LogisticRegression()
# model.fit(X, y)
#
# n_features = 10
# n_parties = 3
# for i, w in enumerate([1e-2, 0.1, 0.5, 0.8, 1, 2, 5, 1e2]):
#     splitter = ImportanceSplitter(num_parties=n_parties, weights=w, seed=i)
#     n_features_summary = []
#     rs_summary = []
#     # np.random.seed(i)
#     for j in range(10000):
#         Xs = splitter.split(X)
#         n_features_per_party = [X.shape[1] for X in Xs]
#
#         # rs = np.random.dirichlet(np.repeat(w, n_parties))    # rs.shape = (n_parties)
#         # rs_summary.append(rs)
#         # n_features_per_party = np.zeros(n_parties)
#         # party_ids = np.random.choice(n_parties, size=n_features, p=rs)
#         # for party_id in party_ids:
#         #     n_features_per_party[party_id] += 1
#
#         n_features_summary.append(n_features_per_party)
#
#     # rs_std = np.std(rs_summary, axis=1)
#     n_features_std = np.std(n_features_summary, axis=1)
#     # print(n_features_std)
#     print(f"{np.mean(n_features_std)=}, {w=}")



# print("Loading data...")
# data = "gisette"
# X, y = load_svmlight_file(f"data/syn/{data}/{data}.libsvm")
# X = X.toarray()
# print("Data loaded.")

from scipy.stats import spearmanr
import time

# print("Loading data...")
# X, y = load_svmlight_file(f"data/syn/mnist/mnist.libsvm")
# X = X.toarray()
# print(f"Data loaded. {X.shape=}")
#
# c = spearmanr(X, X).correlation
# pass

# c = np.random.rand(60000, 784)
#
# start_time = time.time()
# evaluator = CorrelationEvaluator(gpu_id=0)
# result_torch = evaluator.spearmanr_gpu(c)
# print(f"Pandas time seconds: {time.time() - start_time}")
#
# start_time = time.time()
# result_parallel = parallel_spearmanr(c)
# print(f"Parallel time seconds: {time.time() - start_time}")
#
# start_time = time.time()
# result_scipy = spearmanr(c).correlation
# print(f"Scipy time seconds: {time.time() - start_time}")
#
# start_time = time.time()
# result_pandas = pd.DataFrame(c).corr(method='spearman')
# print(f"Pandas time seconds: {time.time() - start_time}")
#
# assert np.allclose(result_torch, result_scipy)
# assert np.allclose(result_torch, result_pandas.values)
#


a = [1, 1, 1]
b = [0, 0, 0]

print(spearmanr(a, b))

a_ = a[:]
a_[0] += 1e-6
b_ = b[:]
b_[0] += 1e-6
print(spearmanr(a_, b_))



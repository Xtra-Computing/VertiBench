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



print("Loading data...")
data = "gisette"
X, y = load_svmlight_file(f"data/syn/{data}/{data}.libsvm")
X = X.toarray()
print("Data loaded.")


# # evenly split the features of X into 4 parties
# print("Splitting features...")
# Xs = np.array_split(X, 4, axis=1)
# print("Features splitted.")
#
# print("Evaluating...")
# start = time.time()
# evaluator = CorrelationEvaluator()
# score = evaluator.fit_evaluate(Xs)
# end = time.time()
# print(f"Evaluation finished. Time cost: {end - start}s")
#
# print(score)

# print("Calculate the standard variance of singular values")
# evaluator = CorrelationEvaluator()
# evaluator.fit([X])
#
#
# start = time.time()
# corr_mtx = torch.tensor(evaluator.corr, dtype=torch.float32).to('cuda:0')
# EX2 = np.linalg.norm(evaluator.corr, ord='fro') ** 2 / min(evaluator.corr.shape)    # faster
# EX = torch.norm(corr_mtx, p='nuc') / min(evaluator.corr.shape)
# EX = EX.cpu().numpy()
# var = np.sqrt(EX2 - EX ** 2)
# print(f"Singular value variance: {var}, time cost: {time.time() - start}s")
#
#
# start = time.time()
# corr_mtx = torch.tensor(evaluator.corr, dtype=torch.float32).to('cuda:0')
# _, s, _ = torch.svd_lowrank(corr_mtx, q=400, niter=4)
# s = s.cpu().numpy()
# shape = min(evaluator.corr.shape)
# s_append_zero = np.concatenate((s, np.zeros(shape - s.shape[0])))
# var1 = np.std(s_append_zero)
# print(f"Singular value variance: {var1}, time cost: {time.time() - start}s")
#
#
# print("Calculate the standard variance of singular values")
# evaluator = CorrelationEvaluator()
# evaluator.fit([X])
#
# start = time.time()
# _, s, _ = randomized_svd(evaluator.corr, n_components=400, n_oversamples=10,
#                          n_iter=4, random_state=0)
# shape = min(evaluator.corr.shape)
# s_append_zero = np.concatenate((s, np.zeros(shape - s.shape[0])))
# var1 = np.std(s_append_zero)
# print(f"Singular value variance: {var1}, time cost: {time.time() - start}s")
#
#
#
# start = time.time()
# # EX2 = np.matrix.trace(evaluator.corr.T @ evaluator.corr) / evaluator.corr.shape[1]
# EX2 = np.linalg.norm(evaluator.corr, ord='fro') ** 2 / min(evaluator.corr.shape)    # faster
# end = time.time()
# print(f"Time cost: {end - start}s")
# E2X = (np.linalg.norm(evaluator.corr, ord='nuc') / min(evaluator.corr.shape)) ** 2
# var = np.sqrt(EX2 - E2X)
# print(f"Singular value variance: {var}, time cost: {time.time() - start}s")
#
# start = time.time()
# score_var = evaluator.mcor_singular(evaluator.corr)
# print(f"Singular value variance: {score_var}, time cost: {time.time() - start}s")
#
#
#
# higgs = pd.read_csv("data/syn/higgs/higgs.csv")
# pass


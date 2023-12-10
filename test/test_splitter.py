import unittest
import random
from itertools import product
from collections import defaultdict
from sklearn.datasets import make_classification

import numpy as np
from scipy.stats import spearmanr
import xgboost as xgb
from src.vertibench.Splitter import ImportanceSplitter, CorrelationSplitter, SimpleSplitter
from src.vertibench.Evaluator import ImportanceEvaluator, CorrelationEvaluator


def generate_data():
    X_1_10, y_1_10 = make_classification(n_samples=1, n_features=10, n_informative=10, n_redundant=0)
    X_1_500, y_1_500 = make_classification(n_samples=1, n_features=500, n_informative=20, n_redundant=0)
    X_10_10, y_10_10 = make_classification(n_samples=10, n_features=10, n_informative=10, n_redundant=0)
    X_1k_10, y_1k_10 = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0)
    X_1k_500, y_1k_500 = make_classification(n_samples=1000, n_features=500, n_informative=20, n_redundant=0)

    data_dict = {
        "1_10": (X_1_10, y_1_10),
        "1_500": (X_1_500, y_1_500),
        "10_10": (X_10_10, y_10_10),
        "1k_10": (X_1k_10, y_1k_10),
        "1k_500": (X_1k_500, y_1k_500),
    }
    return data_dict


class TestImportanceSplitter(unittest.TestCase):
    def setUp(self):
        self.n_parties_list = [0, 1, 2, 4, 32, 512]
        self.const_weight_list = [0.0, 1e-6, 1e-3, 1, 1e3, 1e6]
        np.random.seed(0)
        self.weights10_list = np.random.exponential(5, size=(5, 10))    # 5 different weights for 10 parties
        self.weights1k_list = np.random.exponential(5, size=(5, 1000))  # 5 different weights for 1000 parties
        self.rounds = 1000
        self.data_dict = generate_data()

    def test_dirichlet(self):
        # the ImportanceSplitter.dirichlet function should has the same distribution as np.random.dirichlet

        for sz in range(2, 100, 20):
            Xs = []
            Ys = []
            for _ in range(10000):
                alpha = np.random.exponential(5, size=sz).flatten()
                alpha = alpha / np.sum(alpha)
                xs = np.random.dirichlet(alpha)
                if np.isnan(xs).any():
                    continue
                ys = ImportanceSplitter.dirichlet(alpha)

                Xs.append(xs)
                Ys.append(ys)

            Xs = np.array(Xs)
            Ys = np.array(Ys)
            bins = np.arange(0, 1.01, 0.05)
            for col in range(sz):
                hist_Xs = np.histogram(Xs[:, col], bins=bins)[0] / Xs.shape[0]
                hist_Ys = np.histogram(Ys[:, col], bins=bins)[0] / Ys.shape[0]
                self.assertTrue(np.allclose(hist_Xs, hist_Ys, atol=2e-2), msg=f"col={col}, sz={sz}, "
                                                                              f"diff={abs(hist_Xs - hist_Ys)}")

    def check_params(self):
        splitters = []
        for n_parties, const_weight in product(self.n_parties_list, self.const_weight_list):
            if n_parties <= 1 or const_weight <= 0.0:
                with self.assertRaises(ValueError):
                    ImportanceSplitter(n_parties, const_weight)
            else:
                splitters.append(ImportanceSplitter(n_parties, const_weight))

        n_parties = 10
        for weights in self.weights10_list:
            if any([weight <= 0.0 for weight in weights]):
                with self.assertRaises(ValueError):
                    ImportanceSplitter(n_parties, weights=weights)
            splitters.append(ImportanceSplitter(n_parties, weights=weights))

        return splitters

    def test_split_indices(self):
        splitters = self.check_params()
        for key, (X, y) in self.data_dict.items():
            # train a model on each split
            model = xgb.XGBClassifier()
            model.fit(X, y)
            feature_importances = np.array(model.feature_importances_)

            const_weight_stds_all_parties = defaultdict(list)
            for splitter in splitters:
                if splitter.num_parties > X.shape[1]:
                    continue
                with self.subTest(key=key, n_parties=splitter.num_parties):
                    std_feature_cnt_summary = []
                    std_summary = []
                    mean_by_party_summary = [0 for _ in range(splitter.num_parties)]
                    for _ in range(self.rounds):
                        split_indices = splitter.split_indices(X, allow_empty_party=False)
                        self.assertEqual(len(split_indices), splitter.num_parties)
                        self.assertEqual(sum([len(indices) for indices in split_indices]), X.shape[1])
                        for indices in split_indices:
                            self.assertGreater(len(indices), 0) # no empty party
                        importance_per_party = []
                        for party_id, indices in enumerate(split_indices):
                            score = np.sum(feature_importances[indices])
                            importance_per_party.append(score)
                            mean_by_party_summary[party_id] += score
                        std_summary.append(np.std(importance_per_party, ddof=1))
                        std_feature_cnt_summary.append(np.std([len(indices) for indices in split_indices], ddof=1))
                    mean_by_party_summary = np.array([score / self.rounds for score in mean_by_party_summary])

                    if X.shape[0] <= 1:
                        continue

                    # mean_by_party_summary should be proportional to weights
                    self.assertEqual(len(mean_by_party_summary), len(splitter.weights))
                    self.assertGreater(np.sum(mean_by_party_summary), 0.0, msg=f"key={key}, weights={splitter.weights},"
                                                                               f" mean_by_party_summary={mean_by_party_summary}")
                    scaled_importance = mean_by_party_summary / np.sum(mean_by_party_summary)
                    scaled_weights = splitter.weights / np.sum(splitter.weights)
                    rmse = np.sqrt(np.mean((scaled_importance - scaled_weights) ** 2))
                    self.assertLessEqual(rmse, 0.1, msg=f"{scaled_importance=}, {scaled_weights=}")

                    if np.all(np.isclose(splitter.weights, splitter.weights[0])):
                        const_weight_stds_all_parties[splitter.num_parties].append(
                            (splitter.weights[0], np.mean(std_feature_cnt_summary), np.mean(std_summary)))

            for n_parties, const_weight_stds in const_weight_stds_all_parties.items():
                with self.subTest(key=key, n_parties=n_parties):
                    if len(const_weight_stds) == 0:
                        continue
                    const_weight_stds = np.array(const_weight_stds)
                    # three columns should be ordered
                    order0 = np.argsort(const_weight_stds[:, 0])[::-1]
                    order1 = np.argsort(const_weight_stds[:, 1])
                    order2 = np.argsort(const_weight_stds[:, 2])
                    cor1 = spearmanr(order0, order1)[0]
                    cor2 = spearmanr(order0, order2)[0]
                    self.assertGreater(cor1, 0.6, msg=f"key={key}, cor1={cor1}, cor2={cor2}")
                    self.assertGreater(cor2, 0.6, msg=f"key={key}, cor1={cor1}, cor2={cor2}")

    def test_split_tabular(self):
        # assuming that the split_indices function is correct

        # test with/without given indices
        for key, (X, y) in self.data_dict.items():
            if X.shape[1] < 10:
                continue
            with self.subTest(task="tabular", key=key):
                splitter1 = ImportanceSplitter(10, seed=0)
                split_indices = splitter1.split_indices(X, allow_empty_party=False)
                Xs1 = splitter1.split(X, indices=split_indices)
                splitter2 = ImportanceSplitter(10, seed=0)
                Xs2 = splitter2.split(X)
                self.assertEqual(len(Xs1), 10)
                self.assertEqual(len(Xs2), 10)
                for i in range(10):
                    self.assertTrue(np.allclose(Xs1[i], Xs2[i]))

    def test_split_tabular_fill(self):
        # assuming that the split_indices function is correct

        # test with/without given indices
        for key, (X, y) in self.data_dict.items():
            if X.shape[1] < 10:
                continue
            with self.subTest(task="fill", key=key):
                splitter1 = ImportanceSplitter(10, seed=0)
                split_indices = splitter1.split_indices(X)
                Xs1 = splitter1.split(X, indices=split_indices, fill=-1)
                splitter2 = ImportanceSplitter(10, seed=0)
                Xs2 = splitter2.split(X, fill=-1)
                self.assertEqual(len(Xs1), 10)
                self.assertEqual(len(Xs2), 10)
                for i in range(10):
                    self.assertTrue(np.allclose(Xs1[i], Xs2[i]))

                # check if the fill value is correct
                for i in range(X.shape[0]):
                    for X1 in Xs1:
                        fill_cnt = np.count_nonzero(X1[i] == -1)
                        self.assertLessEqual(fill_cnt, X.shape[1] * (10 - 1))

    def test_split_multi_tabular(self):
        # assuming that the split_indices function is correct

        # test with/without given indices
        X_list = []
        for key, (X, y) in self.data_dict.items():
            if X.shape[1] != 10:
                continue
            X_list.append(X)

        with self.subTest(task="multi_tabular"):
            splitter = ImportanceSplitter(10, seed=0)
            Xs = splitter.split(*X_list)
            self.assertEqual(len(Xs), len(X_list))
            for i in range(len(X_list)):
                self.assertEqual(len(Xs[i]), 10)

            # concat all the parties should be the same as the original data
            X_all = np.concatenate(X_list, axis=0)
            Xs_all = []
            for i in range(len(X_list)):
                Xs_all.append(np.concatenate([Xs[i][j] for j in range(10)], axis=1))
            X_all_split = np.concatenate(Xs_all, axis=0)
            self.assertTrue(np.allclose(np.sort(X_all), np.sort(X_all_split)))

    def test_split_multi_tabular_fill(self):
        # assuming that the split_indices function is correct

        # test with/without given indices
        X_list = []
        for key, (X, y) in self.data_dict.items():
            if X.shape[1] != 10:
                continue
            X_list.append(X)

        with self.subTest(task="multi_tabular_fill"):
            splitter = ImportanceSplitter(10, seed=0)
            Xs = splitter.split(*X_list, fill=-1)
            self.assertEqual(len(Xs), len(X_list))
            for i in range(len(X_list)):
                self.assertEqual(len(Xs[i]), 10)

            # concat all the parties should be the same as the original data
            X_all = np.concatenate(X_list, axis=0)
            Xs_all = []
            for i in range(len(X_list)):
                Xs_all.append(np.concatenate([Xs[i][j] for j in range(10)], axis=1))
            X_all_split = np.concatenate(Xs_all, axis=0)
            self.assertEqual(X_all_split.shape[1], X_all.shape[1] * 10)
            X_all_split_filled = X_all_split[X_all_split != -1].reshape(X_all.shape)
            self.assertTrue(np.allclose(np.sort(X_all), np.sort(X_all_split_filled)))


class TestCorrelationSplitter(unittest.TestCase):
    def setUp(self):
        self.n_parties_list = [0, 1, 2, 4, 32, 512]
        self.beta_list = [0.0, 1e-6, 0.5, 1, 1.1]

        self.rounds = 1000
        self.data_dict = generate_data()

    def test_fit_split(self):
        for key, (X, y) in self.data_dict.items():
            if X.shape[1] <= 1 or X.shape[0] <= 2:
                continue
            for n_parties in self.n_parties_list:
                for beta in self.beta_list:
                    with self.subTest(key=key, n_parties=n_parties, beta=beta):
                        splitter = CorrelationSplitter(n_parties, gpu_id=0)

                        if beta > 1.0 or beta < 0.0:
                            with self.assertRaises(ValueError):
                                splitter.fit_split(X, beta=beta)
                        Xs = splitter.fit_split(X, beta=beta)
                        beta_eval = splitter.evaluator.evaluate_beta(Xs)
                        self.assertAlmostEqual(beta, beta_eval, delta=0.1)




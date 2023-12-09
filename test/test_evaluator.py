import unittest

from collections import defaultdict
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

from src.vertibench.Evaluator import ImportanceEvaluator, CorrelationEvaluator


def generate_data():
    ###### Varying instances ######
    # Generate a small dataset with single instance
    X_1_10, y_1_10 = make_classification(n_samples=1, n_features=10, n_informative=5, n_redundant=3,
                                                   random_state=0, shuffle=True)
    X_2_10, y_2_10 = make_classification(n_samples=2, n_features=10, n_informative=5, n_redundant=3,
                                                    random_state=0, shuffle=True)

    # Generate a small dataset
    X_100_10, y_100_10 = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=3,
                                                       random_state=0, shuffle=True)

    # Generate a large dataset
    X_10k_10, large_y_10k_10 = make_classification(n_samples=10000, n_features=10, n_informative=5,
                                                             n_redundant=3,
                                                             random_state=0, shuffle=True)

    ##### Varying features #####
    # Generate a dataset with single feature
    X_1k_2, y_1k_2 = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                                                   random_state=0, shuffle=True)

    # Generate a dataset with large number of features
    X_1k_1k, y_1k_1k = make_classification(n_samples=1000, n_features=1000, n_informative=500,
                                                     n_redundant=200,
                                                     random_state=0, shuffle=True)

    ###### Constant features ######
    # Generate a dataset with constant features
    X_1k_10_const, y_1k_10_const = make_classification(n_samples=1000, n_features=8, n_informative=5,
                                                                 n_redundant=1,
                                                                 random_state=0, shuffle=True)
    X_1k_10_const = np.concatenate((X_1k_10_const, np.zeros((1000, 2))), axis=1)

    # Generate a dataset with constant features with single feature
    X_1k_2_const, y_1k_2_const = np.zeros((1000, 2)), np.zeros((1000, 1))

    # parse local variables to create data
    data = {'1_10': (X_1_10, y_1_10),
            '2_10': (X_2_10, y_2_10),
            '100_10': (X_100_10, y_100_10),
            '10k_10': (X_10k_10, large_y_10k_10),
            '1k_2': (X_1k_2, y_1k_2),
            '1k_1k': (X_1k_1k, y_1k_1k),
            '1k_10_const': (X_1k_10_const, y_1k_10_const),
            '1k_2_const': (X_1k_2_const, y_1k_2_const)}

    return data

def split_data(data):
    splits = defaultdict(list)
    for k, (X, y) in data.items():
        n_features = X.shape[1]
        step = n_features // 5 if n_features > 5 else 1
        for i in range(1, n_features, step):
            splits[k].append(([X[:, :i], X[:, i:]], y))
    return splits


class TestImportanceEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = ImportanceEvaluator()
        self.small_sample_evaluator = ImportanceEvaluator(sample_rate=0.001)

        self.data = generate_data()
        self.splits = split_data(self.data)

    def test_evaluate(self):
        for k, split_list in self.splits.items():
            for Xs, y in split_list:
                with self.subTest(k=k):
                    model = xgb.XGBClassifier()
                    X = np.concatenate(Xs, axis=1)
                    model.fit(X, y)
                    scores, small_scores = None, None
                    if X.shape[0] * self.evaluator.sample_rate <= 1:
                        self.assertRaises(ValueError, self.evaluator.evaluate, Xs, model.predict)
                    else:
                        scores = self.evaluator.evaluate(Xs, model.predict)
                        self.assertEqual(len(scores), len(Xs))

                    if X.shape[0] * self.small_sample_evaluator.sample_rate <= 1:
                        self.assertRaises(ValueError, self.small_sample_evaluator.evaluate, Xs, model.predict)
                    else:
                        small_scores = self.small_sample_evaluator.evaluate(Xs, model.predict)
                        self.assertEqual(len(small_scores), len(Xs))
                    if scores is not None and small_scores is not None:
                        self.assertAlmostEqual(scores[0], small_scores[0], delta=0.1)

    def test_evaluate_feature(self):
        for k, (X, y) in self.data.items():
            with self.subTest(k=k):
                model = xgb.XGBClassifier()
                model.fit(X, y)
                scores, small_scores = None, None
                if X.shape[0] * self.evaluator.sample_rate <= 1:
                    self.assertRaises(ValueError, self.evaluator.evaluate_feature, X, model.predict)
                else:
                    scores = self.evaluator.evaluate_feature(X, model.predict)
                    self.assertEqual(len(scores), X.shape[1])

                if X.shape[0] * self.small_sample_evaluator.sample_rate <= 1:
                    self.assertRaises(ValueError, self.small_sample_evaluator.evaluate_feature, X, model.predict)
                else:
                    small_scores = self.small_sample_evaluator.evaluate_feature(X, model.predict)
                    self.assertEqual(len(small_scores), X.shape[1])

    def test_multi_party_evaluate(self):
        X, y = self.data['1k_1k']
        for n_parties in [2, 3, 4, 10, 100, 200, 500, 1000]:
            with self.subTest(n_parties=n_parties):
                Xs = np.array_split(X, n_parties, axis=1)
                model = xgb.XGBClassifier()
                model.fit(X, y)
                if X.shape[0] * self.evaluator.sample_rate <= 1:
                    self.assertRaises(ValueError, self.evaluator.evaluate, Xs, model.predict)
                else:
                    scores = self.evaluator.evaluate(Xs, model.predict)
                    self.assertEqual(len(scores), len(Xs))

                if X.shape[0] * self.small_sample_evaluator.sample_rate <= 1:
                    self.assertRaises(ValueError, self.small_sample_evaluator.evaluate, Xs, model.predict)
                else:
                    small_scores = self.small_sample_evaluator.evaluate(Xs, model.predict)
                    self.assertEqual(len(small_scores), len(Xs))


class TestCorrelationEvaluator(unittest.TestCase):
    def setUp(self):
        self.spearmanr_evaluator = CorrelationEvaluator(method='spearmanr', gpu_id=None)
        self.pearsonr_evaluator = CorrelationEvaluator(method='pearson', gpu_id=None)
        self.gpu_spearmanr_evaluator = CorrelationEvaluator(method='spearmanr', gpu_id=0)
        self.gpu_pearsonr_evaluator = CorrelationEvaluator(method='pearson', gpu_id=0)

        self.data = generate_data()
        self.splits = split_data(self.data)

    def subtest_fit_evaluate(self, Xs):
        if Xs[0].shape[0] <= 1:
            self.assertRaises(ValueError, self.spearmanr_evaluator.fit_evaluate, Xs)
            self.assertRaises(ValueError, self.pearsonr_evaluator.fit_evaluate, Xs)
            self.assertRaises(ValueError, self.gpu_spearmanr_evaluator.fit_evaluate, Xs)
            self.assertRaises(ValueError, self.gpu_pearsonr_evaluator.fit_evaluate, Xs)
        else:
            spearmanr_score = self.spearmanr_evaluator.fit_evaluate(Xs)
            pearsonr_score = self.pearsonr_evaluator.fit_evaluate(Xs)
            spearmanr_score_gpu = self.gpu_spearmanr_evaluator.fit_evaluate(Xs)
            pearsonr_score_gpu = self.gpu_pearsonr_evaluator.fit_evaluate(Xs)

            # score should be in range [-1, 1]
            self.assertGreaterEqual(spearmanr_score, -1)
            self.assertLessEqual(spearmanr_score, 1)
            self.assertGreaterEqual(pearsonr_score, -1)
            self.assertLessEqual(pearsonr_score, 1)
            self.assertGreaterEqual(spearmanr_score_gpu, -1)
            self.assertLessEqual(spearmanr_score_gpu, 1)
            self.assertGreaterEqual(pearsonr_score_gpu, -1)
            self.assertLessEqual(pearsonr_score_gpu, 1)

            # GPU score should be the same as CPU score
            self.assertAlmostEqual(spearmanr_score, spearmanr_score_gpu, delta=0.0001)
            self.assertAlmostEqual(pearsonr_score, pearsonr_score_gpu, delta=0.0001)

            # correlation matrix should be in range [-1, 1]
            self.assertTrue((self.spearmanr_evaluator.corr <= 1).all())
            self.assertTrue((self.spearmanr_evaluator.corr >= -1).all())
            self.assertTrue((self.pearsonr_evaluator.corr <= 1).all())
            self.assertTrue((self.pearsonr_evaluator.corr >= -1).all())
            self.assertTrue((self.gpu_spearmanr_evaluator.corr <= 1).all())
            self.assertTrue((self.gpu_spearmanr_evaluator.corr >= -1).all())
            self.assertTrue((self.gpu_pearsonr_evaluator.corr <= 1).all())
            self.assertTrue((self.gpu_pearsonr_evaluator.corr >= -1).all())

            # overall correlation matrix should be symmetric
            self.assertTrue(np.allclose(self.spearmanr_evaluator.corr, self.spearmanr_evaluator.corr.T))
            self.assertTrue(np.allclose(self.pearsonr_evaluator.corr, self.pearsonr_evaluator.corr.T))
            self.assertTrue(np.allclose(self.gpu_spearmanr_evaluator.corr.detach().cpu().numpy(),
                                        self.gpu_spearmanr_evaluator.corr.T.detach().cpu().numpy()))
            self.assertTrue(np.allclose(self.gpu_pearsonr_evaluator.corr.detach().cpu().numpy(),
                                        self.gpu_pearsonr_evaluator.corr.T.detach().cpu().numpy()))

    def test_fit_evaluate(self):
        for k, split_list in self.splits.items():
            for Xs, y in split_list:
                with self.subTest(k=k):
                    self.subtest_fit_evaluate(Xs)

    def test_fit_multi_evaluate(self):
        for k, split_list in self.splits.items():
            Xs = split_list[0][0]
            if Xs[0].shape[0] <= 1:
                self.assertRaises(ValueError, self.spearmanr_evaluator.fit, Xs)
                self.assertRaises(ValueError, self.pearsonr_evaluator.fit, Xs)
                self.assertRaises(ValueError, self.gpu_spearmanr_evaluator.fit, Xs)
                self.assertRaises(ValueError, self.gpu_pearsonr_evaluator.fit, Xs)
                continue
            else:
                self.spearmanr_evaluator.fit(Xs)
                self.pearsonr_evaluator.fit(Xs)
                self.gpu_spearmanr_evaluator.fit(Xs)
                self.gpu_pearsonr_evaluator.fit(Xs)

            for Xs, y in split_list:
                with self.subTest(k=k):
                    spearmanr_score_eval = self.spearmanr_evaluator.evaluate(Xs)
                    pearsonr_score_eval = self.pearsonr_evaluator.evaluate(Xs)
                    spearmanr_score_gpu_eval = self.gpu_spearmanr_evaluator.evaluate(Xs)
                    pearsonr_score_gpu_eval = self.gpu_pearsonr_evaluator.evaluate(Xs)

                    gpu_spearmanr_evaluator = CorrelationEvaluator(method='spearmanr', gpu_id=0)
                    spearmanr_score_gpu_fit_eval = gpu_spearmanr_evaluator.fit_evaluate(Xs)
                    gpu_pearsonr_evaluator = CorrelationEvaluator(method='pearson', gpu_id=0)
                    pearsonr_score_gpu_fit_eval = gpu_pearsonr_evaluator.fit_evaluate(Xs)

                    # eval and fit_eval should be the same
                    self.assertAlmostEqual(spearmanr_score_eval, spearmanr_score_gpu_fit_eval, delta=0.0001)
                    self.assertAlmostEqual(pearsonr_score_eval, pearsonr_score_gpu_fit_eval, delta=0.0001)
                    self.assertAlmostEqual(spearmanr_score_gpu_eval, spearmanr_score_gpu_fit_eval, delta=0.0001)
                    self.assertAlmostEqual(pearsonr_score_gpu_eval, pearsonr_score_gpu_fit_eval, delta=0.0001)

    def test_visualize(self):
        for k, split_list in self.splits.items():
            # before calling fit, visualize should raise error
            self.assertRaises(ValueError, self.spearmanr_evaluator.visualize)
            self.assertRaises(ValueError, self.pearsonr_evaluator.visualize)
            self.assertRaises(ValueError, self.gpu_spearmanr_evaluator.visualize)
            self.assertRaises(ValueError, self.gpu_pearsonr_evaluator.visualize)

        for k, split_list in self.splits.items():
            Xs = split_list[0][0]
            if Xs[0].shape[0] <= 1:
                self.assertRaises(ValueError, self.spearmanr_evaluator.fit, Xs)
                self.assertRaises(ValueError, self.pearsonr_evaluator.fit, Xs)
                self.assertRaises(ValueError, self.gpu_spearmanr_evaluator.fit, Xs)
                self.assertRaises(ValueError, self.gpu_pearsonr_evaluator.fit, Xs)
                continue

            for Xs, y in split_list:
                with self.subTest(k=k):
                    self.spearmanr_evaluator.fit(Xs)
                    self.pearsonr_evaluator.fit(Xs)
                    self.gpu_spearmanr_evaluator.fit(Xs)
                    self.gpu_pearsonr_evaluator.fit(Xs)

                    self.spearmanr_evaluator.visualize(save_path=f"test/tmp/{k}_spearmanr.png")
                    self.pearsonr_evaluator.visualize(save_path=f"test/tmp/{k}_pearsonr.png")
                    self.gpu_spearmanr_evaluator.visualize(save_path=f"test/tmp/{k}_gpu_spearmanr.png")
                    self.gpu_pearsonr_evaluator.visualize(save_path=f"test/tmp/{k}_gpu_pearsonr.png")

    def test_multi_party_fit_evaluate(self):
        X, y = self.data['1k_1k']
        for n_parties in [2, 3, 4, 10, 100, 200, 500, 1000]:
            with self.subTest(n_parties=n_parties):
                Xs = np.array_split(X, n_parties, axis=1)
                self.subtest_fit_evaluate(Xs)


if __name__ == '__main__':
    unittest.main()

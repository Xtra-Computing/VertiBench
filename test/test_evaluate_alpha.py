import unittest

from collections import defaultdict
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression

from src.vertibench.Evaluator import ImportanceEvaluator, CorrelationEvaluator


class TestAlphaEvaluator(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=1000, n_features=100, n_informative=50, n_redundant=0,
                                             random_state=0)
    def test_evaluate_alpha(self):
        # Create a model
        model = xgb.XGBClassifier()
        model.fit(self.X, self.y)

        # Create an evaluator
        evaluator = ImportanceEvaluator(sample_rate=0.1)
        scores = evaluator.evaluate_feature(self.X, model.predict)

        split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        alpha1s = []
        alpha2s = []
        for split_ratio in split_ratios:
            with self.subTest(split_ratio=split_ratio):
                # Split the data
                l_idx = int(split_ratio * self.X.shape[1])
                Xs = [self.X[:, :l_idx], self.X[:, l_idx:]]
                ys = [self.y, self.y]
                l_score, r_score = np.sum(scores[:l_idx]) / np.sum(scores), np.sum(scores[l_idx:]) / np.sum(scores)

                # evaluate feature
                scores_per_party = evaluator.evaluate(Xs, model.predict)
                self.assertEqual(len(scores_per_party), 2)
                self.assertTrue(np.allclose(scores_per_party[0], l_score, atol=0.2),
                                msg=f"{scores_per_party[0]} != {l_score}")
                self.assertTrue(np.allclose(scores_per_party[1], r_score, atol=0.2),
                                msg=f"{scores_per_party[1]} != {r_score}")

                # Evaluate alpha
                alpha1 = evaluator.evaluate_alpha(scores=scores_per_party)
                alpha2 = evaluator.evaluate_alpha(Xs=Xs, model=model.predict)

                alpha1s.append(alpha1)
                alpha2s.append(alpha2)

        with self.subTest("overall"):
            # mse between two alpha evaluations should be small
            diff_scale = np.abs(np.array(alpha1s) - np.array(alpha2s)) / alpha2s
            self.assertLess(np.mean(diff_scale), 0.3)

            # Check monotonicity, alphas should be generally decreasing
            # use linear regression to check monotonicity
            lr = LinearRegression()
            lr.fit(np.array(split_ratios).reshape(-1, 1), np.array(alpha1s).reshape(-1, 1))
            self.assertGreater(lr.coef_[0][0], 0)

from sklearn.datasets import make_classification

# Generate a large dataset
X, y = make_classification(n_samples=10000, n_features=10)


from vertibench import Evaluator

evaluator = Evaluator.ImportanceEvaluator(sample_rate=0.1)

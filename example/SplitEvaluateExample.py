from sklearn.datasets import make_classification

# Generate a large dataset
X, y = make_classification(n_samples=10000, n_features=10)


from vertibench.Evaluator import ImportanceEvaluator, CorrelationEvaluator
from vertibench.Splitter import ImportanceSplitter, CorrelationSplitter
from sklearn.linear_model import LogisticRegression

# Split by importance
imp_splitter = ImportanceSplitter(num_parties=4, weights=[1, 1, 1, 3])
Xs = imp_splitter.split(X)

# Evaluate split by importance
model = LogisticRegression()
model.fit(X, y)
imp_evaluator = ImportanceEvaluator()
imp_scores = imp_evaluator.evaluate(Xs, model.predict)
alpha = imp_evaluator.evaluate_alpha(scores=imp_scores)
print(f"Importance scores: {imp_scores}, alpha: {alpha}")

# Split by correlation
corr_splitter = CorrelationSplitter(num_parties=4)
Xs = corr_splitter.fit_split(X)

# Evaluate split by correlation
corr_evaluator = CorrelationEvaluator()
corr_scores = corr_evaluator.fit_evaluate(Xs)
beta = corr_evaluator.evaluate_beta()
print(f"Correlation scores: {corr_scores}, beta: {beta}")


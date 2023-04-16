import numpy as np
from src.preprocess.FeatureSplitter import ImportanceSplitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification, load_svmlight_file
import xgboost as xgb

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




X, y = load_svmlight_file("data/syn/covtype/covtype.libsvm")
model = xgb.train({'objective': 'multi:softprob', 'tree_method': 'gpu_hist', 'num_class': 7, 'max_depth': 6,
           'eval_metric': 'merror', 'verbose_eval': True, 'gpu_id': 1, 'seed': 0},
          xgb.DMatrix(X.toarray(), label=y), num_boost_round=10)
y_pred = model.predict(xgb.DMatrix(X.toarray()))
acc = np.mean(np.argmax(y_pred, axis=1) == y)
print(f"acc={acc}")


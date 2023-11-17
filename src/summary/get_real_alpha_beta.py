import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import make_classification, make_regression
from scipy.stats import spearmanr
from xgboost import XGBClassifier, XGBRegressor

# import the path of the project
sys.path.append(os.path.abspath("src"))

from preprocess.FeatureSplitter import CorrelationSplitter, ImportanceSplitter
from preprocess.FeatureEvaluator import CorrelationEvaluator, ImportanceEvaluator
from dataset.WideDataset import WideDataset, WideGlobalDataset
from dataset.SatelliteDataset import SatelliteDataset


def pcor_eigen(corr):
    """Summarize the correlation matrix corr"""
    assert corr.shape[0] == corr.shape[1]   # eigenvalues are only defined for square matrices
    eigen_values = np.linalg.eigvals(corr)
    score = np.std(eigen_values, ddof=1)
    return score

def pcor_singular(corr):
    """Summarize the correlation matrix corr"""
    singular_values = np.linalg.svd(corr)[1]
    score = np.std(singular_values, ddof=1)
    return score


# Load Wide dataset
image_path = "data/real/nus-wide/images"
tag_path = "data/real/nus-wide/tags"
label_path = "data/real/nus-wide/labels"

train_test_wide_dataset = WideGlobalDataset.from_source(image_path, label_path)
train_wide_dataset = train_test_wide_dataset.train
test_wide_dataset = train_test_wide_dataset.test
wide_Xs = []
for dataset in train_wide_dataset.local_datasets:
    print(dataset.X.shape)
    wide_Xs.append(dataset.X)


# Load Vehicle dataset
vehicle_X, vehicle_y = load_svmlight_file("data/real/vehicle/processed/vehicle.libsvm")
vehicle_X = vehicle_X.toarray()
vehicle_y = vehicle_y.astype('int') - 1
vehicle_Xs = [vehicle_X[:, :50], vehicle_X[:, 50:]]


# Load satellite dataset
satellite_data = SatelliteDataset.from_pickle("data/real/satellite/cache/", n_jobs=8)
satellite_Xs = []
for dataset in satellite_data.local_datasets:
    satellite_Xs.append(dataset.X)
print(f"Size of Xs: {[X.shape for X in satellite_Xs]}")

# random sample some features for evaluation
satellite_n_features = 100
print(f"Random sample {satellite_n_features} features for evaluation")
satellite_Xs_sample_flatten = []
for X in satellite_Xs:
    X_flatten = X.reshape(X.shape[0], -1)
    X_sample = X_flatten[:, np.random.permutation(X_flatten.shape[1])[:satellite_n_features]]
    satellite_Xs_sample_flatten.append(X_sample)
print(f"Size of Xs: {[X.shape for X in satellite_Xs_sample_flatten]}")

# evaluate Icor for wide dataset
corr_evaluator_wide = CorrelationEvaluator(gpu_id=0)
wide_Xs = wide_Xs[:5]   # remove the tag features
icor_wide = corr_evaluator_wide.fit_evaluate(wide_Xs)
corr_evaluator_wide.visualize("fig/pcor-wide.png", value=icor_wide, fontsize=24)
print(f"icor for wide dataset: {icor_wide}")

# evaluate beta for wide dataset
wide_X = np.concatenate(wide_Xs, axis=1)
corr_splitter_wide = CorrelationSplitter(num_parties=5, evaluator=corr_evaluator_wide, gpu_id=0)
corr_splitter_wide.fit(wide_X, n_elites=200, n_offsprings=700, n_mutants=100, n_gen=100, verbose=True)
beta_wide = corr_splitter_wide.evaluate_beta(icor_wide)

print(f"beta for wide dataset: {beta_wide}, Icor: {icor_wide} in range [{corr_splitter_wide.min_mcor}, {corr_splitter_wide.max_mcor}]")


# evaluate Icor for vehicle dataset
corr_evaluator_vehicle = CorrelationEvaluator(gpu_id=0)
icor_vehicle = corr_evaluator_vehicle.fit_evaluate(vehicle_Xs)
corr_evaluator_vehicle.visualize("fig/pcor-vehicle.png", value=icor_vehicle, fontsize=28)
print(f"icor for vehicle dataset: {icor_vehicle}")

# evaluate beta for vehicle dataset
vehicle_X = np.concatenate(vehicle_Xs, axis=1)
corr_splitter_vehicle = CorrelationSplitter(num_parties=2, evaluator=corr_evaluator_vehicle, gpu_id=0)
corr_splitter_vehicle.fit(vehicle_X, n_elites=200, n_offsprings=700, n_mutants=100, n_gen=50, verbose=True)
beta_vehicle = corr_splitter_vehicle.evaluate_beta(icor_vehicle)

print(f"beta for vehicle dataset: {beta_vehicle}, Icor: {icor_vehicle} in range [{corr_splitter_vehicle.min_mcor}, {corr_splitter_vehicle.max_mcor}]")



# evaluate Icor for satellite dataset and plot
n_satellite_features = 10
satellite_Xs_sample_flatten_less_features = [X[:, :n_satellite_features] for X in satellite_Xs_sample_flatten]

corr_evaluator_satellite = CorrelationEvaluator(gpu_id=0)
icor_satellite = corr_evaluator_satellite.fit_evaluate(satellite_Xs_sample_flatten_less_features)
corr_evaluator_satellite.visualize("fig/pcor-satellite.png", value=icor_satellite, fontsize=16, title_size=28)
print(f"icor for satellite dataset: {icor_satellite}")

# evaluate beta for satellite dataset
satellite_X = np.concatenate(satellite_Xs_sample_flatten_less_features, axis=1)
corr_splitter_satellite = CorrelationSplitter(num_parties=16, evaluator=corr_evaluator_satellite, gpu_id=0)
corr_splitter_satellite.fit(satellite_X, n_elites=200, n_offsprings=700, n_mutants=100, n_gen=50, verbose=True)

beta_satellite = corr_splitter_satellite.evaluate_beta(icor_satellite)
print(f"beta for satellite dataset: {beta_satellite}, Icor: {icor_satellite} in range [{corr_splitter_satellite.min_mcor}, {corr_splitter_satellite.max_mcor}]")


# train a model for wide dataset to evaluate Shapley value
wide_model = XGBClassifier(n_estimators=50, max_depth=6, n_jobs=16, learning_rate=0.1, objective="binary:logistic", tree_method="gpu_hist")
wide_model.fit(wide_X, train_wide_dataset.local_datasets[0].y, verbose=True)

# evaluate importance of wide dataset
imp_evaluator_wide = ImportanceEvaluator(sample_rate=0.01)
imp_wide_scores = imp_evaluator_wide.evaluate(wide_Xs, wide_model.predict, max_evals=1500)
print(f"Importance of wide dataset: {imp_wide_scores}")

# estimate the alpha of wide dataset
wide_dir_ratio = imp_wide_scores / np.sum(imp_wide_scores)
wide_alpha = corr_splitter_wide.evaluate_alpha(wide_dir_ratio)
print(f"alpha of wide dataset: {wide_alpha}")

# train a model for vehicle dataset to evaluate Shapley value
vehicle_model = XGBClassifier(n_estimators=50, max_depth=6, n_jobs=16, learning_rate=0.1, objective="multi:softmax", tree_method="gpu_hist")
vehicle_model.fit(vehicle_X, vehicle_y, verbose=True)

# evaluate importance of vehicle dataset
imp_evaluator_vehicle = ImportanceEvaluator(sample_rate=0.001)
imp_vehicle_scores = imp_evaluator_vehicle.evaluate(vehicle_Xs, vehicle_model.predict, max_evals=1500)
print(f"Importance of vehicle dataset: {imp_vehicle_scores}")


# estimate the alpha of vehicle dataset
vehicle_alpha = corr_splitter_vehicle.evaluate_alpha(imp_vehicle_scores)
print(f"alpha of vehicle dataset: {vehicle_alpha}")


# train a model for satellite dataset to evaluate Shapley value
satellite_model = XGBClassifier(n_estimators=50, max_depth=6, n_jobs=16, learning_rate=0.1, objective="multi:softmax", tree_method="gpu_hist")
satellite_X_train = np.concatenate(satellite_Xs_sample_flatten_less_features, axis=1)
satellite_y_train = satellite_data.local_datasets[0].y
satellite_model.fit(satellite_X_train, satellite_y_train, verbose=True)

# evaluate importance of satellite dataset
imp_evaluator_satellite = ImportanceEvaluator(sample_rate=0.01)
imp_satellite_scores = imp_evaluator_satellite.evaluate(satellite_Xs_sample_flatten_less_features, satellite_model.predict, max_evals=3500)
print(f"Importance of satellite dataset: {imp_satellite_scores}")

# estimate the alpha of satellite dataset
satellite_alpha = corr_splitter_satellite.evaluate_alpha(imp_satellite_scores)
print(f"alpha of satellite dataset: {satellite_alpha}")


# evaluate Icor for shuffled wide dataset
wide_shuffle_indices = np.random.permutation(wide_X.shape[1])
wide_X_shuffle = wide_X[:, wide_shuffle_indices]
wide_feature_cut_points = np.cumsum([X.shape[1] for X in wide_Xs])
wide_Xs_shuffle = np.split(wide_X_shuffle, wide_feature_cut_points, axis=1)[:-1]

assert np.all([X1.shape == X2.shape for X1, X2 in zip(wide_Xs, wide_Xs_shuffle)])

corr_evaluator_shuffle_wide = CorrelationEvaluator(gpu_id=0)
icor_shuffle_wide = corr_evaluator_shuffle_wide.fit_evaluate(wide_Xs_shuffle)
print(f"icor for shuffled wide dataset: {icor_shuffle_wide}")

# evaluate beta for shuffled wide dataset (no need to refit splitter since features are not changed)
beta_shuffle_wide = corr_splitter_wide.evaluate_beta(icor_shuffle_wide)
print(f"beta for shuffled wide dataset: {beta_shuffle_wide}")

# evaluate Icor for shuffled vehicle dataset
vehicle_shuffle_indices = np.random.permutation(vehicle_X.shape[1])
vehicle_X_shuffle = vehicle_X[:, vehicle_shuffle_indices]
vehicle_feature_cut_points = np.cumsum([X.shape[1] for X in vehicle_Xs])
vehicle_Xs_shuffle = np.split(vehicle_X_shuffle, vehicle_feature_cut_points, axis=1)[:-1]

assert np.all([X1.shape == X2.shape for X1, X2 in zip(vehicle_Xs, vehicle_Xs_shuffle)])

corr_evaluator_shuffle_vehicle = CorrelationEvaluator(gpu_id=0)
icor_shuffle_vehicle = corr_evaluator_shuffle_vehicle.fit_evaluate(vehicle_Xs_shuffle)
print(f"icor for shuffled vehicle dataset: {icor_shuffle_vehicle}")

# evaluate beta for shuffled vehicle dataset (no need to refit splitter since features are not changed)
beta_shuffle_vehicle = corr_splitter_vehicle.evaluate_beta(icor_shuffle_vehicle)
print(f"beta for shuffled vehicle dataset: {beta_shuffle_vehicle}")

# evaluate Icor for shuffled satellite dataset
satellite_X_sample_flatten_less_features = np.concatenate(satellite_Xs_sample_flatten_less_features, axis=1)
satellite_shuffle_indices = np.random.permutation(satellite_X_sample_flatten_less_features.shape[1])
satellite_X_sample_flatten_shuffle = satellite_X_sample_flatten_less_features[:, satellite_shuffle_indices]
satellite_feature_cut_points = np.cumsum([X.shape[1] for X in satellite_Xs_sample_flatten_less_features])
satellite_Xs_sample_flatten_shuffle = np.split(satellite_X_sample_flatten_shuffle, satellite_feature_cut_points, axis=1)[:-1]

assert np.all([X1.shape == X2.shape for X1, X2 in zip(satellite_Xs_sample_flatten_less_features, satellite_Xs_sample_flatten_shuffle)])

corr_evaluator_shuffle_satellite = CorrelationEvaluator(gpu_id=0)
icor_shuffle_satellite = corr_evaluator_shuffle_satellite.fit_evaluate(satellite_Xs_sample_flatten_shuffle)
print(f"icor for shuffled satellite dataset: {icor_shuffle_satellite}")

# evaluate beta for shuffled satellite dataset (no need to refit splitter since features are not changed)
beta_shuffle_satellite = corr_splitter_satellite.evaluate_beta(icor_shuffle_satellite)
print(f"beta for shuffled satellite dataset: {beta_shuffle_satellite}")


# evaluate importance for shuffled wide dataset
wide_shuffle_model = XGBClassifier(n_estimators=50, max_depth=6, n_jobs=16, learning_rate=0.1, objective="binary:logistic", tree_method="gpu_hist")
wide_shuffle_model.fit(wide_X_shuffle, train_wide_dataset.local_datasets[0].y, verbose=True)

# evaluate importance of wide dataset
imp_evaluator_wide_shuffle = ImportanceEvaluator(sample_rate=0.01)
imp_shuffle_wide_scores = imp_evaluator_wide.evaluate(wide_Xs_shuffle, wide_shuffle_model.predict, max_evals=1500)
print(f"Importance of wide dataset: {imp_shuffle_wide_scores}")

# evaluate alpha for shuffled wide dataset
alpha_shuffle_wide = corr_splitter_wide.evaluate_alpha(imp_shuffle_wide_scores)
print(f"alpha for shuffled wide dataset: {alpha_shuffle_wide}")

# evaluate importance for shuffled vehicle dataset
vehicle_shuffle_model = XGBClassifier(n_estimators=50, max_depth=6, n_jobs=16, learning_rate=0.1, objective="multi:softmax", tree_method="gpu_hist")
vehicle_shuffle_model.fit(vehicle_X_shuffle, vehicle_y, verbose=True)

# evaluate importance of vehicle dataset
imp_evaluator_vehicle_shuffle = ImportanceEvaluator(sample_rate=0.001)
imp_shuffle_vehicle_scores = imp_evaluator_vehicle_shuffle.evaluate(vehicle_Xs_shuffle, vehicle_shuffle_model.predict, max_evals=1500)
print(f"Importance of vehicle dataset: {imp_shuffle_vehicle_scores}")


# evaluate alpha for shuffled vehicle dataset
alpha_shuffle_vehicle = corr_splitter_vehicle.evaluate_alpha(imp_shuffle_vehicle_scores)
print(f"alpha for shuffled vehicle dataset: {alpha_shuffle_vehicle}")

# evaluate importance for shuffled satellite dataset
satellite_shuffle_model = XGBClassifier(n_estimators=50, max_depth=6, n_jobs=16, learning_rate=0.1, objective="multi:softmax", tree_method="gpu_hist")
satellite_X_train_shuffle = satellite_X_train[:, satellite_shuffle_indices]
satellite_shuffle_model.fit(satellite_X_train_shuffle, satellite_y_train, verbose=True)

# evaluate importance of satellite dataset
imp_evaluator_satellite_shuffle = ImportanceEvaluator(sample_rate=0.01)
imp_shuffle_satellite_scores = imp_evaluator_satellite_shuffle.evaluate(satellite_Xs_sample_flatten_shuffle, satellite_shuffle_model.predict, max_evals=1500)
print(f"Importance of satellite dataset: {imp_shuffle_satellite_scores}")

# evaluate alpha for shuffled satellite dataset
alpha_shuffle_satellite = corr_splitter_satellite.evaluate_alpha(imp_shuffle_satellite_scores)
print(f"alpha for shuffled satellite dataset: {alpha_shuffle_satellite}")

print("Alpha, beta for real datasets:")
print(f"Wide dataset: {wide_alpha}, {beta_wide}")
print(f"Shuffled wide dataset: {alpha_shuffle_wide}, {beta_shuffle_wide}")
print(f"Vehicle dataset: {vehicle_alpha}, {beta_vehicle}")
print(f"Shuffled vehicle dataset: {alpha_shuffle_vehicle}, {beta_shuffle_vehicle}")
print(f"Satellite dataset: {satellite_alpha}, {beta_satellite}")
print(f"Shuffled satellite dataset: {alpha_shuffle_satellite}, {beta_shuffle_satellite}")

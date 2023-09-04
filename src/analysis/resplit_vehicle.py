import os
import sys

import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.FeatureSplitter import CorrelationSplitter
from preprocess.FeatureEvaluator import CorrelationEvaluator


# Load Vehicle dataset
vehicle_X, vehicle_y = load_svmlight_file("data/real/vehicle/processed/vehicle.libsvm")
vehicle_X = vehicle_X.toarray()
vehicle_y = vehicle_y.astype('int') - 1
vehicle_Xs = [vehicle_X[:, :50], vehicle_X[:, 50:]]

print(vehicle_Xs[0].shape)
print(vehicle_Xs[1].shape)


# Split Vehicle dataset
vehicle_splitter = CorrelationSplitter(2, CorrelationEvaluator(gpu_id=1), seed=0, gpu_id=1)
vehicle_split_Xs = vehicle_splitter.fit_splitXs(vehicle_X, verbose=True, beta=0.0,
                                                n_elites=500, n_offsprings=1000, n_mutants=200, n_gen=100, bias=0.8)

# Evaluate Vehicle dataset
vehicle_split_evaluator = CorrelationEvaluator(gpu_id=1)
icor_split = vehicle_split_evaluator.fit_evaluate(vehicle_split_Xs)
vehicle_split_evaluator.visualize(save_path="fig/vehicle_split.png", value=icor_split, fontsize=24)

vehicle_ori_evaluator = CorrelationEvaluator(gpu_id=1)
icor_ori = vehicle_ori_evaluator.fit_evaluate(vehicle_Xs)
vehicle_ori_evaluator.visualize(save_path="fig/vehicle_ori.png", value=icor_ori, fontsize=24)

shuffle_idx = np.random.permutation(vehicle_X.shape[1])
sort_shuffle_idx = np.concatenate([np.sort(shuffle_idx[:50]), np.sort(shuffle_idx[50:])])
vehicle_shuffle_X = vehicle_X[:, shuffle_idx]
vehicle_shuffle_Xs = [vehicle_shuffle_X[:, :50], vehicle_shuffle_X[:, 50:]]
vehicle_shuffle_evaluator = CorrelationEvaluator(gpu_id=1)
icor_shuffle = vehicle_shuffle_evaluator.fit_evaluate(vehicle_shuffle_Xs)
vehicle_shuffle_evaluator.visualize(save_path="fig/vehicle_shuffle.png", value=icor_shuffle, fontsize=24)




import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.FeatureEvaluator import CorrelationEvaluator, ImportanceEvaluator
from dataset.LocalDataset import LocalDataset
from dataset.VFLDataset import VFLAlignedDataset

if __name__ == '__main__':
    dataset = VFLAlignedDataset.from_pickle(dir="data/syn/higgs", dataset="higgs", n_parties=4,
                                            splitter='corr', weight=1, beta=1, seed=0, type='train')
    dataset.visualize_corr(gpu_id=0)
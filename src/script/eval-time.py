import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocess.vertical_split import split_vertical_data
from dataset.GlobalDataset import GlobalDataset

if __name__ == '__main__':

    dataset_base_path = "data/syn/"
    for seed in range(5):
        # covtype msd gisette realsim letter epsilon radar
        for dataset in ['covtype', 'msd', 'gisette', 'realsim', 'letter', 'epsilon', 'radar']:
            fmt = 'libsvm' if dataset != 'radar' else 'csv'
            dataset_path = os.path.join(dataset_base_path, dataset, '.', fmt)
            X, y = GlobalDataset.from_file(dataset_path).data

            for weights in [0.1, 1, 10, 100]:
                start = time.time()
                split_vertical_data(X, num_parties=4,
                                         splitter='imp',
                                         weights=weights,
                                         beta=0,
                                         seed=seed,
                                         gpu_id=0,
                                         n_jobs=1,
                                         verbose=False)
                end = time.time()


            for beta in [0.1, 0.3, 0.6, 1.0]:
                start = time.time()
                split_vertical_data(X, num_parties=4,
                                         splitter='corr',
                                         weights=None,
                                         beta=beta,
                                         seed=seed,
                                         gpu_id=0,
                                         n_jobs=1,
                                         verbose=False)
                end = time.time()

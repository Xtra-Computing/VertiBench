from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import numpy as np

if __name__ == '__main__':
    dataset = "gisette"
    X, y = load_svmlight_file(f'data/syn/{dataset}/{dataset}.libsvm')

    dump_svmlight_file(X, y, f'data/syn/{dataset}/{dataset}.libsvm')
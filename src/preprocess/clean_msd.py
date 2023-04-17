from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import numpy as np

if __name__ == '__main__':
    X, y = load_svmlight_file('data/syn/msd/msd.libsvm')
    scaler = MinMaxScaler((0, 1))
    X = scaler.fit_transform(X.toarray())
    y_scaler = MinMaxScaler((0, 1))
    y = y_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

    dump_svmlight_file(X, y, 'data/syn/msd/msd.libsvm')

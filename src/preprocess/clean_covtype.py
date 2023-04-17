from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import numpy as np

if __name__ == '__main__':
    X, y = load_svmlight_file('data/syn/covtype/covtype.libsvm')
    scaler = MinMaxScaler((0, 1))
    X = scaler.fit_transform(X.toarray())

    dump_svmlight_file(X, y, 'data/syn/covtype/covtype.libsvm')
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import numpy as np

if __name__ == '__main__':
    X, y = load_svmlight_file('data/syn/letter/letter.libsvm')
    y = y - 1
    assert 0 <= np.min(y) and np.max(y) <= 25

    dump_svmlight_file(X, y, 'data/syn/letter/letter.libsvm')

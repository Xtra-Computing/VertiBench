from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('data/syn/higgs/higgs.csv')
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    data_fmt = np.concatenate([X, y.reshape(-1, 1)], axis=1)

    # save data_fmt to csv without header
    pd.DataFrame(data_fmt).to_csv('data/syn/higgs/higgs.csv', header=False, index=False)
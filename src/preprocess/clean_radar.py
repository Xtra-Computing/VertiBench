from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

if __name__ == '__main__':
    data_df = pd.read_csv('data/syn/radar/WinnipegDataset.txt')
    X = data_df.iloc[:, 1:].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    y = data_df.iloc[:, 0].values
    y -= 1  # scale y from 1-7 to 0-6
    assert 0 <= np.min(y) and np.max(y) <= 6

    save_data_df = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
    save_data_df.to_csv('data/syn/radar/radar.csv', index=False, header=False)

import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset


class GlobalDataset(Dataset):
    def __init__(self, X, y=None):
        """
        Parameters
        -------------------
        X: np.ndarray
            the dataset to be split. The last column should be the label.
        y: np.ndarray
            the label vector
        """
        self.X = X
        self.y = y

        self.check_shape()

    def check_shape(self):
        if self.y is not None:
            assert self.X.shape[0] == self.y.shape[0], "The number of samples in X and y should be the same"

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx] if self.y is not None else None
        return X, y

    @classmethod
    def from_csv(cls, csv_path, header=None):
        """
        Load dataset from csv file. The last column is the label, and the rest
        columns are features.
        Parameters
        ----------
        csv_path: str
            path to csv file
        header: int or list of ints, default None
            row number(s) to use as the column names, and the start of the data. Same as the header in pandas.read_csv()
        """
        data = pd.read_csv(csv_path, header=header).values
        X, y = data[:, :-1], data[:, -1]
        return cls(X, y)

    @classmethod
    def from_libsvm(cls, libsvm_path):
        """
        Load dataset from libsvm file.
        Parameters
        ----------
        libsvm_path: str
            path to libsvm file
        """
        X, y = load_svmlight_file(libsvm_path)
        X = X.toarray()
        return cls(X, y)

    @classmethod
    def from_file(cls, path):
        """
        Load dataset from file according to the file extension.
        Parameters
        ----------
        path: str
            path to the file
        """
        if path.endswith('.csv'):
            return cls.from_csv(path)
        elif path.endswith('.libsvm'):
            return cls.from_libsvm(path)
        else:
            raise ValueError('Unknown file extension')

    @property
    def data(self):
        return self.X, self.y

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)



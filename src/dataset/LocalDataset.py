import abc
from typing import Protocol
import pickle

import numpy as np
import pandas
import pandas as pd

import torch
from torch.utils.data import Dataset


class LocalDataset(Dataset):
    """
    Base class for local datasets
    """

    def __init__(self, X, y=None, key=None):
        """
        Required parameters:
        :param X: features (array)

        Optional parameters:
        :param key: key of the ID (array)
        :param y: labels (1d array)
        """
        self.key = key
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32) if y is not None else None

        self.check_shape()

        # if key is not provided, use the index as key
        if self.key is None:
            self.key = np.arange(self.X.shape[0])

    @torch.no_grad()
    def check_shape(self):
        if self.y is not None:
            assert self.X.shape[0] == self.y.shape[0], "The number of samples in X and y should be the same"
        if self.key is not None:
            assert self.X.shape[0] == self.key.shape[0], "The number of samples in X and key should be the same"

    def __len__(self):
        return len(self.key)

    def __getitem__(self, idx):
        """
        :param idx: the index of the item
        :return: key[idx], X[idx], y[idx]
        """
        X = self.X[idx]
        key = self.key[idx] if self.key is not None else None
        y = self.y[idx] if self.y is not None else None
        return key, X, y

    @property
    def data(self):
        return self.key, self.X, self.y

    @classmethod
    def from_csv(cls, csv_path, header=None, key_cols=1):
        """
        Load dataset from csv file. The key_cols columns are keys, the last column is the label, and the rest
        columns are features.
        :param csv_path: path to csv file
        :param header: row number(s) to use as the column names, and the start of the data.
                       Same as the header in pandas.read_csv()
        :param key_cols: Int. Number of key columns.
        """
        df = pd.read_csv(csv_path, header=header)
        assert df.shape[1] > key_cols + 1, "The number of columns should be larger than key_cols + 1"
        key = df.iloc[:, :key_cols].values
        X = df.iloc[:, key_cols:-1].values
        y = df.iloc[:, -1].values
        return cls(key, X, y)

    @classmethod
    def from_pickle(cls, pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

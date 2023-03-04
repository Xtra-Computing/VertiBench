import abc
from typing import Protocol

import numpy
import pandas
import pandas as pd

import torch
from torch.utils.data import Dataset


@torch.no_grad()
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
        self.X = X
        self.y = y

        self.check_shape()

    @staticmethod
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
        :return: key[idx], X[idx], y[idx]    if y is not None and key is not None
                 key[idx], X[idx]            if y is None     and key is not None
                 X[idx], y[idx]              if y is not None and key is None
                 X[idx]                      if y is None     and key is None
        """
        X = self.X[idx]
        key = self.key[idx] if self.key is not None else None
        y = self.y[idx] if self.y is not None else None
        return key, X, y

    @classmethod
    @torch.no_grad()
    def from_csv(cls, csv_path, drop_header=False, key_cols=1):
        """
        Load dataset from csv file. The key_cols columns are keys, the last column is the label, and the rest
        columns are features.
        :param csv_path: path to csv file
        :param drop_header: whether to drop header
        :param key_cols: Int. Number of key columns.
        """
        df = pd.read_csv(csv_path, header=None if drop_header else 0)
        assert df.shape[1] > key_cols + 1, "The number of columns should be larger than key_cols + 1"
        key = torch.tensor(df.iloc[:, :key_cols].values)
        X = torch.tensor(df.iloc[:, key_cols:-1].values)
        y = torch.tensor(df.iloc[:, -1].values)
        return cls(key, X, y)
        

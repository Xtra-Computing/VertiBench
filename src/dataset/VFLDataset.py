import abc
from typing import Protocol

import numpy
import pandas
import pandas as pd

import torch
from torch.utils.data import Dataset

from LocalDataset import LocalDataset


class VFLDataset:
    """
    Base class for vertical federated learning (VFL) datasets. The __len__ and __getitem__ methods are
    undefined because the length of local datasets and the way to get items might be different. They should be
    defined in the subclass.
    """

    def __init__(self, num_parties: int, local_datasets: torch.Tensor, primary_party_id: int = 0):
        """
        :param num_parties: number of parties
        :param local_datasets: local datasets of each party (torch.Tensor)
        :param primary_party_id: primary party (the party with labels) id, should be in range of [0, num_parties)
        """

        self.num_parties = num_parties
        self.local_datasets = local_datasets
        self.primary_party_id = primary_party_id

        self.check_param()

    @torch.no_grad()
    def check_param(self):
        """
        Check if the data is valid
        """
        assert self.local_datasets.shape[0] == self.num_parties, "The number of parties should be the same as the " \
                                                                 "number of local datasets "
        for local_dataset in self.local_datasets:
            assert isinstance(local_dataset, LocalDataset), "local_dataset should be an instance of LocalDataset"

        assert 0 <= self.primary_party_id < self.num_parties, "primary_party_id should be in range of [0, num_parties)"


class VFLRawDataset(VFLDataset):
    """
    Linkable VFL dataset where the keys of local datasets are the same. It does not define the __len__ and
    __getitem__ methods, thus it cannot be directly used in torch.utils.data.DataLoader to train models.
    """
    def __init__(self, num_parties: int, local_datasets: torch.Tensor, primary_party_id: int = 0):
        """
        :param num_parties: number of parties
        :param local_datasets: (torch.Tensor) local datasets of each party, each element is a LocalDataset
        :param primary_party_id: primary party (the party with labels) id, should be in range of [0, num_parties)
        """
        super().__init__(num_parties, local_datasets, primary_party_id)
        self.check_key()

    @torch.no_grad()
    def check_key(self):
        """
        Check if the keys of local datasets are the same
        """
        key = self.local_datasets[0].key
        for local_dataset in self.local_datasets:
            assert torch.equal(key, local_dataset.key), "The keys of local datasets should be the same"


class VFLAlignedDataset(VFLDataset, Dataset):
    """
    Trainable VFL dataset where the number of samples in local datasets is the same. It defines the __len__ and
    __getitem__ methods, thus it can be directly used in torch.utils.data.DataLoader to train models.
    """
    def __init__(self, num_parties: int, local_datasets: torch.Tensor, primary_party_id: int = 0):
        """
        :param num_parties: number of parties
        :param local_datasets: (torch.Tensor) local datasets of each party, each element is a LocalDataset. Note that
                                this CANNOT be changed to a list because a PyTorch multiprocessing issue (see
                                https://github.com/pytorch/pytorch/issues/13246)
        :param primary_party_id: primary party (the party with labels) id, should be in range of [0, num_parties)
        """
        super().__init__(num_parties, local_datasets, primary_party_id)

    @torch.no_grad()
    def check_shape(self):
        """
        Check if the shape of local datasets is aligned
        """
        for local_dataset in self.local_datasets:
            assert len(local_dataset) == len(self.local_datasets[self.primary_party_id]), \
                "The number of samples in local datasets should be the same"

    def __len__(self):
        return len(self.local_datasets[self.primary_party_id])

    def __getitem__(self, idx):
        """
        Invoke __getitem__ of each local dataset to get the item
        :param idx: the index of the item
        :return: a tuple of tensors. The last tensor is a tensor of y. The rest tensors are tensors of X.
        """
        Xs = []
        for local_dataset in self.local_datasets:
            _, X, _ = local_dataset[idx]        # key is omitted because it is not used in training
            Xs.append(X)
        _, _, y = self.local_datasets[self.primary_party_id][idx]
        assert y is not None, f"The primary party {self.primary_party_id} does not have labels"
        return tuple(Xs + [y])

import warnings

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, Normalizer
from scipy.sparse import csr_matrix

import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, KMNIST
from torchvision.transforms import transforms

from typing import List
import os.path
import pickle
import zipfile

import pandas as pd

from .utils import get_closest_factor

from .exceptions import *


np.random.seed(0)


def vertical_split(x, num_parties):
    if isinstance(x, csr_matrix):
        x = x.todense()

    num_features = x.shape[1]
    xs = []
    for i in range(num_parties):
        if i == num_parties - 1:
            x_train_party_i = x[:, i * num_features // num_parties:]
        else:
            x_train_party_i = x[:, i * num_features // num_parties: (i + 1) * num_features // num_parties]
        xs.append(x_train_party_i)
    return xs


def bias_vertical_split_ratio(x_train, x_test, num_good_features, good_feature_ratio_alpha):
    # split data into biased two parts
    if isinstance(x_train, csr_matrix):
        x_train = x_train.todense()
    if isinstance(x_train, csr_matrix):
        x_test = x_test.todense()

    num_features = x_train.shape[1]
    assert num_good_features <= num_features // 2, "Too many good features"
    split_index = int(num_good_features * good_feature_ratio_alpha)
    good_feature_range = np.arange(num_features - num_good_features, num_features)
    np.random.shuffle(good_feature_range)
    a_good_feature_range = good_feature_range[:split_index]
    b_good_feature_range = good_feature_range[split_index:]
    other_feature_range = np.arange(0, num_features - num_good_features)
    np.random.shuffle(other_feature_range)
    a_other_feature_range = other_feature_range[:num_features // 2 - a_good_feature_range.size]
    b_other_feature_range = other_feature_range[num_features // 2 - a_good_feature_range.size:]
    a_feature_range = np.concatenate([a_good_feature_range, a_other_feature_range])
    b_feature_range = np.concatenate([b_good_feature_range, b_other_feature_range])
    assert np.abs(a_feature_range.size - b_feature_range.size) <= 1
    a_train = x_train[:, a_feature_range]
    b_train = x_train[:, b_feature_range]
    a_test = x_test[:, a_feature_range]
    b_test = x_test[:, b_feature_range]
    return [a_train, b_train], [a_test, b_test]

def bias_vertical_split(x, beta):
    # split data into biased two parts
    if isinstance(x, csr_matrix):
        x = x.todense()

    num_features = x.shape[1]
    split_index = int(num_features * beta)
    a = x[:, :split_index]
    b = x[:, split_index:]

    return [a, b]


def vertical_split_image(x, num_parties):
    if len(x.shape) != 4:
        print("Wrong format of image, got {}".format(str(x.shape)))
        raise UnsupportedFormatError

    m, n = x.shape[1], x.shape[2]
    a, b = get_closest_factor(num_parties)  # if num_parties is 9, then m=3, n=3
    if a != b:
        warnings.warn("num_parties is recommended to be perfect square numbers. a={}, b={}".format(a, b))
    if m % a != 0 or n % b != 0:
        warnings.warn("The image size for each party may be not equal. m={}, n={}, a={}, b={}".format(m,n,a,b))
    xs = []
    for i in range(a):
        for j in range(b):
            if i != m - 1 and j != n - 1:
                x_i_j = x[:, i * m // a: (i + 1) * m // a, j * n // b: (j + 1) * n // b, :]
            elif i == m - 1 and j != n - 1:
                x_i_j = x[:, i * m // a:, j * n // b: (j + 1) * n // b, :]
            elif i != m - 1 and j == n - 1:
                x_i_j = x[:, i * m // a: (i + 1) * m // a, j * n // b:, :]
            else:
                x_i_j = x[:, i * m // a:, j * n // b:, :]
            xs.append(x_i_j)
    return xs


def load_data_train_test(train_file_name, test_file_name=None, num_parties=1, test_size=0.2, root="data/",
                         file_type='libsvm', X=None, y=None):
    print("Data splitting")
    if file_type == 'libsvm':
        x_train, y_train = load_svmlight_file(root + train_file_name)
        x_train = x_train.todense()
        normalizer = Normalizer().fit(x_train)
        x_scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaler.fit(x_train)
        x_train = normalizer.transform(x_scaler.transform(x_train))
    elif file_type == 'csv':
        dataset = np.loadtxt(root + train_file_name, delimiter=',', skiprows=1)
        x_train = dataset[:, :-1]
        y_train = dataset[:, -1].reshape(-1)
        normalizer = Normalizer().fit(x_train)
        x_scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaler.fit(x_train)
        x_train = normalizer.transform(x_scaler.transform(x_train))
    elif file_type == 'torch':
        if train_file_name == 'cifar10':
            # trainset will be normalized later in vertical_fl.py after data argumentation
            dataset = CIFAR10(root=root, train=True, transform=None, download=True)
            x_train, y_train = dataset.data, np.array(dataset.targets)
        elif train_file_name == 'mnist':
            dataset = MNIST(root=root, train=True, transform=None, download=True)
            x_train, y_train = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
        elif train_file_name == 'fashion_mnist':
            dataset = FashionMNIST(root=root, train=True, transform=None, download=True)
            x_train, y_train = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
        elif train_file_name == 'kmnist':
            dataset = KMNIST(root=root, train=True, transform=None, download=True)
            x_train, y_train = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
        else:
            raise UnsupportedFormatError
    elif file_type == 'numpy':
        print("Loading existing numpy array")
        assert X is not None and y is not None and test_file_name is None
        x_train, y_train = X, y
        normalizer = Normalizer().fit(x_train)
        x_scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaler.fit(x_train)
        x_train = normalizer.transform(x_scaler.transform(x_train))
    else:
        raise UnsupportedFormatError

    if test_file_name:
        if file_type == 'libsvm':
            x_test, y_test = load_svmlight_file(root + test_file_name)
            x_test = x_test.todense()
            x_test = normalizer.transform(x_scaler.transform(x_test))
        elif file_type == 'csv':
            dataset = np.loadtxt(root + test_file_name, delimiter=',', skiprows=1)
            x_test = dataset[:, :-1]
            y_test = dataset[:, -1].reshape(-1)
            x_test = normalizer.transform(x_scaler.transform(x_test))
        elif file_type == 'torch':
            if test_file_name == 'cifar10':
                dataset = CIFAR10(root=root, train=False, transform=None, download=True)
                x_test, y_test = dataset.data, np.array(dataset.targets)
            elif test_file_name == 'mnist':
                dataset = MNIST(root=root, train=False, transform=None, download=True)
                x_test, y_test = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
            elif test_file_name == 'fashion_mnist':
                dataset = FashionMNIST(root=root, train=False, transform=None, download=True)
                x_test, y_test = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
            elif test_file_name == 'kmnist':
                dataset = KMNIST(root=root, train=False, transform=None, download=True)
                x_test, y_test = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
            else:
                raise UnsupportedFormatError
        else:
            raise UnsupportedFormatError
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=0)

    # scale labels if regression or binary classification
    if file_type in ['libsvm', 'csv', 'numpy']:
        scalar = MinMaxScaler(feature_range=(0, 1))
        scalar.fit(y_train.reshape(-1, 1))
        y_train = scalar.transform(y_train.reshape(-1, 1)).reshape(-1)
        y_test = scalar.transform(y_test.reshape(-1, 1)).reshape(-1)
    elif file_type in ['torch']:
        pass
    else:
        raise UnsupportedFormatError

    # split data into parties
    if file_type in ['libsvm', 'csv', 'numpy']:
        xs_train = vertical_split(x_train, num_parties)
        xs_test = vertical_split(x_test, num_parties)
    elif file_type in ['torch']:
        xs_train = vertical_split_image(x_train, num_parties)
        xs_test = vertical_split_image(x_test, num_parties)
    else:
        raise UnsupportedFormatError
    print("Finished loading data")

    return xs_train, y_train, xs_test, y_test, x_train, x_test


def load_data_cross_validation(file_name, num_parties=1, root="data/", file_type='libsvm', n_fold=5,
                               use_cache=False, cache_path=None, csv_skiprows=1, feature_order=None,
                               shift_alpha=None, feature_ratio_beta=None,
                               num_good_features=None, good_feature_ratio_alpha=None,
                               x_scaler_wrapper=None, y_scaler_wrapper=None, x_normalizer_wrapper=None):
    if use_cache and os.path.isfile(cache_path):
        print("Loading data from cache: " + cache_path)
        with open(cache_path, 'rb') as f:
            results = pickle.load(f)
            return results

    print("Loading data from file")
    if file_type == 'libsvm':
        x, y = load_svmlight_file(root + file_name)
        x = x.todense()
        if x_normalizer_wrapper is not None and len(x_normalizer_wrapper) == 1:
            normalizer = x_normalizer_wrapper[0]
        else:
            normalizer = Normalizer().fit(x)
            if x_normalizer_wrapper is not None:
                x_normalizer_wrapper.append(normalizer)
        if x_scaler_wrapper is not None and len(x_scaler_wrapper) == 1:
            x_scaler = x_scaler_wrapper[0]
        else:
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            x_scaler.fit(x)
            if x_scaler_wrapper is not None:
                x_scaler_wrapper.append(x_scaler)
        x = normalizer.transform(x_scaler.transform(x))
    elif file_type == 'csv':
        try:
            dataset = np.loadtxt(root + file_name, delimiter=',', skiprows=csv_skiprows)
        except ValueError:
            dataset = np.genfromtxt(root + file_name, delimiter=',', skip_header=csv_skiprows, filling_values=0)
        x = dataset[:, :-1]
        y = dataset[:, -1].reshape(-1)
        if x_normalizer_wrapper is not None and len(x_normalizer_wrapper) == 1:
            normalizer = x_normalizer_wrapper[0]
        else:
            normalizer = Normalizer().fit(x)
            if x_normalizer_wrapper is not None:
                x_normalizer_wrapper.append(normalizer)
        if x_scaler_wrapper is not None and len(x_scaler_wrapper) == 1:
            x_scaler = x_scaler_wrapper[0]
        else:
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            x_scaler.fit(x)
            if x_scaler_wrapper is not None:
                x_scaler_wrapper.append(x_scaler)
        x = normalizer.transform(x_scaler.transform(x))

    elif file_type == 'torch':
        if file_name == 'cifar10':
            # trainset will be normalized later in vertical_fl.py after data argumentation
            dataset = CIFAR10(root=root, train=True, transform=None, download=True)
            x, y = dataset.data, np.array(dataset.targets)
        elif file_name == 'mnist':
            dataset = MNIST(root=root, train=True, transform=None, download=True)
            x, y = dataset.data.detach().numpy()[:, :, :, None], np.array(dataset.targets)
        else:
            raise UnsupportedFormatError
    else:
        raise UnsupportedFormatError

    print("Scaling labels")
    # scale labels if regression or binary classification
    if file_type in ['libsvm', 'csv']:
        if y_scaler_wrapper is not None and len(y_scaler_wrapper) == 1:
            scalar = y_scaler_wrapper[0]
        else:
            scalar = MinMaxScaler(feature_range=(0, 1))
            scalar.fit(y.reshape(-1, 1))
            if y_scaler_wrapper is not None:
                y_scaler_wrapper.append(scalar)
        y = scalar.transform(y.reshape(-1, 1)).reshape(-1)
    elif file_type in ['torch']:
        pass
    else:
        raise UnsupportedFormatError

    if feature_order is not None:
        assert feature_order.size == x.shape[1], "Feature orders mismatch the number of features"
        x = x[:, feature_order]

    if shift_alpha is not None:
        shift = -int(x.shape[1] * shift_alpha)
        x = np.roll(x, shift, axis=1)

    print("{} fold spliltting".format(n_fold))
    results = []
    if n_fold > 1:
        k_fold = KFold(n_splits=n_fold, shuffle=True, random_state=0)
        for i, (train_idx, test_idx) in enumerate(k_fold.split(x, y)):
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_test = x[test_idx]
            y_test = y[test_idx]

            # split data into parties
            if file_type in ['libsvm', 'csv']:
                if feature_ratio_beta is not None:
                    assert num_parties == 2
                    xs_train = bias_vertical_split(x_train, feature_ratio_beta)
                    xs_test = bias_vertical_split(x_test, feature_ratio_beta)
                elif good_feature_ratio_alpha is not None and num_good_features is not None:
                    assert num_parties == 2
                    xs_train, xs_test = bias_vertical_split_ratio(
                        x_train, x_test, num_good_features, good_feature_ratio_alpha)
                else:
                    xs_train = vertical_split(x_train, num_parties)
                    xs_test = vertical_split(x_test, num_parties)
            elif file_type in ['torch']:
                xs_train = vertical_split_image(x_train, num_parties)
                xs_test = vertical_split_image(x_test, num_parties)
            else:
                raise UnsupportedFormatError

            results.append([xs_train, y_train, xs_test, y_test])
            print("Fold {} finished".format(i))
    else:       # fold = 1
        # split data into parties
        if file_type in ['libsvm', 'csv']:
            if feature_ratio_beta is not None:
                assert num_parties == 2
                xs = bias_vertical_split(x, feature_ratio_beta)
            elif good_feature_ratio_alpha is not None and num_good_features is not None:
                assert False, "Unsupported"
                # assert num_parties == 2
                # xs_train, xs_test = bias_vertical_split_ratio(
                #     x, x, num_good_features, good_feature_ratio_alpha)
            else:
                xs = vertical_split(x, num_parties)
        elif file_type in ['torch']:
            xs = vertical_split_image(x, num_parties)
        else:
            raise UnsupportedFormatError
        results.append([xs, y])

    if use_cache and cache_path is not None:
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
        print("Saved results to cache: " + str(cache_path))

    return results


def move_item_to_end_(arr, items):
    for item in items:
        arr.insert(len(arr), arr.pop(arr.index(item)))


def move_item_to_start_(arr, items):
    for item in items[::-1]:
        arr.insert(0, arr.pop(arr.index(item)))

class NYBikeTaxiLoader:
    def __init__(self, bike_path, taxi_path=None, link=False):
        print("Loading bike from {}".format(bike_path))
        self.bike_data = pd.read_pickle(bike_path)
        # self.bike_data = self.bike_data.head(10000)
        # print("Remove N/A from bike")
        # self.bike_data.dropna()
        print("Loaded.")
        if taxi_path is not None:
            print("Loading taxi from {}".format(taxi_path))
            self.taxi_data = pd.read_pickle(taxi_path)
            print("Loaded.")

        if link:
            self.labels = self.bike_data['tripduration'].to_numpy()
            self.bike_data.drop(columns=['tripduration'], inplace=True)

            # move lon and lat to end of airbnb
            bike_cols = list(self.bike_data)
            move_item_to_end_(bike_cols, ['start_lon', 'start_lat', 'end_lon', 'end_lat',
                                          'start_hour', 'end_hour'])
            self.bike_data = self.bike_data[bike_cols]
            self.bike_data.drop(columns=['start_hour', 'end_hour'], inplace=True)
            print("Current bike columns: " + str(list(self.bike_data)))
            self.bike_data = self.bike_data.to_numpy()

            # move lon and lat to the front of taxi
            tx_cols = list(self.taxi_data)
            move_item_to_start_(tx_cols, ['start_lon', 'start_lat', 'end_lon', 'end_lat',
                                          'start_hour', 'end_hour'])
            self.taxi_data = self.taxi_data[tx_cols]
            self.taxi_data.drop(columns=['start_hour', 'end_hour'], inplace=True)
            print("Current taxi columns: " + str(list(self.taxi_data)))
            self.taxi_data = self.taxi_data.to_numpy()
        else:
            print("Remove columns that are used for linkage")
            self.bike_data.drop(columns=['start_lon', 'start_lat', 'end_lon', 'end_lat',
                                         'start_hour', 'end_hour'], inplace=True)
            print('Extract labels')
            self.labels = self.bike_data['tripduration'].to_numpy()
            print("Extract data")
            self.bike_data = self.bike_data.drop(columns=['tripduration']).to_numpy()

    def load_single(self):
        return self.bike_data, self.labels

    def load_parties(self):
        return [self.bike_data, self.taxi_data], self.labels


def load_bike(train_file_name, num_parties=1, test_size=0.2, root="data/"):
    loader = NYBikeTaxiLoader(os.path.join(root, train_file_name))
    X, y = loader.load_single()
    print("Shape of X: {}".format(X.shape))
    return load_data_train_test(train_file_name, num_parties=num_parties, test_size=test_size, root=root,
                                file_type='numpy', X=X, y=y)



class LocalDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            data_i = self.transform(self.data[index])
        else:
            data_i = self.data[index]
        return index, data_i, self.target[index]

    def __len__(self):
        return self.target.shape[0]

    def update_targets(self, indices: List[int], new_targets):
        self.target[indices, :] = new_targets


class AggDataset(Dataset):
    def __init__(self, X, Z, y, transform=None):
        self.X = X      # X can be images
        self.Z = Z
        self.transform = transform
        self.y = y

    def __getitem__(self, index):
        if self.transform is not None:
            X_i = self.transform(self.X[index])
        else:
            X_i = self.X[index]
        return index, X_i, self.Z[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]

    def update_Z(self, indices: List[int], new_Z):
        self.Z[indices] = new_Z


class ImageDataset(Dataset):
    def __init__(self, xs: list, y=None, transform=None):
        if y is not None:
            for x in xs:
                assert x.shape[0] == y.shape[0]
        self.xs = xs
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        result = []
        for x in self.xs:
            x_i = x[idx]
            if self.transform:
                x_i = self.transform(x_i)
            result.append(x_i)

        if self.y is not None:
            result.append(self.y[idx])

        return result

    def __len__(self):
        return self.xs[0].shape[0]



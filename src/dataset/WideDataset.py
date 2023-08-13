import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.LocalDataset import LocalDataset
from dataset.VFLDataset import VFLAlignedDataset, VFLSynAlignedDataset


class WideGlobalDataset():
    def __init__(self, num_parties: int = 6, local_datasets=None, primary_party_id: int = 0):
        self.num_parties = num_parties
        self.primary_party_id = primary_party_id
        self.local_datasets = local_datasets
        self.local_train_datasets = None
        self.local_test_datasets = None

    @classmethod
    def from_source(cls, image_dir, label_dir, num_parties=5, label_type='sky', primary_party_id=0):
        """
        Load data from original source of NUS-WIDE dataset.

        Parameters
        ----------
        image_dir : str
            The directory of low-level images. The directory should contain multiple .dat files.
        tag_dir : str
            The directory of tags. The directory should contain multiple .dat files.
        label_dir : str
            The directory of labels. The directory should contain multiple .dat files.
        num_parties : int, optional
            The number of parties, by default 6
        label_type : str, optional
            The type of the label, by default 'airport'. See NUS-WIDE website for more details.
            :param primary_party_id:
        """
        img_file_ids = ['CH', 'CORR', 'EDH', 'WT', 'CM55']
        train_img_features = []
        test_img_features = []
        for img_file_id in img_file_ids:
            print(f'Loading {img_file_id}...')
            train_data_path = os.path.join(image_dir, f"Train_Normalized_{img_file_id}.dat")
            test_data_path = os.path.join(image_dir, f"Test_Normalized_{img_file_id}.dat")
            train_data = pd.read_csv(train_data_path, sep=' ', header=None)
            test_data = pd.read_csv(test_data_path, sep=' ', header=None)
            train_img_features.append(train_data.values[:, :-1])   # the last column is the nan
            test_img_features.append(test_data.values[:, :-1])     # the last column is the nan

        # print('Loading tags...')
        # train_tag_path = os.path.join(tag_dir, 'Train_Tags1k.dat')
        # test_tag_path = os.path.join(tag_dir, 'Test_Tags1k.dat')
        # train_tag_features = pd.read_csv(train_tag_path, sep='\t', header=None).values
        # test_tag_features = pd.read_csv(test_tag_path, sep='\t', header=None).values

        print('Loading labels...')
        train_label_path = os.path.join(label_dir, f"TrainTestLabels/Labels_{label_type}_Train.txt")
        test_label_path = os.path.join(label_dir, f"TrainTestLabels/Labels_{label_type}_Test.txt")
        train_labels = pd.read_csv(train_label_path, sep=' ', header=None).values.flatten()
        test_labels = pd.read_csv(test_label_path, sep=' ', header=None).values.flatten()

        # # remove rows if any of the train_img_features/test_img_features/train_labels/test_labels contains nan
        # train_nan_indices = set()
        # test_nan_indices = set()
        # for i in range(len(train_img_features)):
        #     train_nan_indices |= set(np.argwhere(np.isnan(train_img_features[i])).flatten().tolist())
        #     test_nan_indices |= set(np.argwhere(np.isnan(test_img_features[i])).flatten().tolist())
        # train_nan_indices |= set(np.argwhere(np.isnan(train_labels)).flatten().tolist())
        # test_nan_indices |= set(np.argwhere(np.isnan(test_labels)).flatten().tolist())
        # train_nan_indices = np.array(list(train_nan_indices))
        # test_nan_indices = np.array(list(test_nan_indices))
        # print(f"Removing {len(train_nan_indices)} rows from train set and {len(test_nan_indices)} rows from test set")
        # train_img_features = [np.delete(train_img_features[i], train_nan_indices, axis=0)
        #                       for i in range(len(train_img_features))]
        # test_img_features = [np.delete(test_img_features[i], test_nan_indices, axis=0)
        #                         for i in range(len(test_img_features))]
        # train_labels = np.delete(train_labels, train_nan_indices, axis=0)
        # test_labels = np.delete(test_labels, test_nan_indices, axis=0)

        # create image features and tag features to LocalDataset
        local_train_datasets = []
        local_test_datasets = []
        for i in range(len(train_img_features)):
            local_train_datasets.append(LocalDataset(train_img_features[i], train_labels))
            local_test_datasets.append(LocalDataset(test_img_features[i], test_labels))
        # local_train_datasets.append(LocalDataset(train_tag_features, train_labels))
        # local_test_datasets.append(LocalDataset(test_tag_features, test_labels))

        obj = cls(num_parties, None, primary_party_id)
        obj.local_train_datasets = local_train_datasets
        obj.local_test_datasets = local_test_datasets

        return obj

    @property
    def train(self):
        """
        Get a copy of the training dataset.
        :return: A WideDataset object.
        """
        return WideDataset(self.num_parties, self.local_train_datasets, self.primary_party_id)

    @property
    def test(self):
        """
        Get a copy of the testing dataset.
        :return: A WideDataset object.
        """
        return WideDataset(self.num_parties, self.local_test_datasets, self.primary_party_id)

    def to_pickle(self, folder):
        """
        Save the dataset to pickle files.

        Parameters
        ----------
        folder : str
            The folder to save the pickle files.
        """
        os.makedirs(folder, exist_ok=True)

        for i in range(self.num_parties):
            train_path = os.path.join(folder, f'wide_party{self.num_parties}-{i}_train.pkl')
            test_path = os.path.join(folder, f'wide_party{self.num_parties}-{i}_test.pkl')
            self.local_train_datasets[i].to_pickle(train_path)
            print(f"Saved {self.local_train_datasets[i].X.shape}, {self.local_train_datasets[i].y.shape} to {train_path}")
            self.local_test_datasets[i].to_pickle(test_path)
            print(f"Saved {self.local_test_datasets[i].X.shape}, {self.local_test_datasets[i].y.shape} to {test_path}")


class WideDataset(VFLSynAlignedDataset):
    def __init__(self, num_parties: int = 5, local_datasets=None, primary_party_id: int = 0):
        super().__init__(num_parties, local_datasets, primary_party_id)


if __name__ == '__main__':
    image_path = "data/real/nus-wide/images"
    tag_path = "data/real/nus-wide/tags"
    label_path = "data/real/nus-wide/labels"

    train_test_dataset = WideGlobalDataset.from_source(image_path, label_path)

    train_test_dataset.to_pickle('data/real/nus-wide/processed')

    # try loading the dataset
    # train_dataset = WideDataset.from_pickle("data/real/nus-wide/processed/", 'wide', 5, 0, splitter='simple', type='train')
    # test_dataset = WideDataset.from_pickle("data/real/nus-wide/processed/", 'wide', 5, 0, splitter='simple', type='test')
    pass

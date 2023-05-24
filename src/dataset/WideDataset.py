import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .LocalDataset import LocalDataset
from .VFLDataset import VFLAlignedDataset


class WideDataset(VFLAlignedDataset):
    def __init__(self, num_parties: int = 6, local_datasets=None, primary_party_id: int = 0):
        super().__init__(num_parties, local_datasets, primary_party_id)
        self.local_train_datasets = None
        self.local_test_datasets = None

    @classmethod
    def from_source(cls, image_dir, tag_dir, label_dir, num_parties=6, label_type='sky', primary_party_id=0):
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
            train_img_features.append(train_data.values)
            test_img_features.append(test_data.values)

        print('Loading tags...')
        train_tag_path = os.path.join(tag_dir, 'Train_Tags1k.dat')
        test_tag_path = os.path.join(tag_dir, 'Test_Tags1k.dat')
        train_tag_features = pd.read_csv(train_tag_path, sep='\t', header=None).values
        test_tag_features = pd.read_csv(test_tag_path, sep='\t', header=None).values

        print('Loading labels...')
        train_label_path = os.path.join(label_dir, f"TrainTestLabels/Labels_{label_type}_Train.txt")
        test_label_path = os.path.join(label_dir, f"TrainTestLabels/Labels_{label_type}_Test.txt")
        train_labels = pd.read_csv(train_label_path, sep=' ', header=None).values
        test_labels = pd.read_csv(test_label_path, sep=' ', header=None).values


        # create image features and tag features to LocalDataset
        local_train_datasets = []
        local_test_datasets = []
        for i in range(len(train_img_features)):
            local_train_datasets.append(LocalDataset(train_img_features[i], train_labels))
            local_test_datasets.append(LocalDataset(test_img_features[i], test_labels))
        local_train_datasets.append(LocalDataset(train_tag_features, train_labels))
        local_test_datasets.append(LocalDataset(test_tag_features, test_labels))

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



# if __name__ == '__main__':
#     image_path = "data/real/nus-wide/uncompressed/images"
#     tag_path = "data/real/nus-wide/uncompressed/tags"
#     label_path = "data/real/nus-wide/uncompressed/labels"
#
#     train_test_dataset = WideDataset.from_source(image_path, tag_path, label_path)
#     train_dataset: WideDataset = train_test_dataset.train
#     test_dataset: WideDataset = train_test_dataset.test
#
#     print(train_dataset.local_datasets[0].X.shape)

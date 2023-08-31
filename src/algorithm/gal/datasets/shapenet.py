import anytree
import codecs
import numpy as np
import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index, find_classes


class ShapeNet55(Dataset):
    data_name = 'ShapeNet55'
    file = [('http://maxwell.cs.umass.edu/mvcnn-data/shapenet55v1.tar', None)]

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'))

    def __getitem__(self, index):
        data = [self.transform({'data': Image.open(self.data[index][i]).convert('RGB')})['data'] for i in
                range(len(self.data[index]))]
        id, data, target = torch.tensor(self.id[index]), torch.stack(data, dim=-1), torch.tensor(self.target[index])
        input = {'id': id, 'data': data, 'target': target}
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def load_data(self, path):
        return self.transform({'data': Image.open(path).convert('RGB')})['data']

    def make_data(self):
        train_df = pd.read_csv(os.path.join(self.raw_folder, 'shapenet55v1', 'train.csv'))
        test_df = pd.read_csv(os.path.join(self.raw_folder, 'shapenet55v1', 'val.csv'))
        train_files = os.listdir(os.path.join(self.raw_folder, 'shapenet55v1', 'train'))
        test_files = os.listdir(os.path.join(self.raw_folder, 'shapenet55v1', 'val'))
        train_files_id = pd.DataFrame(list(set([int(x.split('_')[1]) for x in train_files])), columns=['id'])
        test_files_id = pd.DataFrame(list(set([int(x.split('_')[1]) for x in test_files])), columns=['id'])
        train_df = train_df.merge(train_files_id, left_on='id', right_on='id')
        test_df = test_df.merge(test_files_id, left_on='id', right_on='id')
        classes, train_target = np.unique(train_df['synsetId'].to_numpy(), return_inverse=True)
        train_target = train_target.astype(np.int64)
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        test_target = np.vectorize(classes_to_labels.get)(test_df['synsetId']).astype(np.int64)
        train_data, test_data = [], []
        makedir_exist_ok(os.path.join(self.raw_folder, 'shapenet55v1', 'train_32x32'))
        makedir_exist_ok(os.path.join(self.raw_folder, 'shapenet55v1', 'val_32x32'))
        transform = transforms.Resize((32, 32))
        for i in tqdm(range(len(train_df['id']))):
            name = 'model_{0:06}'.format(train_df['id'][i])
            views = []
            for i in range(1, 13):
                view = '{0:03}'.format(i)
                org_path_i = os.path.join(self.raw_folder, 'shapenet55v1', 'train', '{}_{}.jpg'.format(name, view))
                transform_path_i = os.path.join(self.raw_folder, 'shapenet55v1', 'train_32x32',
                                                '{}_{}.jpg'.format(name, view))
                data_i = Image.open(org_path_i).convert('RGB')
                data_i = transform(data_i)
                data_i.save(transform_path_i)
                views.append(transform_path_i)
            train_data.append(views)
        for i in tqdm(range(len(test_df['id']))):
            name = 'model_{0:06}'.format(test_df['id'][i])
            views = []
            for i in range(1, 13):
                view = '{0:03}'.format(i)
                org_path_i = os.path.join(self.raw_folder, 'shapenet55v1', 'val', '{}_{}.jpg'.format(name, view))
                transform_path_i = os.path.join(self.raw_folder, 'shapenet55v1', 'val_32x32',
                                                '{}_{}.jpg'.format(name, view))
                data_i = Image.open(org_path_i).convert('RGB')
                data_i = transform(data_i)
                data_i.save(transform_path_i)
                views.append(transform_path_i)
            test_data.append(views)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = anytree.Node('U', index=[])
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)

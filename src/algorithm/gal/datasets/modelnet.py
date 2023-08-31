import anytree
import codecs
import numpy as np
import os
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index, find_classes


class ModelNet40(Dataset):
    data_name = 'ModelNet40'
    file = [('http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v1.tar', None)]

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

    def make_data(self):
        classes_to_labels = find_classes(os.path.join(self.raw_folder, 'modelnet40v1'))
        classes = list(classes_to_labels.keys())
        train_data, test_data = [], []
        train_target, test_target = [], []
        transform = transforms.Resize((32, 32))
        for target in classes:
            makedir_exist_ok(os.path.join(self.raw_folder, 'modelnet40v1', target, 'train_32x32'))
            filenames = os.listdir(os.path.join(self.raw_folder, 'modelnet40v1', target, 'train'))
            filename_set = set()
            for i in tqdm(range(len(filenames))):
                filename = filenames[i]
                filename_list = filename.split('_')
                name = '_'.join(filename_list[:-1])
                if name not in filename_set:
                    filename_set.add(name)
                    views = []
                    for i in range(1, 13):
                        view = '{0:03}'.format(i)
                        org_path_i = os.path.join(self.raw_folder, 'modelnet40v1', target, 'train',
                                                  '{}_{}.jpg'.format(name, view))
                        transform_path_i = os.path.join(self.raw_folder, 'modelnet40v1', target, 'train_32x32',
                                                        '{}_{}.jpg'.format(name, view))
                        data_i = Image.open(org_path_i).convert('RGB')
                        data_i = transform(data_i)
                        data_i.save(transform_path_i)
                        views.append(transform_path_i)
                    train_data.append(views)
                    train_target.append(classes_to_labels[target])
            makedir_exist_ok(os.path.join(self.raw_folder, 'modelnet40v1', target, 'test_32x32'))
            filenames = os.listdir(os.path.join(self.raw_folder, 'modelnet40v1', target, 'test'))
            filename_set = set()
            for i in tqdm(range(len(filenames))):
                filename = filenames[i]
                filename_list = filename.split('_')
                name = '_'.join(filename_list[:-1])
                if name not in filename_set:
                    filename_set.add(name)
                    views = []
                    for i in range(1, 13):
                        view = '{0:03}'.format(i)
                        org_path_i = os.path.join(self.raw_folder, 'modelnet40v1', target, 'test',
                                                  '{}_{}.jpg'.format(name, view))
                        transform_path_i = os.path.join(self.raw_folder, 'modelnet40v1', target, 'test_32x32',
                                                        '{}_{}.jpg'.format(name, view))
                        data_i = Image.open(org_path_i).convert('RGB')
                        data_i = transform(data_i)
                        data_i.save(transform_path_i)
                        views.append(transform_path_i)
                    test_data.append(views)
                    test_target.append(classes_to_labels[target])
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = anytree.Node('U', index=[])
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)

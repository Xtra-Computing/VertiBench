import numpy as np
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tqdm import tqdm


class MIMICL(Dataset):
    data_name = 'MIMICL'

    def __init__(self, root, split):
        self.root = os.path.expanduser(root)
        self.split = split
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.target, self.length = load(os.path.join(self.processed_folder,
                                                                         '{}.pt'.format(self.split)))
        self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'))

    def __getitem__(self, index):
        id, data, target, length = torch.tensor(self.id[index]), torch.tensor(self.data[index]), \
                                   torch.tensor(self.target[index]), torch.tensor(self.length[index])
        input = {'id': id, 'data': data, 'target': target, 'length': length}
        return input

    def __len__(self):
        return len(self.id)

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
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(self.__class__.__name__, self.__len__(), self.root,
                                                                     self.split)
        return fmt_str

    def make_data(self):
        train_files = os.listdir(os.path.join(self.raw_folder, 'length-of-stay', 'train'))
        test_files = os.listdir(os.path.join(self.raw_folder, 'length-of-stay', 'test'))
        train_id = []
        train_df = []
        for i in tqdm(range(len(train_files))):
            train_df_i = pd.read_csv(os.path.join(self.raw_folder, 'length-of-stay', 'train', train_files[i]))
            train_id.append(np.repeat(i, train_df_i.shape[0], axis=0).astype(np.int64))
            train_df_i['Glascow coma scale eye opening'].replace(
                {'None': np.nan, '1 No Response': 1, 'To pain': 2, 'To Pain': 2, '2 To pain': 2, 'To Speech': 3,
                 '3 To speech': 3, 'Spontaneously': 4, '4 Spontaneously': 4}, inplace=True)
            train_df_i['Glascow coma scale motor response'].replace(
                {'No response': 1, '1 No Response': 1, 'Abnormal extension': 2, '2 Abnorm extensn': 2,
                 'Abnormal Flexion': 3, '3 Abnorm flexion': 3, 'Flex-withdraws': 4, '4 Flex-withdraws': 4,
                 'Localizes Pain': 5, '5 Localizes Pain': 5, 'Obeys Commands': 6, '6 Obeys Commands': 6}, inplace=True)
            train_df_i['Glascow coma scale verbal response'].replace(
                {'No Response': 1, '1 No Response': 1, '1.0 ET/Trach': 1, 'No Response-ETT': 1,
                 'Incomprehensible sounds': 2, '2 Incomp sounds': 2, 'Inappropriate Words': 3, '3 Inapprop words': 3,
                 'Confused': 4, '4 Confused': 4, 'Oriented': 5, '5 Oriented': 5}, inplace=True)
            train_df_i['ICD9_CODE'] = train_df_i['ICD9_CODE'].astype(str)
            train_df.append(train_df_i)
        train_id = np.concatenate(train_id, axis=0)
        train_df = pd.concat(train_df).reset_index(drop=True)
        train_df = train_df.set_index(train_id)
        numerical_idx = np.arange(len(train_df.columns) - 1)[train_df.iloc[:, :-1].dtypes == 'float64']
        categorical_idx = np.arange(len(train_df.columns) - 1)[train_df.iloc[:, :-1].dtypes == 'object']
        numerical_pipeline = Pipeline(
            [('normalizer', StandardScaler()), ('imputer', SimpleImputer(strategy='constant', fill_value=0))])
        categorical_pipeline = Pipeline(
            [('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
             ('imputer', SimpleImputer(strategy='constant', fill_value=-1))])
        numerical_ct = ColumnTransformer([('numerical', numerical_pipeline, numerical_idx)])
        numerical_ct.fit(train_df.iloc[:, :-1])
        categorical_ct = ColumnTransformer([('categorical', categorical_pipeline, categorical_idx)])
        categorical_ct.fit(train_df.iloc[:, :-1])
        train_df.iloc[:, numerical_idx] = numerical_ct.transform(train_df.iloc[:, :-1])
        train_df.iloc[:, categorical_idx] = categorical_ct.transform(train_df.iloc[:, :-1])
        train_df.iloc[:, categorical_idx] = train_df.iloc[:, categorical_idx] + 1
        train_df = train_df.astype(np.float32)
        test_id = []
        test_df = []
        for i in tqdm(range(len(test_files))):
            test_df_i = pd.read_csv(os.path.join(self.raw_folder, 'length-of-stay', 'test', test_files[i]))
            test_id.append(np.repeat(i, test_df_i.shape[0], axis=0).astype(np.int64))
            test_df_i['Glascow coma scale eye opening'].replace(
                {'None': np.nan, '1 No Response': 1, 'To pain': 2, 'To Pain': 2, '2 To pain': 2, 'To Speech': 3,
                 '3 To speech': 3, 'Spontaneously': 4, '4 Spontaneously': 4}, inplace=True)
            test_df_i['Glascow coma scale motor response'].replace(
                {'No response': 1, '1 No Response': 1, 'Abnormal extension': 2, '2 Abnorm extensn': 2,
                 'Abnormal Flexion': 3, '3 Abnorm flexion': 3, 'Flex-withdraws': 4, '4 Flex-withdraws': 4,
                 'Localizes Pain': 5, '5 Localizes Pain': 5, 'Obeys Commands': 6, '6 Obeys Commands': 6}, inplace=True)
            test_df_i['Glascow coma scale verbal response'].replace(
                {'No Response': 1, '1 No Response': 1, '1.0 ET/Trach': 1, 'No Response-ETT': 1,
                 'Incomprehensible sounds': 2, '2 Incomp sounds': 2, 'Inappropriate Words': 3, '3 Inapprop words': 3,
                 'Confused': 4, '4 Confused': 4, 'Oriented': 5, '5 Oriented': 5}, inplace=True)
            test_df_i['ICD9_CODE'] = test_df_i['ICD9_CODE'].astype(str)
            test_df.append(test_df_i)
        test_id = np.concatenate(test_id, axis=0)
        test_df = pd.concat(test_df).reset_index(drop=True)
        test_df = test_df.set_index(test_id)
        test_df.iloc[:, numerical_idx] = numerical_ct.transform(test_df.iloc[:, :-1])
        test_df.iloc[:, categorical_idx] = categorical_ct.transform(test_df.iloc[:, :-1])
        test_df.iloc[:, categorical_idx] = test_df.iloc[:, categorical_idx] + 1
        test_df = test_df.astype(np.float32)
        train_id, test_id = np.unique(train_id), np.unique(test_id)
        train_data, train_target = train_df.iloc[:, :-1], train_df.iloc[:, [-1]]
        test_data, test_target = test_df.iloc[:, :-1], test_df.iloc[:, [-1]]
        train_data = [train_data.loc[[train_id[i]]].to_numpy() for i in range(len(train_id))]
        train_target = [train_target.loc[[train_id[i]]].to_numpy() for i in range(len(train_id))]
        train_length = [len(train_data[i]) for i in range(len(train_id))]
        test_data = [test_data.loc[[test_id[i]]].to_numpy() for i in range(len(test_id))]
        test_target = [test_target.loc[[test_id[i]]].to_numpy() for i in range(len(test_id))]
        test_length = [len(test_data[i]) for i in range(len(test_id))]
        target_size = 1
        return (train_id, train_data, train_target, train_length), (test_id, test_data, test_target,
                                                                    test_length), target_size


class MIMICM(Dataset):
    data_name = 'MIMICM'

    def __init__(self, root, split):
        self.root = os.path.expanduser(root)
        self.split = split
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.target, self.length = load(os.path.join(self.processed_folder,
                                                                         '{}.pt'.format(self.split)))
        self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'))

    def __getitem__(self, index):
        id, data, target, length = torch.tensor(self.id[index]), torch.tensor(self.data[index]), \
                                   torch.tensor(self.target[index]), torch.tensor(self.length[index])
        input = {'id': id, 'data': data, 'target': target, 'length': length}
        return input

    def __len__(self):
        return len(self.id)

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
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(self.__class__.__name__, self.__len__(), self.root,
                                                                     self.split)
        return fmt_str

    def make_data(self):
        train_files = os.listdir(os.path.join(self.raw_folder, 'in-hospital-mortality', 'train'))
        test_files = os.listdir(os.path.join(self.raw_folder, 'in-hospital-mortality', 'test'))
        train_id = []
        train_df = []
        for i in tqdm(range(len(train_files))):
            train_df_i = pd.read_csv(os.path.join(self.raw_folder, 'in-hospital-mortality', 'train', train_files[i]))
            train_id.append(np.repeat(i, train_df_i.shape[0], axis=0).astype(np.int64))
            train_df_i['Glascow coma scale eye opening'].replace(
                {'None': np.nan, '1 No Response': 1, 'To pain': 2, 'To Pain': 2, '2 To pain': 2, 'To Speech': 3,
                 '3 To speech': 3, 'Spontaneously': 4, '4 Spontaneously': 4}, inplace=True)
            train_df_i['Glascow coma scale motor response'].replace(
                {'No response': 1, '1 No Response': 1, 'Abnormal extension': 2, '2 Abnorm extensn': 2,
                 'Abnormal Flexion': 3, '3 Abnorm flexion': 3, 'Flex-withdraws': 4, '4 Flex-withdraws': 4,
                 'Localizes Pain': 5, '5 Localizes Pain': 5, 'Obeys Commands': 6, '6 Obeys Commands': 6}, inplace=True)
            train_df_i['Glascow coma scale verbal response'].replace(
                {'No Response': 1, '1 No Response': 1, '1.0 ET/Trach': 1, 'No Response-ETT': 1,
                 'Incomprehensible sounds': 2, '2 Incomp sounds': 2, 'Inappropriate Words': 3, '3 Inapprop words': 3,
                 'Confused': 4, '4 Confused': 4, 'Oriented': 5, '5 Oriented': 5}, inplace=True)
            train_df_i['ICD9_CODE'] = train_df_i['ICD9_CODE'].astype(str)
            train_df.append(train_df_i)
        train_id = np.concatenate(train_id, axis=0)
        train_df = pd.concat(train_df).reset_index(drop=True)
        train_df = train_df.set_index(train_id)
        numerical_idx = np.arange(len(train_df.columns) - 1)[train_df.iloc[:, :-1].dtypes == 'float64']
        categorical_idx = np.arange(len(train_df.columns) - 1)[train_df.iloc[:, :-1].dtypes == 'object']
        numerical_pipeline = Pipeline(
            [('normalizer', StandardScaler()), ('imputer', SimpleImputer(strategy='constant', fill_value=0))])
        categorical_pipeline = Pipeline(
            [('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
             ('imputer', SimpleImputer(strategy='constant', fill_value=-1))])
        numerical_ct = ColumnTransformer([('numerical', numerical_pipeline, numerical_idx)])
        numerical_ct.fit(train_df.iloc[:, :-1])
        categorical_ct = ColumnTransformer([('categorical', categorical_pipeline, categorical_idx)])
        categorical_ct.fit(train_df.iloc[:, :-1])
        train_df.iloc[:, numerical_idx] = numerical_ct.transform(train_df.iloc[:, :-1])
        train_df.iloc[:, categorical_idx] = categorical_ct.transform(train_df.iloc[:, :-1])
        train_df.iloc[:, categorical_idx] = train_df.iloc[:, categorical_idx] + 1
        train_df = train_df.astype(np.float32)
        test_id = []
        test_df = []
        for i in tqdm(range(len(test_files))):
            test_df_i = pd.read_csv(os.path.join(self.raw_folder, 'in-hospital-mortality', 'test', test_files[i]))
            test_id.append(np.repeat(i, test_df_i.shape[0], axis=0).astype(np.int64))
            test_df_i['Glascow coma scale eye opening'].replace(
                {'None': np.nan, '1 No Response': 1, 'To pain': 2, 'To Pain': 2, '2 To pain': 2, 'To Speech': 3,
                 '3 To speech': 3, 'Spontaneously': 4, '4 Spontaneously': 4}, inplace=True)
            test_df_i['Glascow coma scale motor response'].replace(
                {'No response': 1, '1 No Response': 1, 'Abnormal extension': 2, '2 Abnorm extensn': 2,
                 'Abnormal Flexion': 3, '3 Abnorm flexion': 3, 'Flex-withdraws': 4, '4 Flex-withdraws': 4,
                 'Localizes Pain': 5, '5 Localizes Pain': 5, 'Obeys Commands': 6, '6 Obeys Commands': 6}, inplace=True)
            test_df_i['Glascow coma scale verbal response'].replace(
                {'No Response': 1, '1 No Response': 1, '1.0 ET/Trach': 1, 'No Response-ETT': 1,
                 'Incomprehensible sounds': 2, '2 Incomp sounds': 2, 'Inappropriate Words': 3, '3 Inapprop words': 3,
                 'Confused': 4, '4 Confused': 4, 'Oriented': 5, '5 Oriented': 5}, inplace=True)
            test_df_i['ICD9_CODE'] = test_df_i['ICD9_CODE'].astype(str)
            test_df.append(test_df_i)
        test_id = np.concatenate(test_id, axis=0)
        test_df = pd.concat(test_df).reset_index(drop=True)
        test_df = test_df.set_index(test_id)
        test_df.iloc[:, numerical_idx] = numerical_ct.transform(test_df.iloc[:, :-1])
        test_df.iloc[:, categorical_idx] = categorical_ct.transform(test_df.iloc[:, :-1])
        test_df.iloc[:, categorical_idx] = test_df.iloc[:, categorical_idx] + 1
        test_df = test_df.astype(np.float32)
        train_id, test_id = np.unique(train_id), np.unique(test_id)
        train_data, train_target = train_df.iloc[:, :-1], train_df.iloc[:, [-1]]
        test_data, test_target = test_df.iloc[:, :-1], test_df.iloc[:, [-1]]
        train_target = train_target.fillna(-65535).astype(np.int64)
        test_target = test_target.fillna(-65535).astype(np.int64)
        train_data = [train_data.loc[[train_id[i]]].to_numpy() for i in range(len(train_id))]
        train_target = [train_target.loc[[train_id[i]]].to_numpy().reshape(-1) for i in range(len(train_id))]
        train_length = [len(train_data[i]) for i in range(len(train_id))]
        test_data = [test_data.loc[[test_id[i]]].to_numpy() for i in range(len(test_id))]
        test_target = [test_target.loc[[test_id[i]]].to_numpy().reshape(-1) for i in range(len(test_id))]
        test_length = [len(test_data[i]) for i in range(len(test_id))]
        target_size = 2
        return (train_id, train_data, train_target, train_length), (test_id, test_data, test_target,
                                                                    test_length), target_size

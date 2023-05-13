import argparse
from concurrent.futures import ThreadPoolExecutor as Executor, as_completed


# add to syspath
import sys
sys.path.append('..')
sys.path.append('./src')

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import mne # pip install mne
import nibabel as nib # pip install nibabel
import pandas as pd
from dataset.VFLDataset import VFLAlignedDataset
from dataset.LocalDataset import LocalDataset

class FMRIGlobalDataset(Dataset):
    def __init__(self, base_path, X=None, y=None, POIs=None):
        self.base_path = base_path # path to 'xxx/xxx/CWL_Data/'
        if X is not None and y is not None:
            self.X = X
            self.y = y
            self.POIs = POIs
            return

        # 8 subjects
        eeg_datalist = [
            os.path.join(self.base_path, 'eeg/in-scan/trio1_mrcorrected_eoec_in-scan_hpump-on.set'),
            os.path.join(self.base_path, 'eeg/in-scan/trio2_mrcorrected_eoec_in-scan_hpump-on.set'),
            os.path.join(self.base_path, 'eeg/in-scan/trio3_mrcorrected_eoec_in-scan_hpump-on.set'),
            os.path.join(self.base_path, 'eeg/in-scan/trio4_mrcorrected_eoec_in-scan_hpump-on.set'),
            os.path.join(self.base_path, 'eeg/in-scan/verio5_mrcorrected_eoec_in-scan_hpump-on.set'),
            os.path.join(self.base_path, 'eeg/in-scan/verio6_mrcorrected_eoec_in-scan_hpump-on.set'),
            os.path.join(self.base_path, 'eeg/in-scan/verio7_mrcorrected_eoec_in-scan_hpump-on.set'),
            os.path.join(self.base_path, 'eeg/in-scan/verio8_mrcorrected_eoec_in-scan_hpump-on.set')
        ]
        
        mri_datalist = [
            os.path.join(self.base_path, 'mri/epi_normalized/rwatrio1_eoec_in-scan_hpump-on.nii'),
            os.path.join(self.base_path, 'mri/epi_normalized/rwatrio2_eoec_in-scan_hpump-on.nii'),
            os.path.join(self.base_path, 'mri/epi_normalized/rwatrio3_eoec_in-scan_hpump-on.nii'),
            os.path.join(self.base_path, 'mri/epi_normalized/rwatrio4_eoec_in-scan_hpump-on.nii'),
            os.path.join(self.base_path, 'mri/epi_normalized/rwaverio5_eoec_in-scan_hpump-on.nii'),
            os.path.join(self.base_path, 'mri/epi_normalized/rwaverio6_eoec_in-scan_hpump-on.nii'),
            os.path.join(self.base_path, 'mri/epi_normalized/rwaverio7_eoec_in-scan_hpump-on.nii'),
            os.path.join(self.base_path, 'mri/epi_normalized/rwaverio8_eoec_in-scan_hpump-on.nii')
        ]

        # Read EEG data
        self.eeg = []
        for data in eeg_datalist:
            raw = mne.io.read_raw_eeglab(data)
            rawdf = raw.to_data_frame()
            anndf = raw.annotations.to_data_frame()
            anndf = anndf[anndf['description'].isin(['eeo', 'eec', 'beo', 'bec', 'mri'])]
            anndf = anndf.rename(columns={'onset': 'time'})
            anndf['time'] = anndf['time'].apply(lambda x: round(x.timestamp() , 3))
            merge = pd.merge(rawdf, anndf, on='time', how='left')
            assert (True == merge[merge['F3'].isna()].empty) # Check if there is any NAN
            self.eeg.append(merge)
            # Annotations are Described in Section 2.4 in the original dataset paper:
            # mri: start of a MRI scan
            # R:   heartbeat marker peaks
            # S 1: begin of flash. (Black/white 'flashes' helped the subject notice a change in condition when the eyes were closed.)
            # beo: begin eyes open
            # eeo: end eyes open
            # bec: begin eyes closed
            # eec: end eyes closed 
        
        # Read fMRI data
        self.fmri = []
        for data in mri_datalist:
            img = nib.load(data)
            header = img.header
            imgdata = img.get_fdata()
            self.fmri.append(imgdata) # (61, 72, 61, 139) <==> (img_width, img_height, brain_slice, echo_time)


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    @torch.no_grad()
    def sample(self, n_samples=None, ratio=None, seed=None, n_jobs=1):
        if seed is not None:
            np.random.seed(seed)
        if n_samples is None:
            assert ratio is not None and 0 < ratio < 1
            n_samples = int(len(self) * ratio)
        indices = np.random.choice(len(self), n_samples, replace=False)


        with Executor(n_jobs) as executor, tqdm(total=n_samples) as pbar:
            futures = [None for _ in range(n_samples)]
            for i, idx in enumerate(indices):
                futures[i] = executor.submit(lambda j, k: (self.__getitem__(k), j), i, idx)

            Xs = torch.zeros((len(self), 16, 13, 158, 158), dtype=torch.uint8)
            # ys = torch.zeros((0, 4, 1054, 1054), dtype=torch.uint8)
            ys = torch.zeros(n_samples, dtype=torch.long)
            for future in as_completed(futures):
                (X, y), i = future.result()
                Xs[i] = X
                ys[i] = y
                pbar.update(1)
        return Xs, ys

    @torch.no_grad()
    def split_train_test(self, test_ratio=0.2, seed=None):
        n_test = int(len(self) * test_ratio)
        train_indices, test_indices = train_test_split(np.arange(len(self)), test_size=n_test, random_state=seed)
        X = np.array(self.X)
        y = np.array(self.y)
        train_dataset = FMRIGlobalDataset(self.base_path, X[train_indices], y[train_indices])
        test_dataset = FMRIGlobalDataset(self.base_path, X[test_indices], y[test_indices])
        return train_dataset, test_dataset


class FMRIDataset(VFLAlignedDataset):
    def __init__(self, local_datasets, num_parties=16):
        super().__init__(num_parties=num_parties, local_datasets=local_datasets)

    @classmethod
    @torch.no_grad()
    def from_global(cls, global_dataset: FMRIGlobalDataset, n_jobs=1):
        local_Xs = [torch.zeros((len(global_dataset), 13, 158, 158), dtype=torch.uint8) for _ in range(16)]

        if n_jobs == 1:
            for i in tqdm(range(len(global_dataset))):
                X_full_i, y_i = global_dataset[i]
                for party_id in range(16):
                    X_i = X_full_i[party_id]
                    local_Xs[party_id][i] = X_i
            local_datasets = [LocalDataset(X=x, y=None) for x in local_Xs]
            local_datasets[0].y = global_dataset.y
        elif n_jobs > 1:
            with Executor(n_jobs) as executor, tqdm(total=len(global_dataset)) as pbar:
                futures = [None for i in range(len(global_dataset))]
                for i in range(len(global_dataset)):
                    futures[i] = executor.submit(lambda idx: (global_dataset.__getitem__(idx), idx), i)

                for future in as_completed(futures):
                    (X_full_i, y_i), i = future.result()
                    for party_id in range(16):
                        X_i = X_full_i[party_id]
                        local_Xs[party_id][i] = X_i
                    pbar.update(1)
            local_datasets = [LocalDataset(X=x, y=global_dataset.y) for x in local_Xs]
        else:
            raise ValueError("n_jobs must be a positive integer")
        return cls(local_datasets=local_datasets)

    def to_pickle(self, folder, type='train'):
        os.makedirs(folder, exist_ok=True)
        for party_id in range(16):
            path = os.path.join(folder, f"FMRI_party{party_id}_{type}.pkl")
            self.local_datasets[party_id].to_pickle(path)
            print(f"Saved {path}")

    @classmethod
    def from_pickle(cls, folder, type='train', n_parties=16, n_jobs=1):
        if n_jobs == 1:
            local_datasets = []
            for party_id in range(n_parties):
                path = os.path.join(folder, f"FMRI_party{party_id}_{type}.pkl")
                local_datasets.append(LocalDataset.from_pickle(path))
                print(f"Loaded {path}")
            return cls(local_datasets=local_datasets, num_parties=n_parties)
        elif n_jobs > 1:
            local_datasets = [None for _ in range(n_parties)]
            with Executor(n_jobs) as executor, tqdm(total=n_parties) as pbar:
                futures = [None for _ in range(n_parties)]
                for party_id in range(n_parties):
                    path = os.path.join(folder, f"FMRI_party{party_id}_{type}.pkl")
                    futures[party_id] = executor.submit(lambda party_id, path: (party_id, LocalDataset.from_pickle(path)), party_id, path)

                for future in as_completed(futures):
                    pid, local_dataset = future.result()
                    local_datasets[pid] = local_dataset
                    pbar.update(1)

                # check None
                for party_id in range(n_parties):
                    if local_datasets[party_id] is None:
                        raise ValueError(f"Error loading party {party_id}")
            return cls(local_datasets=local_datasets, num_parties=n_parties)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_path", type=str, default="/home/junyi/datasets/fmri_eeg/CWL_Data")
    args = parser.parse_args()

    FMRI = FMRIGlobalDataset(args.base_path)
    print(FMRI.eeg[0]) # Subject 0 EEG
    print(FMRI.fmri[0].shape) # Subject 0 fMRI

    # fmri shape (61, 72, 61, 139) <==> (img_width, img_height, brain_slice, echo_time)
    # fmri shape (61, 72, 61, 150) <==> (img_width, img_height, brain_slice, echo_time)
    # fmri shape (61, 72, 61, 142) <==> (img_width, img_height, brain_slice, echo_time)
    # fmri shape (61, 72, 61, 140) <==> (img_width, img_height, brain_slice, echo_time)
    # fmri shape (61, 72, 61, 143) <==> (img_width, img_height, brain_slice, echo_time)
 
    # X, y = FMRI.sample(n_samples=50, n_jobs=5)
    # print(X.shape, y.shape)
    # print(y)
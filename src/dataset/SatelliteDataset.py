import argparse
from concurrent.futures import ThreadPoolExecutor as Executor, as_completed

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import tifffile
import cv2
import json
from tqdm import tqdm

from dataset.VFLDataset import VFLAlignedDataset
from dataset.LocalDataset import LocalDataset

class SatelliteGlobalDataset(Dataset):
    def __init__(self, base_path, X=None, y=None, POIs=None):
        self.base_path = base_path
        if X is not None and y is not None:
            self.X = X
            self.y = y
            self.POIs = POIs
            return

        self.X = []
        self.y = []
        self.POIs = {}  # for future use

        # total 3927 available unique locations.

        # each location have 1 high-res image, and 1 low-res image
        # POI: point of interest
        # AOI: area of interest
        # PAN: PAN stands for panchromatic, and it refers to satellite images that have a single band (i.e., grayscale) with very high spatial resolution, often as low as 0.5 meters per pixel. Panchromatic images are useful for tasks such as feature extraction, where high spatial resolution is more important than color information.
        # PS: PS stands for pan-sharpened, and it refers to satellite images that combine a panchromatic band with one or more multispectral bands (i.e., bands with information about different wavelengths of light, such as red, green, and blue). The multispectral bands provide color information, while the panchromatic band provides high spatial resolution. The combination of the two can produce images that are both high-resolution and high-quality.
        # RGB: RGB stands for red, green, blue, and it refers to satellite images that have three bands corresponding to those colors. RGB images are often used for visual interpretation, as they mimic what our eyes see. They can also be used for tasks such as vegetation analysis and land cover classification.
        # RGBN: RGBN stands for red, green, blue, near-infrared, and it refers to satellite images that have four bands corresponding to those colors (including the near-infrared band). Near-infrared is outside the range of human vision, but it can be useful for tasks such as vegetation analysis and land cover classification, as it provides information about plant health and moisture content.

        dirs = os.listdir(self.base_path)
        label_encode = {'Amnesty POI': 0, 'ASMSpotter': 1, 'Landcover': 2, 'UNHCR': 3}
        label_cnt = [0, 0, 0, 0]
        for aoi in tqdm(dirs):
            if (
                "Amnesty" in aoi or "ASMSpotter" in aoi
            ):  # for the 9 AOIs around each POI
                pos1 = aoi.rfind("-")
                pos2 = aoi.rfind("-", 0, pos1)
                if pos2 == -1:
                    continue
                poi = aoi[:pos2]
            else:
                poi = aoi  # for the 1 AOI around each POI

            highres = f"{aoi}/{aoi}_ps.tiff"  # 1 high-res image
            lowres = [
                f"{aoi}/L1C/{aoi}-{i}-L1C_data.tiff" for i in range(1, 17)
            ]  # 16 low-res images
            self.POIs[poi] = self.POIs.get(poi, []) + [
                {"name": aoi, "highres": highres, "lowres": lowres}
            ]

            try:
                label = label_encode[aoi.split("-")[0]]
                label_cnt[label] += 1
            except KeyError as e:
                print(f"Undefined label: {aoi.split('-')[0]}")
                raise e

            self.X.append(lowres)  # 16 low-res images
            self.y.append(label)  # 1 high-res image

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        # print the ratio of each class
        print(f"Total {len(self.y)} images: 'Amnesty POI': {label_cnt[0]}, 'ASMSpotter': {label_cnt[1]}, "
              f"'Landcover': {label_cnt[2]}, 'UNHCR': {label_cnt[3]}")


    @staticmethod
    def read_tiff(fpath: str) -> list:
        res = []
        img = tifffile.imread(fpath)
        img = (img / img.max() * 255).astype(np.uint8)
        if img.shape[0] < 1000:  # low-res image resize to 158x15
            img = cv2.resize(img, (158, 158), interpolation=cv2.INTER_CUBIC)

        if len(img.shape) == 2:
            res.append(img)
            return res

        for i in range(img.shape[2]):
            res.append(img[:, :, i])  # 1054x1054

        return res

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = []
        for i in range(16):
            X.append(self.read_tiff(os.path.join(self.base_path, self.X[idx][i])))
        # y = self.read_tiff(os.path.join(self.base_path, self.y[idx]))

        X = (
            torch.tensor(np.array(X), dtype=torch.uint8).view(16, 13, 158, 158).detach()
        )  # 16 revisit, each visit has 13 channel 158x158
        # y = (
        #     torch.tensor(np.array(y), dtype=torch.uint8).view(4, 1054, 1054).detach()
        # )  # 4 channel RGBN image 1054x1054
        y = torch.tensor(self.y[idx], dtype=torch.long).detach()
        return X, y

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
        train_dataset = SatelliteGlobalDataset(self.base_path, X[train_indices], y[train_indices])
        test_dataset = SatelliteGlobalDataset(self.base_path, X[test_indices], y[test_indices])
        return train_dataset, test_dataset


class SatelliteDataset(VFLAlignedDataset):
    def __init__(self, local_datasets, num_parties=16):
        super().__init__(num_parties=num_parties, local_datasets=local_datasets)

    @classmethod
    @torch.no_grad()
    def from_global(cls, global_dataset: SatelliteGlobalDataset, n_jobs=1):
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
            path = os.path.join(folder, f"satellite_party{party_id}_{type}.pkl")
            self.local_datasets[party_id].to_pickle(path)
            print(f"Saved {path}")

    @classmethod
    def from_pickle(cls, folder, type='train', n_parties=16, n_jobs=1, primary_party_id=0):
        if n_parties == 1:
            path = os.path.join(folder, f"satellite_party{primary_party_id}_{type}.pkl")
            local_dataset = LocalDataset.from_pickle(path)
            print(f"Loaded {path}")
            return cls(local_datasets=[local_dataset], num_parties=n_parties)

        if n_jobs == 1:
            local_datasets = []
            for party_id in range(n_parties):
                path = os.path.join(folder, f"satellite_party{party_id}_{type}.pkl")
                local_datasets.append(LocalDataset.from_pickle(path))
                print(f"Loaded {path}")
            return cls(local_datasets=local_datasets, num_parties=n_parties)
        elif n_jobs > 1:
            local_datasets = [None for _ in range(n_parties)]
            with Executor(n_jobs) as executor, tqdm(total=n_parties) as pbar:
                futures = [None for _ in range(n_parties)]
                for party_id in range(n_parties):
                    path = os.path.join(folder, f"satellite_party{party_id}_{type}.pkl")
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
    parser.add_argument("-b", "--base_path", type=str, default="data/real/satellite/clean")
    args = parser.parse_args()

    satellite = SatelliteGlobalDataset(args.base_path)
    print("Total", len(satellite), "locations")
    X, y = satellite.sample(n_samples=50, n_jobs=5)
    print(X.shape, y.shape)
    print(y)
    # for i in tqdm(range(0, len(satellite))):
    #     dirs = os.listdir(satellite.base_path)
    #
    #     X, y = satellite[i]
    #     # X.shape == (16, 13, 158, 158). 16 parties, each party has 13 channel 158x158 image
    #     # y.shape == (4, 1054, 1054). 4 channel RGBN image 1054x1054
    #     print(i, X.shape, y.shape)

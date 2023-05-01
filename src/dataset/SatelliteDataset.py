import argparse

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import tifffile
import cv2
import json
from tqdm import tqdm


class SatelliteDataset(Dataset):
    def __init__(self, base_path, X=None, y=None):
        self.base_path = base_path
        self.X = X if X is not None else []
        self.y = y if y is not None else []
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

            self.X.append(lowres)  # 16 low-res images
            self.y.append(highres)  # 1 high-res image

    def read_tiff(self, fpath: str) -> list:
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
        y = self.read_tiff(os.path.join(self.base_path, self.y[idx]))

        X = (
            torch.tensor(np.array(X), dtype=torch.uint8).view(16, 13, 158, 158).detach()
        )  # 16 revisit, each visit has 13 channel 158x158
        y = (
            torch.tensor(np.array(y), dtype=torch.uint8).view(4, 1054, 1054).detach()
        )  # 4 channel RGBN image 1054x1054
        return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_path", type=str, default="data/real/satellite/clean")
    args = parser.parse_args()

    satellite = SatelliteDataset(args.base_path)
    print("Total", len(satellite), "locations")
    for i in tqdm(range(0, len(satellite))):
        dirs = os.listdir(satellite.base_path)

        X, y = satellite[i]
        # X.shape == (16, 13, 158, 158). 16 parties, each party has 13 channel 158x158 image
        # y.shape == (4, 1054, 1054). 4 channel RGBN image 1054x1054
        print(i, X.shape, y.shape) 

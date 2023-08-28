from torch.utils.data import Dataset
import os
import torch

from .base import get_dataset

class MSD(Dataset):
    # 回归任务
    def __init__(self, split, typ: str, val:str, dataseed: str, num_clients: int = 4) -> None:
        super().__init__()
        self.split = split
        # split: train, test
        # typ: corr, imp
        # val: 0.1, 0.3, 0.5, 0.7, 0.9


        self.parties = [None] * num_clients
        self.partitions = [None] * num_clients
        for i in range(num_clients):
            self.parties[i] = get_dataset("msd", i, typ, val, split, dataseed, num_clients)
            self.partitions[i] = self.parties[i].X.shape[1]

        self.key = list(torch.tensor(self.parties[0].key).clone().detach())
        self.y = torch.tensor(self.parties[0].y, dtype=torch.float32).unsqueeze(1).clone().detach()
        self.X = torch.cat([ torch.tensor(i.X).clone().detach() for i in self.parties], dim=1)
        self.classes = torch.unique(self.y)
        print("🟢 MSD dataset loaded.")
        print("🟢 X.shape: ", self.X.shape)
        print("🟢 y.shape: ", self.y.shape)
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return len(self.key)
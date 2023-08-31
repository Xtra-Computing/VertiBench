from torch.utils.data import Dataset
import os
import torch

from .base import get_dataset

class Letter(Dataset):
    # 26分类任务
    def __init__(self, split, typ: str, val:str, dataseed: str, num_clients: int = 4) -> None:
        super().__init__()
        self.split = split
        # split: train, test
        # typ: corr, imp
        # val: 0.1, 0.3, 0.5, 0.7, 0.9


        self.parties = [None] * num_clients
        self.partitions = [None] * num_clients
        for i in range(num_clients):
            self.parties[i] = get_dataset("letter", i, typ, val, split, dataseed, num_clients)
            self.partitions[i] = self.parties[i].X.shape[1]

        self.key = list(torch.tensor(self.parties[0].key).clone().detach())
        self.y = torch.tensor(self.parties[0].y, dtype=torch.int64).clone().detach()
        self.X = torch.cat([ torch.tensor(i.X).clone().detach() for i in self.parties], dim=1)
        self.classes = torch.unique(self.y)
        
        self.target = self.y
        self.target_size = len(torch.unique(self.target)) # 26个类
        
    def __getitem__(self, index):
        input = {'id': self.key[index], 'data': self.X[index], 'target': self.y[index]}
        return input

    def __len__(self) -> int:
        return len(self.key)

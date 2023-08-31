from torch.utils.data import Dataset
import torch

from .base import get_dataset

class CIFAR10_VB(Dataset):
    def __init__(self, split, typ: str, val:str, dataseed: str, num_clients: int = 4) -> None:
        super().__init__()
        self.split = split
        # split: train, test
        # typ: corr, imp
        # val: 0.1, 0.3, 0.5, 0.7, 0.9


        self.parties = [None] * num_clients
        self.partitions = [None] * num_clients
        for i in range(num_clients):
            self.parties[i] = get_dataset("cifar10", i, typ, val, split, dataseed, num_clients)
            self.partitions[i] = 3*32*32 # same as the shape
            self.parties[i].X = torch.tensor(self.parties[i].X / 255.0, dtype=torch.float32)
            self.parties[i].X = self.parties[i].X.reshape((-1, 3, 32, 32))
            print("cifar10 shape", self.parties[i].X.shape)

        self.X = torch.cat([ p.X for p in self.parties], dim=1)
        self.key = list(torch.tensor(self.parties[0].key).clone().detach())
        self.y = torch.tensor(self.parties[0].y, dtype=torch.int64).clone().detach()
        self.target = self.y
        self.target_size = len(torch.unique(self.target))
        
    def __getitem__(self, index):
        input = {'id': self.key[index], 'data': self.X[index], 'target': self.y[index]}
        return input

    def __len__(self) -> int:
        return len(self.key)

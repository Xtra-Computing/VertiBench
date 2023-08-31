from torch.utils.data import Dataset
import torch

from .base import get_dataset

class MNIST_VB(Dataset):
    def __init__(self, split, typ: str, val:str, dataseed: str, num_clients: int = 4) -> None:
        super().__init__()
        self.split = split
        # split: train, test
        # typ: corr, imp
        # val: 0.1, 0.3, 0.5, 0.7, 0.9


        self.parties = [None] * num_clients
        self.partitions = [None] * num_clients
        for i in range(num_clients):
            self.parties[i] = get_dataset("mnist", i, typ, val, split, dataseed, num_clients)
            self.partitions[i] = 1 * 28 * 28 # same as the shape
            self.parties[i].X = torch.tensor(self.parties[i].X / 255.0, dtype=torch.float32)
            self.parties[i].X = self.parties[i].X.reshape((-1, 1, 28, 28)) # (50000, 1, 28, 28)
            print("MNIST shape", self.parties[i].X.shape)

        # assert num_clients == 4, "Only support 4 clients for the image dataset"
        # a = self.parties[0].X
        # b = self.parties[1].X
        # c = self.parties[2].X
        # d = self.parties[3].X
        # abcd = torch.cat([a,b,c,d], dim=1) # concat to (50000,4,28,28)
        # gal 会认为一个 X 是 (4, 28, 28)
        self.X = torch.cat([ p.X for p in self.parties], dim=1)
        # self.X = abcd
        self.key = list(torch.tensor(self.parties[0].key).clone().detach())
        self.y = torch.tensor(self.parties[0].y, dtype=torch.int64).clone().detach()
        self.target = self.y
        self.target_size = len(torch.unique(self.target))
        
    def __getitem__(self, index):
        input = {'id': self.key[index], 'data': self.X[index], 'target': self.y[index]}
        return input

    def __len__(self) -> int:
        return len(self.key)

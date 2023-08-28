from torch.utils.data import Dataset
import torch

from .base import get_dataset_real

# wide 是 2 分类
class Wide(Dataset):
    def __init__(self, split) -> None:
        super().__init__()
        self.split = split
        # split: train, test
        # typ: corr, imp
        # val: 0.1, 0.3, 0.5, 0.7, 0.9
        num_clients = 5 # "NUS-WIDE only have 5 clients"
        print("NOTICE: dataseed for NUS-WIDE is not used")
        assert split in ["train", "test"], "split must be train or test"

        self.parties = [None] * num_clients
        self.partitions = [None] * num_clients
        for i in range(num_clients):
            self.parties[i] = get_dataset_real("wide", i, split)
            self.partitions[i] = self.parties[i].X.shape[1]

        self.key = list(torch.tensor(self.parties[0].key).clone().detach())
        self.y = torch.tensor(self.parties[0].y, dtype=torch.int64).clone().detach()
        self.X = torch.cat([ torch.tensor(i.X).clone().detach() for i in self.parties], dim=1)
        self.classes = torch.unique(self.y) 
        print(f"NUS-WIDE have {self.classes} classes")
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return len(self.key)

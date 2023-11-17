import sys
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
sys.path.append("/home/junyi/VertiBenchGH/src/")

import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from src.utils import PartyPath
from dataset.VFLDataset import VFLAlignedDataset, VFLSynAlignedDataset
from dataset.LocalDataset import LocalDataset

class MNISTGlobalDataset(Dataset):
    pass

class MNISTDataset(VFLSynAlignedDataset):
    def __init__(self, local_datasets, num_parties: int, primary_party_id: int = 0):
        super().__init__(num_parties=num_parties, local_datasets=local_datasets, primary_party_id=primary_party_id)

    @classmethod
    def from_pickle(cls, dir: str, dataset: str, n_parties, primary_party_id: int = 0,
                    splitter: str = 'imp', weight: float = 1, beta: float = 1, seed: int = 0, type='train'):
        """
        Load a VFLAlignedDataset from pickle file. The pickle files are local datasets of each party.

        Parameters
        ----------
        dir : str
            The directory of pickle files.
        dataset : str
            The name of the dataset.
        n_parties : int
            The number of parties.
        primary_party_id : int, optional
            The primary party id, by default 0
        splitter : str, optional
            The splitter used to split the dataset, by default 'imp'
        weight : float, optional
            The weight of the primary party, by default 1
        beta : float, optional
            The beta of the primary party, by default 1
        seed : int, optional
            The seed of the primary party, by default 0
        type : str, optional
            The type of the dataset, by default 'train'. It should be ['train', 'test'].
        """
        assert type in ['train', 'test'], "type should be 'train' or 'test'"
        local_datasets = []
        for party_id in range(n_parties):

            path_in_dir = f"mnist_{type}_party4-{party_id}_{splitter}_"

            if splitter == 'imp':
                path_in_dir += f"weight{weight}_seed{seed}_train.pkl"
            elif splitter == 'corr':
                path_in_dir += f"beta{beta}_seed{seed}_train.pkl"
                
            # path_in_dir = PartyPath(dataset_path=dataset, n_parties=n_parties, party_id=party_id,
                                    # splitter=splitter, weight=weight, beta=beta, seed=seed, fmt='pkl').data(type)
            path = os.path.join(dir, path_in_dir)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} does not exist")
            local_dataset = LocalDataset.from_pickle(path)

            if party_id != primary_party_id:    # remove y of secondary parties
                local_dataset.y = None
            local_datasets.append(local_dataset)
            
            # reshape the 1d data to 3d
            local_dataset.X = local_dataset.X.reshape(-1, 1, 28, 28)
            # transform to 0-1
            local_dataset.X = torch.tensor(local_dataset.X / 255.0, dtype=torch.float32)

        return cls(num_parties=n_parties, local_datasets=local_datasets, primary_party_id=primary_party_id)

if __name__ == "__main__":
    
    print(os.listdir("./"))
    train_data = MNISTDataset.from_pickle(
        f"data/syn/mnist", 'mnist', 4, 0, 'imp', 0.1, 0.0, 0, 'train'
    )

    test_data = MNISTDataset.from_pickle(
        f"data/syn/mnist", 'mnist', 4, 0, 'imp', 0.1, 0.0, 0, 'test'
    )

    print(train_data)
    print(test_data)
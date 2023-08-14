from sklearn.datasets import load_svmlight_file

from dataset.VFLDataset import VFLAlignedDataset, VFLSynAlignedDataset
from dataset.LocalDataset import LocalDataset


class VehicleDataset(VFLSynAlignedDataset):
    def __init__(self, num_parties: int, local_datasets, primary_party_id: int = 0):
        """
        Vertical partitioned vehicle dataset

        Parameters
        ----------
        num_parties : int
            number of parties
        primary_party_id : int, optional
           primary party (the party with labels) id, should be in range of [0, num_parties), by default 0
        """
        super().__init__(num_parties, local_datasets, primary_party_id)

    @classmethod
    def from_libsvm(cls, path: str, n_parties: int, primary_party_id: int = 0):
        """
        Load a VFLAlignedDataset from libsvm file. The libsvm files are local datasets of each party.

        Parameters
        ----------
        path : str
            The path of the global vehicle dataset (one libsvm file).
        n_parties : int
            The number of parties.
        primary_party_id : int, optional
            The primary party id, by default 0
        """
        if n_parties > 2:
            raise ValueError("Vehicle dataset only supports <=2 parties")
        global_X, global_y = load_svmlight_file(path)
        global_Xs = [global_X[:, :global_X.shape[1] // 2], global_X[:, global_X.shape[1] // 2:]]
        local_datasets = []
        for party_id in range(n_parties):
            if party_id == primary_party_id:
                local_y = global_y
            else:
                local_y = None
            local_X = global_Xs[party_id]
            local_datasets.append(LocalDataset(local_X, y=local_y))
        return cls(n_parties, local_datasets, primary_party_id)





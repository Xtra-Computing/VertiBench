from dataset.VFLDataset import VFLDataset, VFLRawDataset, VFLAlignedDataset
from dataset.LocalDataset import LocalDataset


class MyRealLocalDataset1(LocalDataset):
    def __init__(self, key, X, y, **kwargs):    # Dataset of first party
        """
        Local dataset of first party. If the local datasets are 2d tabular data, you can directly
        use the LocalDataset class. If the local datasets are other types, you may need to implement a subclass of
        LocalDataset and redefine the __len__ and __getitem__ methods.

        Parameters
        ----------
        key : numpy.ndarray
            key of the dataset
        X : numpy.ndarray
            features of the dataset
        y : numpy.ndarray
            labels of the dataset
        """
        super().__init__(key, X, y)
        self.check_param(**kwargs)

    def check_param(self, **kwargs):
        pass

    """
    Redefine the __len__ and __getitem__ methods if needed
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
    """


class MyRealLocalDataset2(LocalDataset):        # Dataset of second party
    def __init__(self, key, X, y, **kwargs):
        super().__init__(key, X, y)
        self.check_param(**kwargs)

    def check_param(self, **kwargs):
        pass


class MyRealVFLRawDataset(VFLRawDataset):

    def __init__(self, num_parties: int, local_datasets: list, primary_party_id: int = 0, **kwargs):
        """
        Vertical partitioned real dataset

        Parameters
        ----------
        num_parties : int
            number of parties
        local_datasets : list
            local datasets of each party, each element is a LocalDataset (or subclass of LocalDataset)
        primary_party_id : int, optional
           primary party (the party with labels) id, should be in range of [0, num_parties), by default 0
        """
        super().__init__(num_parties, local_datasets, primary_party_id)
        self.check_param(**kwargs)

    def check_param(self, **kwargs):
        pass

    def link(self, *args, **kwargs):
        """
        To be implemented
        Link the local datasets into a VFLAlignedDataset.
        """
        raise NotImplementedError


class MyRealVFLAlignedDataset(VFLAlignedDataset):
    def __init__(self, num_parties: int, local_datasets, primary_party_id: int = 0, **kwargs):
        """
        Vertical partitioned real dataset. If the local datasets has well-defined __getitem__ and __len__, you can
        directly use the VFLAlignedDataset class. If the local datasets are other types, you may need to implement a
        subclass of VFLAlignedDataset and redefine the __len__ and __getitem__ methods.

        Parameters
        ----------
        num_parties : int
            number of parties
        local_datasets : list
            local datasets of each party, each element is a LocalDataset (or subclass of LocalDataset)
        primary_party_id : int, optional
            primary party (the party with labels) id, should be in range of [0, num_parties), by default 0
        """
        super().__init__(num_parties, local_datasets, primary_party_id)
        self.check_param(**kwargs)

    def check_param(self, **kwargs):
        pass

    """
    Redefine the __len__ and __getitem__ methods if needed
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    """

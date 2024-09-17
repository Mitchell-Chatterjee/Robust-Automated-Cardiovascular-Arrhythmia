import os

import h5py
import numpy as np
import torch
from torch.utils.data.dataset import T_co

from src.core.datasets.ecg_dataset import EcgDataset
from src.core.datasets.ecg_interface import Split, ClassificationType


class CodeDatasetAnnotated(EcgDataset):
    """
    The child classes of CodeDataset need only implement a few different functions. In particular the import_key_data
    function from the AbstractEcgDataset ensures that we can utilize all methods from the base class.

    Otherwise, we simply need to implement the builtin __getitem__ method for the base Dataset class.
    """
    csv_path = 'modified_annotations.csv'

    def __init__(self, root_path, **dataset_kwargs):
        # Open the reading hook
        self._h5_path = os.path.join(root_path, f"{self.__class__.__name__}.h5")
        self._h5_file = h5py.File(self._h5_path, 'r')['tracings']

        super().__init__(root_path=root_path, **dataset_kwargs)

    @property
    def h5_file(self):
        return self._h5_file

    def __len__(self):
        return len(self._data_y)

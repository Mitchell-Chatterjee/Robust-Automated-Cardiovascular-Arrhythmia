import os

import h5py
import numpy as np
import torch
from torch.utils.data.dataset import T_co

from src.core.datasets.ecg_dataset import EcgDataset
from src.core.datasets.ecg_interface import Split, ClassificationType, LeadSelection


class UnifiedDataset(EcgDataset):
    """
    This dataset unifies the three major annotated datasets we have access to: Chapman, CINC-2020, CODE_Annotated.

    """

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

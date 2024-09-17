import os

import h5py
import numpy as np
import torch
from torch.utils.data.dataset import T_co

from src.core.datasets.ecg_dataset import EcgDataset
from src.core.datasets.ecg_interface import Split, ClassificationType


class Cinc2020Dataset(EcgDataset):
    """
    The CINC 2020 dataset from physionet. More information available here:
    https://physionet.org/content/challenge-2020/1.0.2/

    There are four separate datasets present within this training dataset. In particular, they are:
        1. CPSC Database and CPSC-Extra Database
        2. INCART Database
        3. PTB and PTB-XL Database
        4. The Georgia 12-lead ECG Challenge (G12EC) Database
    We have removed the INCART dataset as it is too difficult to rectify with our current requirements.
    """
    PTB_XL = 'ptb'
    cinc_no_ptb_xl_dataset = 'cinc_no_ptb_xl'

    def __init__(self, root_path, remove_ptb_xl=False, remove_segmented_data=False, **dataset_kwargs):
        # Open the reading hook
        self._h5_path = os.path.join(root_path, f"{self.__class__.__name__}.h5")
        self._h5_group = h5py.File(self._h5_path, 'a')
        self._h5_file = self._h5_group['tracings']

        super().__init__(root_path=root_path, **dataset_kwargs)

        if remove_ptb_xl:
            self.remove_ptb_xl_dataset()

        if remove_segmented_data:
            self.remove_segmented_data()

    @property
    def h5_file(self):
        return self._h5_file

    def __len__(self):
        return len(self._data_y)

    def remove_ptb_xl_dataset(self):
        # Filter out values and keep original indices
        self._data_y = self._data_y[~self._data_y['filename'].str.contains(self.PTB_XL)].reset_index(drop=True)

    def remove_segmented_data(self):
        # Filter out values and keep original indices
        self._data_y = self._data_y[~self._data_y['filename'].str.contains(self.SEGMENTED)].reset_index(drop=True)

import copy
import glob
import os

import h5py
import numpy as np
import pandas as pd
import numpy.random
import torch
from torch.utils.data.dataset import T_co

from src.core.datasets.ecg_dataset import EcgDataset
from src.core.datasets.ecg_interface import Split, ClassificationType
from src.core.utils.basics import _torch


class CodeDatasetUnannotated(EcgDataset):
    """
    The child classes of CodeDataset need only implement a few different functions. In particular the import_key_data
    function from the AbstractEcgDataset ensures that we can utilize all methods from the base class.

    Otherwise, we need to implement the builtin __getitem__ method for the base Dataset class.

    Additional exceptions are made for the Unannotated Code Dataset as there are obviously no annotations. Therefore,
    we intentionally bypass all functions in the AbstractDatasetClass that require labels.
    """
    csv_path = 'modified_unannotated.csv'

    def __init__(self, root_path, data_sub_path, **dataset_kwargs):
        # Create a dictionary for accessing values
        self.h5py_file_dict = {
            os.path.splitext(os.path.basename(filename))[0]: h5py.File(filename, 'r')['tracings']
            for filename in glob.glob(os.path.join(root_path, data_sub_path, '*.h5'))
        }

        super().__init__(root_path=root_path, **dataset_kwargs)

    def __len__(self):
        return len(self._data_y)

    def __getitem__(self, index) -> T_co:
        row = self._data_y.iloc[index]
        filename, group_id, index_in_group = row['filename'], row['group_id'], row['index_in_group']

        ecg_data = self.read_ecg_from_h5py(group_id, index_in_group)

        # Select the leads we are using
        ecg_data = ecg_data[self._custom_lead_selection]
        label = copy.deepcopy(ecg_data)

        # Randomly apply transformations to the data (For now only one)
        if self.transformations:
            return _torch(np.random.choice(self.transformations, 1)[0]((ecg_data, label)))

        # Return raw ecg signal as it will be filtered and moved to gpu by patch masker
        return _torch((ecg_data, label))

    def read_ecg_from_h5py(self, key, index):
        """Read ecg from h5py file"""
        return self.h5py_file_dict[key][index, :, :]

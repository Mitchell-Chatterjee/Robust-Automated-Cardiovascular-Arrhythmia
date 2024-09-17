import os.path

import h5py

from src.core.datasets.ecg_dataset import EcgDataset
from src.core.datasets.ecg_interface import Split, ClassificationType


class ChapmanDataset(EcgDataset):
    """
    The Chapman University, Shaoxing People’s Hospital dataset from physionet. More information available here:
    https://physionet.org/content/ecg-arrhythmia/1.0.0/

    This dataset contains 45,152 patient ECGs. At a sampling rate of 500Hz and a recording length of 10 seconds.
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

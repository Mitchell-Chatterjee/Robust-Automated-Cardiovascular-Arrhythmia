import os

import h5py
import pandas as pd

from src.core.datasets.ecg_preprocessing import EcgPreprocessing
from src.core.datasets.ptb_xl.ptb_xl_dataset import PtbXlDataset
from src.core.datasets.unified_annotated_dataset.unified_dataset import UnifiedDataset


class UnifiedPreprocessing(EcgPreprocessing):
    """
    This dataset unifies the three major annotated datasets we have access to: Chapman, CINC-2020, CODE_Annotated.

    """

    def __init__(self, root_path, data_sub_path, datasets, h5_name=f'{UnifiedDataset.__name__}.h5'):
        # Prepare datasets
        self.datasets = datasets
        assert None not in self.datasets, "Datasets cannot be none. At least one from: chapman, ptb-xl, cinc, code."

        """
        By default the h5 name is the name of the unified class. However, we allow it to be changed in the case 
        that we want to override the standard handling of the dataset. For instance, we may pass in the PTB-XL class
        name if we want to use the PTB-XL class functions on the unified dataset.
        """
        self._h5_path = os.path.join(root_path, h5_name)

        super().__init__(root_path=root_path, data_sub_path=data_sub_path, preprocessing=True)

    def _postprocess_data(self, **kwargs):
        """Not required for this class"""
        pass

    def _preprocess_data(self, row, to_delete, to_add):
        """Not required for this class"""
        pass

    def __read_data__(self):
        # Get label legend
        snomed_mappings = pd.read_csv(self.snomed_path, dtype={'SNOMED CT Code': str})
        # Rename the column for snomed mappings
        snomed_mappings = snomed_mappings.rename(columns={'SNOMED CT Code': 'label'})

        # If the csv already exists or the datasets are not defined then skip
        if not os.path.exists(self._root_path):
            # Make the root folder
            os.makedirs(self._root_path)
            os.mkdir(self._modified_dataset_path)

        # Concatenate the annotations together and write to csv
        self._data_y = pd.concat([elem.data_y for elem in self.datasets]).reset_index(drop=True)

        # If the PTB-XL dataset is in the list of datasets we will handle things slightly differently
        self._data_y = self._data_y[['label', 'filename']]
        self._data_y.to_csv(self._dataset_path, index=False)

        # Create new h5 file
        self._h5_file = h5py.File(self._h5_path, 'w')
        # Create an h5py dataset with dimensions (num of entries, num of leads, sample rate)
        h5_dataset = self._h5_file.create_dataset('tracings', (
            len(self._data_y), self.NUMBER_OF_LEADS, self.resample_rate))

        # Concatenate their h5py files together
        count = 0
        for (df, h5_file) in [(elem.data_y, elem.h5_file) for elem in self.datasets]:
            # Have to do it iteratively as some of the datasets have options to remove indices
            for _, (original_index, *_) in df.iterrows():
                h5_dataset[count, :, :] = h5_file[original_index, :, :]
                count += 1
        # Close the write file
        self._h5_file.close()

        # Save snomed map
        self._snomed_map = snomed_mappings
        # Read from the csv file and open the h5 file for reading
        self._data_y = pd.read_csv(self._dataset_path)
        self._h5_file = h5py.File(self._h5_path, mode='r')['tracings']

    def __preprocess_data__(self):
        """Not required as all data is pre-processed in subordinate classes"""
        pass

    def import_key_data(self, path):
        """Not required as all data is imported from subordinate classes"""
        pass

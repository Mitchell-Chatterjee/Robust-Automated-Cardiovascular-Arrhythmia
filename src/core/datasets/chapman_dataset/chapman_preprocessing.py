import math
import os
import shutil

import h5py
import numpy as np
import pandas as pd
import wfdb
from scipy.io import loadmat
import scipy.signal as sgn

from src.core.datasets.chapman_dataset.chapman_dataset import ChapmanDataset
from src.core.datasets.ecg_interface import ClassificationType
from src.core.datasets.ecg_preprocessing import EcgPreprocessing


class ChapmanPreprocessing(EcgPreprocessing):
    """
    The Chapman University, Shaoxing People’s Hospital dataset from physionet. More information available here:
    https://physionet.org/content/ecg-arrhythmia/1.0.0/

    This dataset contains 45,152 patient ECGs. At a sampling rate of 500Hz and a recording length of 10 seconds.
    """
    _original_dataset_counter = 0

    def __init__(self, root_path, data_sub_path, preprocessing):
        self._temp_h5_path = os.path.join(root_path, f"temp.h5")
        self._temp_h5_file = None
        self._temp_h5_dataset = None

        self._h5_path = os.path.join(root_path, f"{ChapmanDataset.__name__}.h5")
        super().__init__(root_path, data_sub_path, preprocessing)

    def extract_and_resample(self, filename):
        # Extract data
        _, freq, leads = self.read_ecg_from_wfdb(filename)
        signal = np.asarray(loadmat(filename)['val'], dtype=np.float64)
        if freq == self.resample_freq:
            return signal
        # Need to consider the difference in length for a 10-second time signal
        scaling_constant = signal.shape[1] / (freq * self.time_window)
        # Resample data to resample_rate * scaled 10-second constant
        return sgn.resample(signal, math.ceil(scaling_constant * self.resample_rate), axis=1)

    def _preprocess_data(self, row, to_delete, to_add):
        """Potential downsampling required from 500Hz"""
        # Extract data from rows
        label, filename = row

        if row.name == 0:
            # Create new h5 file
            self._temp_h5_file = h5py.File(self._temp_h5_path, 'w')
            # Create an h5py dataset with dimensions (num of entries, num of leads, sample rate)
            self._temp_h5_dataset = self._temp_h5_file.create_dataset('tracings', (
                len(self._data_y), self.NUMBER_OF_LEADS, self.resample_rate))

        # Extract and downsample data into h5_dataset
        try:
            self._temp_h5_dataset[self._original_dataset_counter, :, :] = self.extract_and_resample(filename=filename)
            self._original_dataset_counter += 1
        except Exception:
            to_delete.append(row.name)

    def _postprocess_data(self, **kwargs):
        """Move all data to an h5py file"""
        # Create new h5 file
        self._h5_file = h5py.File(self._h5_path, 'w')
        # Create an h5py dataset with dimensions (num of entries, num of leads, sample rate)
        h5_dataset = self._h5_file.create_dataset('tracings', (
            len(self._data_y), self.NUMBER_OF_LEADS, self.resample_rate))
        # Append data to the main h5 file
        h5_dataset[:self._original_dataset_counter, :, :] = \
            self._temp_h5_dataset[:self._original_dataset_counter, :, :]

        # Close the writing hook
        self._temp_h5_file.close()
        self._h5_file.close()

        # Remove files and folders other than h5py and modified dataset folder
        for file in [os.path.join(self._root_path, f) for f in os.listdir(self._root_path)]:
            if file not in [self._h5_path, self._modified_dataset_path]:
                if os.path.isfile(file):
                    os.remove(file)
                else:
                    shutil.rmtree(file)

    def import_key_data(self, path):
        def load_challenge_data(f_name):
            x = loadmat(f_name)
            temp_data = np.asarray(x['val'], dtype=np.float64)
            new_file = f_name.replace('.mat', '.hea')
            input_header_file = os.path.join(new_file)
            with open(input_header_file, 'r') as f:
                head_data = f.readlines()
            return temp_data, head_data

        labels = []
        ecg_filenames = []
        for subdir, dirs, files in sorted(os.walk(path)):
            for filename in files:
                filepath = subdir + os.sep + filename
                if filepath.endswith(".mat"):
                    data, header_data = load_challenge_data(filepath)

                    # Append the data
                    labels.append(header_data[15][5:-1])
                    ecg_filenames.append(filepath)
                    # gender.append(header_data[14][6:-1])
                    # age.append(header_data[13][6:-1])

        # Return a dataframe
        df_data = pd.DataFrame(
            {
                'label': labels,
                'filename': ecg_filenames
            },
            dtype=str
        )

        return df_data

    @staticmethod
    def read_ecg_from_wfdb(path):
        """Read wfdb record"""
        record = wfdb.rdrecord(os.path.splitext(path)[0])
        return record.p_signal.T, record.fs, record.sig_name

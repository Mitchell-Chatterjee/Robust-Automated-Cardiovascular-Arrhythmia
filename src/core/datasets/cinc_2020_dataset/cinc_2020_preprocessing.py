import math
import os
import shutil
from itertools import islice

import h5py
import numpy as np
import pandas as pd
import wfdb
from scipy.io import loadmat
import scipy.signal as sgn

from src.core.datasets.cinc_2020_dataset.cinc_2020_dataset import Cinc2020Dataset
from src.core.datasets.ecg_preprocessing import EcgPreprocessing


class Cinc2020Preprocessing(EcgPreprocessing):
    """
    The CINC 2020 dataset from physionet. More information available here:
    https://physionet.org/content/challenge-2020/1.0.2/

    There are four separate datasets present within this training dataset. In particular, they are:
        1. CPSC Database and CPSC-Extra Database
        2. INCART Database
        3. PTB and PTB-XL Database
        4. The Georgia 12-lead ECG Challenge (G12EC) Database
    We have removed the INCART dataset as it is too difficult to rectify with our current requirements.

    Some dataset specific pre-processing is required for each dataset. This includes, bringing each dataset sample to a
    length of 10 seconds. While also ensuring each signal is sampled at 500Hz.
        1. CPSC Database and CPSC-Extra Database
            - Sampling rate: 500Hz
            - Length of recordings: 6 - 60 seconds
                - Managed with a sliding window for longer segments and zero padding for shorter segments.
                - Note that there are approximately 700 entries with multiple annotations. According to the guidelines
                it is acceptable in this case to only take one of the annotations for the entire record as they are
                record level annotations. This is what we do. This may however, be changed in the future.
        2. PTB and PTB-XL Database
            - Sampling rate: 500Hz
                - First 516 records are sampled at 1000Hz. This is handled with a resample rate of 500Hz.
            - Length of recordings: 10 seconds
        3. The Georgia 12-lead ECG Challenge (G12EC) Database
            - Sampling rate: 500Hz
            - Length of recordings: 10 seconds
    To modify these signals we follow the steps outlined in: https://doi.org/10.1038/s41598-023-38532-9
    """

    PETERSBURG = 'petersburg'

    _original_dataset_counter = 0
    _segmented_dataset_counter = 0

    def __init__(self, root_path, data_sub_path, preprocessing, segment_size=0.7):
        """
        segment_size: Minimum size of a segmented signal (as a percentage) to be considered for classification.
        """
        # For the original signals
        self._original_h5_path = os.path.join(root_path, f"original.h5")
        self._original_h5_file = None
        self._original_h5_dataset = None

        # For the segmented signals
        self._segmented_h5_path = os.path.join(root_path, f"segmented.h5")
        self._segmented_h5_file = None
        self._segmented_h5_dataset = None

        self._segment_size = segment_size

        self._h5_path = os.path.join(root_path, f"{Cinc2020Dataset.__name__}.h5")

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

    def _preprocess_data(self, row, to_add, to_delete):
        """
        We must make an exception for any files within the CPSC dataset as they may be between 6 and 60 seconds.

        To do this we will use a sliding window on the data. If the data is longer than 10 seconds. We will break it
        up into multiple signals and create some new files with the same label.

        If it is less than 10 seconds, we will zero pad the data and rewrite it to the same file we have read from.
        """
        # Extract data from rows
        label, filename = row

        # If this is the first row then instantiate the h5 file
        if row.name == 0:
            # Create new h5 file
            self._original_h5_file = h5py.File(self._original_h5_path, 'w')
            self._segmented_h5_file = h5py.File(self._segmented_h5_path, 'w')
            # Create an h5py dataset with dimensions (num of entries, num of leads, sample rate)
            self._original_h5_dataset = self._original_h5_file.create_dataset('tracings', (
                len(self._data_y), self.NUMBER_OF_LEADS, self.resample_rate))
            self._segmented_h5_dataset = self._segmented_h5_file.create_dataset('tracings', (
                len(self._data_y), self.NUMBER_OF_LEADS, self.resample_rate))

        # We simply remove the St. Petersburg dataset
        if self.PETERSBURG in filename:
            to_delete.append(row.name)
            return

        # Extract data from mat file
        signal = self.extract_and_resample(filename)

        # No need to process the data if it is already of the correct size
        if signal.shape[1] == self.resample_rate:
            self._original_h5_dataset[self._original_dataset_counter, :, :] = signal
            self._original_dataset_counter += 1
            return

        # Break signal into segments of length self.resample_rate
        segments = np.array_split(ary=signal,
                                  indices_or_sections=list(
                                      range(self.resample_rate, signal.shape[1], self.resample_rate)),
                                  axis=1)

        # Get last slice and apply zero padding if necessary
        last_segment = segments.pop()
        front_pad = (self.resample_rate - last_segment.shape[1]) // 2
        back_pad = self.resample_rate - last_segment.shape[1] - front_pad

        # We will only re-append the last segment if the signal contains at least _segment_size% non-zero values.
        if last_segment.shape[1] / self.resample_rate >= self._segment_size:
            segments.append(
                np.pad(last_segment, [(0, 0), (front_pad, back_pad)], 'constant',
                       constant_values=0)
            )

        if len(segments) == 0:
            to_delete.append(row.name)
            return

        # If segments is greater than 1, we need to save the new files
        for data_segment in segments[1:]:
            # Add the segmented data to the segmented data h5py file
            self._segmented_h5_dataset[self._segmented_dataset_counter, :, :] = data_segment
            self._segmented_dataset_counter += 1

            # Append to lists
            to_add['label'].append(label)
            to_add['filename'].append(self.SEGMENTED)

        # Write the current data to the h5py file
        self._original_h5_dataset[self._original_dataset_counter, :, :] = segments[0]
        self._original_dataset_counter += 1

    def _postprocess_data(self, **kwargs):
        """Move all data to an h5py file"""
        # Create new h5 file
        self._h5_file = h5py.File(self._h5_path, 'w')
        # Create an h5py dataset with dimensions (num of entries, num of leads, sample rate)
        h5_dataset = self._h5_file.create_dataset('tracings', (
            len(self._data_y), self.NUMBER_OF_LEADS, self.resample_rate))

        # Append data to the main h5 file
        h5_dataset[:self._original_dataset_counter, :, :] = \
            self._original_h5_dataset[:self._original_dataset_counter, :, :]
        h5_dataset[self._original_dataset_counter:, :, :] = \
            self._segmented_h5_dataset[:self._segmented_dataset_counter, :, :]

        # Close the writing hook
        self._original_h5_file.close()
        self._segmented_h5_file.close()
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

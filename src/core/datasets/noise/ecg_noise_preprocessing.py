import math
import os
import shutil

import h5py
import numpy as np
import wfdb
import scipy.signal as sgn

from src.core.datasets.ecg_interface import EcgInterface
from src.core.datasets.noise.ecg_noise_dataset import EcgNoiseDataset


class EcgNoisePreprocessing(EcgInterface):
    """
    The MIT-BIH Noise Stress Test Database. Requires similar pre-processing to other signal data.

    https://www.physionet.org/content/nstdb/1.0.0/
    """

    def __init__(self, root_path, ma_path='ma', em_path='em'):
        # Own variables
        self._root_path = root_path
        self._ma_path = os.path.join(root_path, ma_path)
        self._ma_path = os.path.join(root_path, em_path)

        self._h5_path_muscle = os.path.join(root_path, f"{EcgNoiseDataset.__name__}_ma.h5")
        self._h5_path_electrode = os.path.join(root_path, f"{EcgNoiseDataset.__name__}_em.h5")

        self.__read_data__()
        self.__preprocess_data__()
        self.__postprocess_data__()

    def extract_and_resample(self, filename):
        # Extract data
        signal, freq, leads = self.read_ecg_from_wfdb(filename)
        if freq == self.resample_freq:
            return signal
        # Need to consider the difference in length for a 10-second time signal
        scaling_constant = signal.shape[1] / (freq * self.time_window)
        # Resample data to resample_rate * scaled 10-second constant
        return sgn.resample(signal, math.ceil(scaling_constant * self.resample_rate), axis=1)

    def __read_data__(self):
        """Read in the muscle artefact and electrode motion signals"""
        # Read in the muscle artefact signal and flatten to combine leads
        self.ma_signal = self.extract_and_resample(self._ma_path).flatten()
        # Read in the electrode motion signal and flatten to combine leads
        self.em_signal = self.extract_and_resample(self._ma_path).flatten()

        # Calculate number of segments and take floor
        ma_segments = self.ma_signal.shape[0] // self.resample_rate
        em_segments = self.em_signal.shape[0] // self.resample_rate

        # Prepare the h5 datasets for both types of noise signals
        self._ma_h5_file = h5py.File(self._h5_path_muscle, 'w')
        self._em_h5_file = h5py.File(self._h5_path_electrode, 'w')

        # Create an h5py dataset with dimensions (num of entries (segments), sample rate)
        self._ma_h5_dataset = self._ma_h5_file.create_dataset('tracings', (
            ma_segments, self.resample_rate))
        self._em_h5_dataset = self._em_h5_file.create_dataset('tracings', (
            em_segments, self.resample_rate))

    def __preprocess_data__(self):
        """Read in and pre-process the signal data into separate h5 files"""
        # Pass a sliding window over the ma_signal and em_signal, breaking it into parts
        ma_segments = np.array_split(ary=self.ma_signal,
                                     indices_or_sections=list(
                                         range(self.resample_rate, self.ma_signal.shape[0], self.resample_rate)),
                                     axis=0)
        em_segments = np.array_split(ary=self.em_signal,
                                     indices_or_sections=list(
                                         range(self.resample_rate, self.em_signal.shape[0], self.resample_rate)),
                                     axis=0)

        # Break off last section from both in case it's a different size
        ma_segments.pop()
        em_segments.pop()

        # Now save the segments to their respective files
        for i, ma_signal in enumerate(ma_segments):
            self._ma_h5_dataset[i, :] = ma_signal
        for j, em_signal in enumerate(em_segments):
            self._em_h5_dataset[j, :] = em_signal

    def __postprocess_data__(self):
        """Remove other files"""
        # Close writing hook
        self._ma_h5_file.close()
        self._em_h5_file.close()

        # Remove files and folders other than h5py files for electrode motion and muscle artefact
        for file in [os.path.join(self._root_path, f) for f in os.listdir(self._root_path)]:
            if file not in [self._h5_path_muscle, self._h5_path_electrode]:
                if os.path.isfile(file):
                    os.remove(file)
                else:
                    shutil.rmtree(file)

    @staticmethod
    def read_ecg_from_wfdb(path):
        """Read wfdb record"""
        record = wfdb.rdrecord(path)
        return np.asarray(record.p_signal.T, dtype=np.float64), record.fs, record.sig_name
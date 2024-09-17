import os

import h5py
import numpy as np
import torch

from src.core.datasets.ecg_interface import EcgInterface


class EcgNoiseDataset(EcgInterface):
    """
    The MIT-BIH Noise Stress Test Database.
    Dataset containing ecg noise. Allows us to sample noise in order to make our model more robust to anomalies.

    https://www.physionet.org/content/nstdb/1.0.0/
    """

    def __init__(self, root_path, number_of_leads, proportion_em=0.5):
        """
        proportion_em: Value between 0-1. Decides what proportion of the noise will be electrode motion and what
        proportion will be muscle artefact.
        """
        # Open the reading hook
        self._h5_file_muscle = h5py.File(os.path.join(root_path, f"{self.__class__.__name__}_ma.h5"), 'r')['tracings']
        self._h5_file_electrode = h5py.File(os.path.join(root_path, f"{self.__class__.__name__}_em.h5"), 'r')[
            'tracings']

        # Prepare random sampling measures
        assert proportion_em <= 1, 'Value for proportion_em must be between 0-1'
        self.number_of_leads = number_of_leads
        self.proportion_em = round(proportion_em * self.number_of_leads)
        self.proportion_ma = self.number_of_leads - self.proportion_em

    @property
    def h5_file_muscle_artefacts(self):
        return self._h5_file_muscle

    @property
    def h5_file_eletrode_motion(self):
        return self._h5_file_electrode

    def __len__(self):
        return len(self._h5_file_muscle) + len(self._h5_file_electrode)

    def get_noise_batch(self, batch_size, seq_len):
        """
        Generate a noise signal, with given number of leads, by sampling from electrode motion and muscle artefact noise.
        """
        # Select number of leads from electrode motion
        em_indicies = sorted(list(
            set(np.random.choice(a=len(self._h5_file_electrode), size=batch_size * self.proportion_em, replace=True))
        ))
        em_leads = self._h5_file_electrode[em_indicies, :]
        # Perform resampling ourselves as h5py doesn't allow you to access the same index twice (sampling with replacement)
        em_resample_leads = np.random.choice(a=len(em_leads), size=batch_size * self.proportion_em - len(em_leads),
                                             replace=True)
        em_leads = np.concatenate([em_leads, em_leads[em_resample_leads]], axis=0)
        np.random.shuffle(em_leads)

        # Select number of leads from muscle artefacts
        ma_indicies = sorted(list(
            set(np.random.choice(a=len(self._h5_file_muscle), size=batch_size * self.proportion_ma, replace=True))
        ))
        ma_leads = self._h5_file_muscle[ma_indicies, :]
        # Perform resampling ourselves as h5py doesn't allow you to access the same index twice (sampling with replacement)
        ma_resample_leads = np.random.choice(a=len(ma_leads), size=batch_size * self.proportion_ma - len(ma_leads),
                                             replace=True)
        ma_leads = np.concatenate([ma_leads, ma_leads[ma_resample_leads]], axis=0)
        np.random.shuffle(ma_leads)

        # Concatenate the two sets of noise data and shuffle them
        noise_ecg = np.concatenate([em_leads, ma_leads], axis=0)
        np.random.shuffle(noise_ecg)

        # Reshape into batches
        noise_ecg = noise_ecg.reshape((self.number_of_leads, seq_len))

        # Return noise ecg sample
        return noise_ecg

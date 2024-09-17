import math
import os
import shutil
from ast import literal_eval

import h5py
import numpy as np
import pandas as pd
import wfdb
import scipy.signal as sgn

from src.core.datasets.ecg_preprocessing import EcgPreprocessing
from src.core.datasets.ptb_xl.ptb_xl_dataset import PtbXlDataset, DiagnosticClass


class PtbXlPreprocessing(EcgPreprocessing):
    """
    The PTB-XL dataset. More information available here:
    https://physionet.org/content/ptb-xl/1.0.3/

    The PTB-XL ECG dataset is a large dataset of 21799 clinical 12-lead ECGs from 18869 patients of 10 second length.
    Sampled at both 500Hz and 100Hz. This dataset is set up to use the 500Hz and downsample if required.

    While it is true that the PTB-XL dataset is a subset of the CINC-2020 dataset. We would like to keep the additional
    information for stratification folds, patient ids, etc, that are not present CINC-2020 dataset. Therefore,
    we will need to use the PTB-XL dataset itself.
    """
    dataset_df = 'ptbxl_database.csv'
    dataset_cols = ['patient_id', 'scp_codes', 'strat_fold', 'filename_lr', 'filename_hr', 'ST-DEPR-MI', 'ST-ELEV-MI']

    scp_codes = 'scp_statements.csv'
    scp_cols = ['Unnamed: 0', 'description', 'diagnostic', 'rhythm', 'form', 'diagnostic_class',
                'diagnostic_subclass', 'Statement Category']

    def __init__(self, root_path, data_sub_path, preprocessing):
        self.dataset_df_path = os.path.join(root_path, self.dataset_df)
        self.scp_codes_path = os.path.join(root_path, self.scp_codes)
        self._h5_path = os.path.join(root_path, f"{PtbXlDataset.__name__}.h5")

        super().__init__(root_path=root_path, data_sub_path=data_sub_path, preprocessing=preprocessing)

    def extract_and_resample(self, filename):
        # Extract data
        signal, freq, leads = self.read_ecg_from_wfdb(filename)
        if freq == self.resample_freq:
            return signal
        # Need to consider the difference in length for a 10-second time signal
        scaling_constant = signal.shape[1] / (freq * self.time_window)
        # Resample data to resample_rate * scaled 10-second constant
        return sgn.resample(signal, math.ceil(scaling_constant * self.resample_rate), axis=1)

    def map_label_names(self, df_data, snomed_mappings):
        """
        Overrides mapping from superclass.
        """
        # Convert label names to list
        df_data['label'] = df_data['label'].apply(lambda label: list(literal_eval(label).keys()))

        # Convert diagnostic, diagnostic-sub and diagnostic-super to dictionaries so we can apply them to each label
        label_to_super = {code: label for _, (code, label) in
                          snomed_mappings[['label', 'diagnostic_class']].iterrows()}
        label_to_sub = {code: label for _, (code, label) in
                        snomed_mappings[['label', 'diagnostic_subclass']].iterrows()}

        label_to_form = {label: label if form else None for _, (label, form) in
                         snomed_mappings[['label', 'form']].iterrows()}
        label_to_rhythm = {label: label if rhythm else None for _, (label, rhythm) in
                           snomed_mappings[['label', 'rhythm']].iterrows()}
        label_to_diagnostic = {label: label if rhythm else None for _, (label, rhythm) in
                               snomed_mappings[['label', 'diagnostic']].iterrows()}

        df_data[DiagnosticClass.SUPER.value] = df_data['label'].apply(lambda labels:
                                                                      list(set(elem for elem in
                                                                               [*map(label_to_super.get, labels)]
                                                                               if elem is not None)))
        df_data[DiagnosticClass.SUB.value] = df_data['label'].apply(lambda labels:
                                                                    list(set(elem for elem in
                                                                             [*map(label_to_sub.get, labels)]
                                                                             if elem is not None)))

        df_data[DiagnosticClass.FORM.value] = df_data['label'].apply(lambda labels:
                                                                     list(set(elem for elem in
                                                                              [*map(label_to_form.get, labels)]
                                                                              if elem is not None)))
        df_data[DiagnosticClass.RHYTHM.value] = df_data['label'].apply(lambda labels:
                                                                       list(set(elem for elem in
                                                                                [*map(label_to_rhythm.get, labels)]
                                                                                if elem is not None)))
        df_data[DiagnosticClass.DIAG.value] = df_data['label'].apply(lambda labels:
                                                                     list(set(elem for elem in
                                                                              [*map(label_to_diagnostic.get, labels)]
                                                                              if elem is not None)))

        # Drop rows without label
        return df_data[~df_data['label'].isna()]

    def __read_data__(self):
        """Slightly different from other datasets as the dataframes have already been created"""
        # Read in labels
        snomed_mappings = pd.read_csv(self.scp_codes_path, usecols=self.scp_cols, dtype={'description': str})
        snomed_mappings[['rhythm', 'form', 'diagnostic']] = \
            snomed_mappings[['rhythm', 'form', 'diagnostic']].fillna(0).astype(int)
        snomed_mappings = snomed_mappings.rename(columns={'Unnamed: 0': 'label'})
        snomed_mappings[['form', 'rhythm', 'diagnostic']] = \
            snomed_mappings[['form', 'rhythm', 'diagnostic']].astype(bool)
        snomed_mappings = snomed_mappings.replace({np.nan: None})

        # Create folder for hosting datasets
        if not os.path.exists(self._modified_dataset_path):
            os.mkdir(self._modified_dataset_path)

        if os.path.exists(self._dataset_path) and not self._preprocessing:
            df_data = pd.read_csv(self._dataset_path, dtype={'label': str})
        else:
            # Read in data
            df_data = pd.read_csv(self.dataset_df_path)

            # Check if we are using version 1.0.2 or version 1.0.3 of the dataset
            if self.dataset_cols[-1] not in df_data:
                self.dataset_cols = self.dataset_cols[:-2]

            df_data = df_data[self.dataset_cols]

            df_data[['patient_id', 'strat_fold']] = df_data[['patient_id', 'strat_fold']].astype(int)

            if self.resample_freq == 100:
                df_data = df_data.rename(columns={'scp_codes': 'label', 'filename_lr': 'filename'})
                df_data = df_data.drop(columns='filename_hr')
            else:
                df_data = df_data.rename(columns={'scp_codes': 'label', 'filename_hr': 'filename'})
                df_data = df_data.drop(columns='filename_lr')

            # Augment with the full root path
            df_data['filename'] = df_data['filename'].apply(lambda filename: os.path.join(self._root_path, filename))

            # Map to label names
            df_data = self.map_label_names(df_data, snomed_mappings)

            # Write to a csv for future use
            df_data.to_csv(self._dataset_path, index=False)

        self._snomed_map = snomed_mappings
        self._data_y = df_data

    def import_key_data(self, path):
        """Not required as the dataframe has already been read in"""
        pass

    def _preprocess_data(self, row, to_delete, to_add):
        """Potential downsampling required from 500Hz"""
        # Extract data from rows
        label, filename = row['label'], row['filename']

        if row.name == 0:
            # Create new h5 file
            self._h5_file = h5py.File(self._h5_path, 'w')
            # Create an h5py dataset with dimensions (num of entries, num of leads, sample rate)
            self._h5_dataset = self._h5_file.create_dataset('tracings', (
                len(self._data_y), self.NUMBER_OF_LEADS, self.resample_rate))

        # Extract and downsample data into h5_dataset
        self._h5_dataset[row.name, :, :] = self.extract_and_resample(filename=filename)

    def _postprocess_data(self, **kwargs):
        """Move all data to an h5py file"""
        # Close the writing hook
        self._h5_file.close()

        # Remove files and folders other than h5py and modified dataset folder
        for file in [os.path.join(self._root_path, f) for f in os.listdir(self._root_path)]:
            if file not in [self._h5_path, self._modified_dataset_path]:
                if os.path.isfile(file):
                    os.remove(file)
                else:
                    shutil.rmtree(file)

    @staticmethod
    def read_ecg_from_wfdb(path):
        """Read wfdb record"""
        record = wfdb.rdrecord(path)
        return np.asarray(record.p_signal.T, dtype=np.float64), record.fs, record.sig_name

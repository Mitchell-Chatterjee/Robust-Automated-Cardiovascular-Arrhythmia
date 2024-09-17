import glob
import math
import shutil
import tarfile

import h5py
import numpy as np
import pandas as pd
import scipy.signal as sgn
import os
import logging

import wfdb
from scipy.io import loadmat

from src.core.datasets.code_dataset.code_dataset_annotated import CodeDatasetAnnotated
from src.core.datasets.ecg_preprocessing import EcgPreprocessing


class CodePreprocessing(EcgPreprocessing):
    """
    This dataset is a child class of the CodeDataset.

    Due to the unique nature of the CODE dataset itself, we require another class whose sole purpose is to enable the
    correct pre-processing of the CODE dataset itself. As this class does.

    It is not to be used for anything other than the initial call to pre-process the dataset from the shell script.
    """

    # Place to save modified annotated and unannotated data
    modified_annotations_file = 'modified_annotations.csv'
    modified_unannotated_file = 'modified_unannotated.csv'
    modified_preprocessing_file = 'modified_preprocessing.csv'

    # An empty dictionary that will contain a reference to all h5py files
    h5py_file_dict = {}

    # Maps the label to snomed ct codes for consolidation with other datasets
    labels = {'1dAVb': '270492004', 'RBBB': '59118001', 'LBBB': '164909002',
              'SB': '426177001', 'AF': '164889003', 'ST': '427084000'}

    # Assigns normal sinus rhythm to heartbeats without a label
    NO_LABEL = 'NO_LABEL'

    csv_path = 'preprocessing_dataset.csv'

    current_h5_file = None
    current_h5_dataset = None
    last_folder_name = None

    def __init__(self, root_path, data_sub_path, preprocessing, record_file='records.txt',
                 annotations_file='annotations.csv', segment_size=0.7):
        """
        segment_size: Minimum size of a segmented signal (as a percentage) to be considered for classification.
        """
        self._record_file = record_file
        self._annotations_file = annotations_file
        self._segment_size = segment_size

        self._modified_dataset_path = os.path.join(root_path, self.dataset_folder)
        self._modified_annotated_path = os.path.join(self._modified_dataset_path, self.modified_annotations_file)
        self._modified_unannotated_path = os.path.join(self._modified_dataset_path, self.modified_unannotated_file)
        self._modified_preprocessing_path = os.path.join(root_path, self.modified_preprocessing_file)

        self._annotated_h5py_path = os.path.join(root_path, CodeDatasetAnnotated.__name__)

        self.annotated_data, self.unannotated_data = None, None

        # Create a dictionary for transferring from 8-leads to 12
        self.indices_leads_to_copy = [self.ecg_dict[lead] for lead in self.eight_leads]

        super().__init__(root_path=root_path, data_sub_path=data_sub_path, preprocessing=preprocessing)

    def _get_file_paths(self):
        """Retrieves and massages file paths from CODE records file."""
        # Read in the records from the csv file
        files = pd.read_csv(os.path.join(self._root_path, self._record_file), names=['filename'], header=None)
        # Get group name. This will be used for finding the h5py file later.
        files['group_id'] = files['filename'].str.partition('/')[0].str.upper()
        # Now assign the index of each group member in the group. To be used for indexing the h5py file later.
        files['index_in_group'] = files.groupby('group_id')['group_id'].rank(method='first').astype(int) - 1
        # Augment with the full root path
        files['filename'] = files['filename'].apply(lambda filename:
                                                    os.path.join(self._root_path, self._data_sub_path, filename))
        return files

    def import_key_data(self, path):
        """Overwrite from AbstractEcgClass to simply read from generated csv file."""
        return pd.read_csv(self._modified_preprocessing_path, dtype={'index_in_group': int})

    def map_label_names(self, df_data, snomed_mappings):
        """As this dataset does not contain true labels, we just skip this step."""
        return df_data

    def __read_data__(self):
        # Get file paths
        if not os.path.exists(self._modified_preprocessing_path):
            file_paths = self._get_file_paths()
            file_paths.to_csv(self._modified_preprocessing_path, index=False)

        super().__read_data__()

    def _add_leads(self, current_ecg, leads):
        """Generates additional leads, given an 8-lead signal. Returns a 12-lead signal."""
        # Prep new 12-lead array
        ecg_targetleads = np.zeros([len(self.all_leads), current_ecg.shape[1]])

        # Copy over existing leads
        ecg_targetleads[self.indices_leads_to_copy] = current_ecg

        # Add leads
        ecg_targetleads[self.ecg_dict['III'], :] = ecg_targetleads[self.ecg_dict['II'], :] \
                                                    - ecg_targetleads[self.ecg_dict['I'], :]
        ecg_targetleads[self.ecg_dict['AVR'], :] = -(ecg_targetleads[self.ecg_dict['I'], :]
                                                     + ecg_targetleads[self.ecg_dict['II'], :]) / 2
        ecg_targetleads[self.ecg_dict['AVL'], :] = (ecg_targetleads[self.ecg_dict['I'], :]
                                                    - ecg_targetleads[self.ecg_dict['III'], :]) / 2
        ecg_targetleads[self.ecg_dict['AVF'], :] = (ecg_targetleads[self.ecg_dict['II'], :]
                                                    + ecg_targetleads[self.ecg_dict['III'], :]) / 2
        return ecg_targetleads

    def downsample(self, signal, freq):
        if freq == self.resample_freq:
            return signal
        # Need to consider the difference in length for a 10-second time signal
        scaling_constant = signal.shape[1] / (freq * 10)
        # Resample data to resample_rate * scaled 10-second constant
        return sgn.resample(signal, math.ceil(scaling_constant * self.resample_rate), axis=1)

    def _pre_process_aux(self, signal, freq, leads):
        """Resample to resample_rate, zero-pad signals, add missing leads and normalize.
                This follows the steps from: https://github.com/antonior92/ecg-preprocessing/blob/main/preprocess.py"""

        # If this signal has already been augmented then we simply skip it
        if len(leads) == self.NUMBER_OF_LEADS:
            return

        # Upsample to self.resample_freq
        ecg_resampled = self.downsample(signal, freq)
        # If the signal is longer than self.resample_rate. Remove equally from both sides.
        if ecg_resampled.shape[1] > self.resample_rate:
            index = (ecg_resampled.shape[1] - self.resample_rate) // 2
            remainder = (ecg_resampled.shape[1] - self.resample_rate) % 2
            ecg_resampled = ecg_resampled[:, index:-index-remainder]

        # Generate missing leads
        ecg_targetleads = self._add_leads(current_ecg=ecg_resampled, leads=leads)

        # We will only add this signal if the signal contains at least _segment_size% non-zero values.
        if ecg_targetleads.shape[1] / self.resample_rate < self._segment_size:
            # Throw an exception if it is less than the threshold. The outer method will catch it and remove the row.
            raise Exception('ECG is not long enough to be used for classification.')

        # Zero-pad the signal
        front_pad = (self.resample_rate - ecg_targetleads.shape[1]) // 2
        back_pad = self.resample_rate - ecg_targetleads.shape[1] - front_pad
        ecg_preprocessed = np.pad(ecg_targetleads, [(0, 0), (front_pad, back_pad)], 'constant',
                                  constant_values=0)
        return ecg_preprocessed

    def _process_folder(self, folder_name):
        folder_path = os.path.join(self._root_path, self._data_sub_path, folder_name)

        # Delete last folder and archive. Close last h5py file output channel.
        if self.current_h5_file is not None:
            # Delete last folder and archive
            self.delete_archive(
                folder_path=os.path.join(self._root_path, self._data_sub_path, self.last_folder_name),
                archive_path=os.path.join(self._root_path, self._data_sub_path, self.last_folder_name + '.tar'),
                name=self.last_folder_name
            )
            # Close output channel
            self.current_h5_file.close()
            logging.debug(f'Done pre-processing {self.last_folder_name}\n')

        # Extract elements of the tar file
        logging.debug(f'Pre-processing {folder_name}')
        tarfile.open(f'{folder_path}.tar').extractall(
            os.path.join(self._root_path, self._data_sub_path)
        )

        # Filter the files so we are not getting the same file twice (.hea and .dat)
        files = glob.glob(os.path.join(folder_path, '*.hea'))

        # Create new h5 file
        self.current_h5_file = h5py.File(f'{folder_path}.h5', 'w')
        # Create an h5py dataset with dimensions (num of entries, num of leads, sample rate)
        self.current_h5_dataset = self.current_h5_file.create_dataset('tracings', (
            len(files), self.NUMBER_OF_LEADS, self.resample_rate)
                                                                      )

        # Update remaining params
        self.last_folder_name = folder_name

    def _preprocess_data(self, row, to_delete, to_add):
        """Assuming the rows are sorted according to their group. This method will pre-process those each folder
        containing the data into an h5py file."""

        # Extract data from rows
        filename, folder_name, index_in_group = row

        # If the folder no longer exists then it has already been pre-processed. Catches restarted program.
        if not os.path.exists(os.path.join(self._root_path, self._data_sub_path, f'{folder_name}.tar')):
            return

        # If this is the first entry in the group, we must perform some special steps.
        if index_in_group == 0:
            self._process_folder(folder_name)

        # Pre-process ecg and write to the dataset at the given index
        try:
            self.current_h5_dataset[index_in_group, :, :] = self._pre_process_aux(**self.read_ecg_from_wfdb(filename))
        except Exception:
            # Catch any form of exception and just delete the row from the main dataset
            # No need to fix indices, etc. The rest is already handled
            to_delete.append(row.name)

    def _get_annotations(self):
        """Retrieves and massages annotations from CODE annotations file."""
        # Get the original annotations file
        original_annotations = pd.read_csv(os.path.join(self._root_path, self._annotations_file))
        # Generate a new column filled with the no label value
        original_annotations['label'] = self.NO_LABEL
        # Iterate over the label columns in the annotations and convert them from one-hot to a single column
        for key in self.labels.keys():
            original_annotations.loc[original_annotations[key] == 1, 'label'] = self.labels[key]
        # We only need the labels
        return original_annotations[['id_exam', 'label']]

    def _merge_annotations_and_file_paths(self, annotations, file_paths):
        """Returns two dataframes. The first containing the annotated data. The second containing the un-annotated
        data."""
        # Get the first occurence of a unique id. This is the annotated file.
        # Here we extract the unique id from each record name
        extract_ids = np.array(file_paths['filename'].str.split('TNMG').str[-1].str.split('_N').str[0], dtype=np.int64)
        # Get unique ids and return their index. Make sure to sort both, so they are in the correct order.
        unique_ids, unique_ids_index = np.unique(extract_ids, return_index=True)
        # Get the subset of files belonging to the first entry of each unique id
        files_subset = (file_paths.iloc[unique_ids_index, :]).reset_index()
        # Attach the unique id for merging
        files_subset['id_exam'] = unique_ids

        # We then merge, as not all unique ids are annotated themselves.
        annotated_data = files_subset.merge(annotations, on='id_exam', how='inner')
        # Get un-annotated files by dropping the files with annotations from the main dataframe
        unannotated_data = file_paths.drop(index=annotated_data['index']).reset_index(drop=True)

        # Get subset of rows we are interested in from annotated data
        annotated_data = annotated_data[['label', 'filename', 'group_id', 'index_in_group']]

        # Get the annotated rows without label and append them to the unannotated dataset while dropping them from
        # the annotated dataset
        index_no_label = annotated_data['label'].str.contains(self.NO_LABEL)
        unannotated_data = pd.concat([
            unannotated_data,
            annotated_data[index_no_label][['filename', 'group_id', 'index_in_group']]
        ],
            ignore_index=True)
        annotated_data = annotated_data[~index_no_label].reset_index(drop=True)

        return annotated_data, unannotated_data

    def _postprocess_data(self, **kwargs):
        def accumulate_annotated_data():
            # Create new h5 file
            annotated_h5_file = h5py.File(f'{self._annotated_h5py_path}.h5', 'w')
            # Create an h5py dataset with dimensions (num of entries, num of leads, sample rate)
            h5_dataset = annotated_h5_file.create_dataset('tracings', (
                len(self.annotated_data), self.NUMBER_OF_LEADS, self.resample_rate))

            # Read each ecg into the h5py file
            for i, (label, filename, group_id, index_in_group) in self.annotated_data.iterrows():
                h5_dataset[i, :, :] = self.read_ecg_from_h5py(group_id, index_in_group)
            # Close the writing hook
            annotated_h5_file.close()

            # Drop the additional columns in _data_y
            self.annotated_data = self.annotated_data[['label', 'filename']]
            # Overwrite the original csv file
            self.annotated_data.to_csv(self._modified_annotated_path, index=False)

        # For some reason this is required
        self._data_y['index_in_group'] = self._data_y['index_in_group'].astype(int)
        # Separate into annotated and un_annotated
        # Now combine these with the file paths
        self.annotated_data, self.unannotated_data = self._merge_annotations_and_file_paths(
            annotations=self._get_annotations(),
            file_paths=self._data_y
        )

        # Map label names to annotated data
        self.annotated_data = super().map_label_names(self.annotated_data, self._snomed_map)

        # Now save this as our modified annotated and modified unannotated file
        self.annotated_data.to_csv(self._modified_annotated_path, index=False)
        self.unannotated_data.to_csv(self._modified_unannotated_path, index=False)

        # Done preprocessing last folder. Delete last folder and archive
        if self.last_folder_name:
            self.delete_archive(
                folder_path=os.path.join(self._root_path, self._data_sub_path, self.last_folder_name),
                archive_path=os.path.join(self._root_path, self._data_sub_path, self.last_folder_name + '.tar'),
                name=self.last_folder_name
            )
        if self.current_h5_file:
            self.current_h5_file.close()
            logging.debug(f'Done pre-processing {self.last_folder_name}\n')

        # Get all h5 files
        h5_files = glob.glob(os.path.join(self._root_path, self._data_sub_path, '*.h5'))

        # Create a dictionary for pre-processing
        self.h5py_file_dict = {
            os.path.splitext(os.path.basename(filename))[0]: h5py.File(filename, 'r')['tracings']
            for filename in h5_files
        }

        # Accumulate all annotated data into a single h5py file
        accumulate_annotated_data()

        # Remove files and folders other than h5py and modified dataset folder
        temp_path = os.path.join(self._root_path, self._data_sub_path)
        required_files = [self.modified_preprocessing_file, self._modified_unannotated_path, temp_path,
                          self._modified_annotated_path, f'{self._annotated_h5py_path}.h5', self._modified_dataset_path]
        for file in [os.path.join(self._root_path, f) for f in os.listdir(self._root_path)]:
            if file not in required_files:
                if os.path.isfile(file):
                    os.remove(file)
                else:
                    shutil.rmtree(file)

        for file in [os.path.join(temp_path, f) for f in os.listdir(temp_path)]:
            if file not in h5_files:
                if os.path.isfile(file):
                    os.remove(file)
                else:
                    shutil.rmtree(file)

    def read_ecg_from_h5py(self, key, index):
        """Read ecg from h5py file"""
        return self.h5py_file_dict[key][index, :, :]

    @staticmethod
    def read_ecg_from_wfdb(path):
        """Read wfdb record"""
        record = wfdb.rdrecord(path)
        return {'signal': np.asarray(record.p_signal.T, dtype=np.float64), 'freq': record.fs, 'leads': record.sig_name}

    @staticmethod
    def _convert_labels_to_string(df_data, snomed_mappings):
        """As this dataset does not contain true labels, we just skip this step."""
        return df_data, snomed_mappings

    @staticmethod
    def delete_archive(folder_path, archive_path, name):
        try:
            shutil.rmtree(folder_path)
            logging.debug(f'Removed folder for {name}')
            os.remove(archive_path)
            logging.debug(f'Removed archive(tar) for {name}')
        except PermissionError as e:
            # Exception will arise if we don't have permission to delete the files
            logging.debug(f'Could not remove folder or archive(tar) for {name}')
            print(f'Error: {e}')
        except OSError as e:
            print(f'Error: {e}')

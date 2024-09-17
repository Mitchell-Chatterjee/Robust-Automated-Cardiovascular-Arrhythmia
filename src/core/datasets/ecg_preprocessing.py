import logging
import os
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd

from src.core.datasets.ecg_interface import EcgInterface


class EcgPreprocessing(ABC, EcgInterface):
    """
    Defines the base class for all ecg pre-processing
    """

    def __init__(self, root_path, data_sub_path, preprocessing):
        self._root_path = root_path
        self._data_sub_path = data_sub_path
        self._preprocessing = preprocessing

        self._data_y = None
        self._snomed_map = None

        self._modified_dataset_path = os.path.join(self._root_path, self.dataset_folder)
        self._dataset_path = os.path.join(self._modified_dataset_path, self.csv_path)

        # First we read in the data, then perform pre-processing, then initialize the class with additional parameters.
        logging.debug('Reading metadata phase')
        self.__read_data__()
        logging.debug('Done reading metadata phase\n\n')

        logging.debug('Pre-processing phase')
        self.__preprocess_data__()
        logging.debug('Done pre-processing phase\n\n')

    def map_label_names(self, df_data, snomed_mappings):
        """
        Function to merge the snomed mappings with the header data.

        Parameters
        ----------
        df_data: A dataframe containing the header data.
        snomed_mappings: A dataframe containing a legend of the CT codes and abbreviations.

        Returns
        -------
        A dataframe containing the merged data from snomed_mappings and df_data.
        """
        df_data['label'] = df_data['label'].str.strip()
        snomed_mappings['label'] = snomed_mappings['label'].str.strip()
        # Convert snomed mappings to a dictionary for quick conversion
        code_to_label_dict = {code: label for _, (code, label) in snomed_mappings[['Code', 'label']].iterrows()}
        # As the labels come in a list form, convert them to such
        df_data['label'] = df_data['label'].str.split(',')
        # We then have to transform them using the labels snomed mappings
        df_data['label'] = df_data['label'].apply(lambda labels:
                                                  list(set(elem for elem in [*map(code_to_label_dict.get, labels)]
                                                           if elem not in [None, np.nan])))

        # Remove empty rows
        df_data = df_data[~df_data['label'].isna()]
        df_data = df_data[df_data['label'].map(len) > 0]

        return df_data

    def __read_data__(self):
        # Get label legend
        snomed_mappings = pd.read_csv(self.snomed_path, dtype={'SNOMED CT Code': str})
        # Rename the column for snomed mappings
        snomed_mappings = snomed_mappings.rename(columns={'SNOMED CT Code': 'Code', 'Abbreviation': 'label'})

        # Create folder for hosting datasets
        if not os.path.exists(self._modified_dataset_path):
            os.mkdir(self._modified_dataset_path)

        if os.path.exists(self._dataset_path) and not self._preprocessing:
            df_data = pd.read_csv(self._dataset_path, dtype={'label': str})
        else:
            # Get the header data
            df_data = self.import_key_data(os.path.join(self._root_path, self._data_sub_path))

            # Map to label names
            df_data = self.map_label_names(df_data, snomed_mappings)

            # Write to a csv for future use
            df_data.to_csv(self._dataset_path, index=False)

        # Save snomed map
        self._snomed_map = snomed_mappings
        # Assign whole dataset for potential pre-processing by child classes. Prior to initialization.
        self._data_y = df_data

    def __preprocess_data__(self):
        """General method that applies a pre-processing function to each member of the dataset. Class specific
        action is enforced by abstract _preprocess_data method."""
        if self._preprocessing:
            # Pass by reference variables for adding or deleting data from the dataframe
            to_delete = []
            to_add = {col: [] for col in self._data_y.columns}

            # Run the pre-processing
            self._data_y.apply(lambda row: self._preprocess_data(row=row, to_delete=to_delete, to_add=to_add), axis=1)
            # Delete any data by dropping the row indexes
            self._data_y = self._data_y.drop(to_delete).reset_index(drop=True)
            # Add any additional data by concatenating the two dataframes
            self._data_y = pd.concat([self._data_y, pd.DataFrame(to_add)], ignore_index=True)

            # Overwrite the saved csv with our new dataset
            self._data_y.to_csv(self._dataset_path, index=False)
        # Additional post-processing step, if any classes seek to implement it
        self._postprocess_data()

    @abstractmethod
    def _preprocess_data(self, row, to_delete, to_add):
        """Class specific method that overrides the general call to pre-process data"""
        pass

    @abstractmethod
    def _postprocess_data(self, **kwargs):
        """Class specific method that override the general call to post-process data"""
        pass

    @abstractmethod
    def import_key_data(self, path):
        """Class specific method that overrides general call to import key data"""
        pass

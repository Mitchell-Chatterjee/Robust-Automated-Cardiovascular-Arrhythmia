import os
import random
from ast import literal_eval
from enum import Enum

import h5py
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from src.core.datasets.ecg_dataset import EcgDataset
from src.core.datasets.ecg_interface import Split, ClassificationType


class DiagnosticClass(Enum):
    ALL = 'all'
    SUPER = 'superclass'
    SUB = 'subclass'
    DIAG = 'diagnostic'
    FORM = 'form_only'
    RHYTHM = 'rhythm_only'
    STEMI = 'ST-ELEV-MI'


class PtbXlDataset(EcgDataset):
    """
    The PTB-XL dataset. More information available here:
    https://physionet.org/content/ptb-xl/1.0.3/

    The PTB-XL ECG dataset is a large dataset of 21799 clinical 12-lead ECGs from 18869 patients of 10 second length.
    Sampled at both 500Hz and 100Hz. This dataset is set up to use the 500Hz and downsample if required.

    The PTB-XL may contain multiple labels per each record. Making the classification task a multi-label classification
    task, as opposed to multi-class classification seen in many other ECG datasets.
    """
    train_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    val_indices = [9]
    test_indices = [10]

    def __init__(self, root_path, diagnostic_class: DiagnosticClass = DiagnosticClass.ALL,
                 classification_type=ClassificationType.MULTI_LABEL, **dataset_kwargs):
        """
        custom_class_selection: This parameter expects a list of class names to be passed in. Allowing the user to
        manually select the classes they wish to include in classification. If the keyword 'Other' is among the names
        in the list. All remaining classes not present in the list will be aggregated into the 'Other' class.
            example: custom_class_selection=['MI', 'AFIB', 'Other'] --> dataset will contain classes 'MI', 'AFIB', 'Other'

        random_stratification: If true. The dataset will randomly assign the groups to training (8/10 groups),
        validation (1/10 groups) and testing (1/10 groups). Otherwise, this will be done deterministically with
        groups 1-8 going to training, group 9 used for validation and group 10 used for testing.
        """
        self._diagnostic_class = diagnostic_class

        if self._diagnostic_class == DiagnosticClass.STEMI:
            classification_type = ClassificationType.BINARY

        # Open the reading hook
        self._h5_path = os.path.join(root_path, f"{self.__class__.__name__}.h5")
        self._h5_file = h5py.File(self._h5_path, 'r')['tracings']

        super().__init__(root_path=root_path, classification_type=classification_type, **dataset_kwargs)

    @property
    def h5_file(self):
        return self._h5_file

    def __len__(self):
        return len(self._data_y)

    def preprocess_labeled_data(self, label_encoder, n_hot_vector):
        """Will require different rules if implemented as this is multi-label."""
        return None, n_hot_vector, label_encoder

    def get_data_subset_annotated(self, df_indices=None):
        """
        Override the parent method from ecg_dataset. This class already has suggested stratification folds that we will
        use for sampling.
        """
        if self._split == Split.TRAIN:
            train_indices = self._data_y[self._data_y['strat_fold'].isin(self.train_indices)].index.values
            return self._data_y.loc[train_indices, :].reset_index(drop=True), self._n_hot_vector[train_indices]
        elif self._split == Split.VALIDATION:
            val_indices = self._data_y[self._data_y['strat_fold'].isin(self.val_indices)].index.values
            return self._data_y.loc[val_indices, :].reset_index(drop=True), self._n_hot_vector[val_indices]
        elif self._split == Split.TEST:
            test_indices = self._data_y[self._data_y['strat_fold'].isin(self.test_indices)].index.values
            return self._data_y.loc[test_indices, :].reset_index(drop=True), self._n_hot_vector[test_indices]

    def get_diagnostic_subset(self):
        # Convert label to diagnostic class
        if self._diagnostic_class != DiagnosticClass.ALL:
            self._data_y = self._data_y.rename(columns={'label': 'all', self._diagnostic_class.value: 'label'})

        # Essentially just remove any rows without values
        return self._data_y[self._data_y['label'].map(lambda labels: len(literal_eval(labels))) > 0]\
            .reset_index(drop=True)

    def initialize_label_encoder_multi_label(self):
        """
        Override parent function as we will need to remove columns from the one-hot vector depending on the diagnostic
        level we are using according to the Diagnostic Class.
        """
        # Get diagnostic subset first
        self._data_y = self.get_diagnostic_subset()
        # Get one hot encoding of everything
        multi_label_encoder = MultiLabelBinarizer()
        # Concatenate all labels into a single list and run fit_transform
        labels = self._data_y['label'].apply(lambda val: literal_eval(val))
        return multi_label_encoder, multi_label_encoder.fit_transform(labels.tolist())

    def __initialize_multi_class__(self):
        """
        Override the parent function. This will be used for binary classification.
        """
        # Get diagnostic class
        if self._diagnostic_class != DiagnosticClass.ALL:
            self._data_y = self._data_y.rename(columns={'label': 'all', self._diagnostic_class.value: 'label'})

        # Convert binary classes to labels
        self._data_y['label'] = self._data_y['label'].astype(int)
        self._data_y['label'] = self._data_y['label'].map({0: 'NORM', 1: self._diagnostic_class.name})

        # Initialize the label encoder
        self._label_encoder, self._n_hot_vector = self.initialize_label_encoder_multi_class()
        # Get current data subset: train, validation, test, and the corresponding labels
        self._data_y, self._n_hot_vector = self.get_data_subset_annotated(None)

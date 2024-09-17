import copy
import os
import random
from abc import ABC

import logging
from ast import literal_eval
from collections import Counter

import kornia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from skmultilearn.model_selection import IterativeStratification
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from src.core.datasets.ecg_interface import Split, EcgInterface, LeadSelection, ClassificationType
from src.core.constants.definitions import DataAugmentation
from src.core.support.transforms import transformations_from_strings
from src.core.utils.basics import _torch, _torch_single


class EcgDataset(ABC, Dataset, EcgInterface):
    """
    Defines the base class for all ecg datasets.
    """

    # Static variables when performing iterative stratification
    train_index, val_index, test_index = None, None, None

    def __init__(self, root_path, min_class_size, top_n_classes, custom_lead_selection, focal_loss=False,
                 focal_alpha=0.25, split=Split.TRAIN, custom_class_selection=None,
                 classification_type=ClassificationType.MULTI_LABEL,
                 transformations=None, t_params=None, reset_strat_folds=False,
                 data_augmentation=DataAugmentation.none, chunk_size=250, chunk_step=125, alt_lead_ordering=False):
        """
        custom_class_selection: This parameter expects a list of class names to be passed in. Allowing the user to
        manually select the classes they wish to include in classification. If the keyword 'Other' is among the names
        in the list. All remaining classes not present in the list will be aggregated into the 'Other' class.
            example: custom_class_selection=['MI', 'AFIB', 'Other'] --> dataset will contain classes 'MI', 'AFIB', 'Other'
        """
        assert isinstance(split, Split)
        self._split = split

        self._root_path = root_path

        self._modified_dataset_path = os.path.join(self._root_path, self.dataset_folder)
        self._dataset_path = os.path.join(self._modified_dataset_path, self.csv_path)

        self._data_y = pd.read_csv(self._dataset_path, dtype={'label': str})
        self._label_encoder, self._n_hot_vector = None, None
        self._reset_strat_folds = reset_strat_folds

        # Specifies either the minimum class size or the top-n classes the algorithm will consider.
        # Top-n takes precedence if present.
        self._min_class_size = min_class_size
        self._top_n_classes = top_n_classes

        # Specifies the classification type and whether we will be using specific classes.
        self._custom_class_classification = custom_class_selection
        self._classification_type = classification_type

        # Specifies the leads to be used for classification
        self._custom_lead_selection = sorted(self.transform_lead_selection(custom_lead_selection))
        self._alt_lead_ordering = alt_lead_ordering

        # Focal loss terms
        self._focal_loss = focal_loss
        self._focal_alpha = focal_alpha

        # Test time augmentation
        self._data_augmentation = data_augmentation
        self._chunk_size = chunk_size
        self._chunk_step = chunk_step

        if self._data_augmentation is DataAugmentation.pre_train_cpc:
            self._data_y = self._data_y.loc[list(range(round(len(self._data_y) // 8))), :]

        # Initialize the class with additional parameters.
        logging.debug('Initialization phase')
        self.__initialize_class__()
        logging.debug('Done initialization phase\n\n')

        # Data Augmentations (a bit hacky, setting transformations to none is required so we can get item for length)
        self.transformations = None
        if len(self._data_y) > 0 and self._split == Split.TRAIN:
            self.transformations = transformations_from_strings(transformations, t_params,
                                                                len(self._custom_lead_selection),
                                                                self.__getitem__(0)[0].shape[1])

    def __getitem__(self, index) -> T_co:
        original_index = self._data_y['original_index'][index]
        ecg_data = self._h5_file[original_index, :, :]  # need original index when retrieving from dataset

        # Select the leads we are using
        ecg_data = _torch_single(ecg_data[self._custom_lead_selection])

        # Change the ordering of the leads
        if self._alt_lead_ordering:
            lead_mapping = [EcgInterface.all_leads.index(elem) for elem in EcgInterface.alt_order]
            ecg_data[:, :] = ecg_data[lead_mapping, :]

        # dataframe index when retrieving from labels
        if self._classification_type == ClassificationType.PRETRAIN:
            label = copy.deepcopy(ecg_data)
        else:
            label = np.asarray(self._n_hot_vector[index])

        # Randomly apply transformations to the data (For now only one)
        if self.transformations:
            ecg_data, label = _torch(np.random.choice(self.transformations, 1)[0]((ecg_data, label)))

        if self._data_augmentation is DataAugmentation.test_time_aug_cpc:
            if self._split == Split.TRAIN:
                # Take random split from signal
                start_idx = np.random.randint(0, ecg_data.shape[1] - self._chunk_size)
                ecg_data = ecg_data[:, start_idx:start_idx + self._chunk_size]
            else:
                # Test time augmentation
                ecg_data = ecg_data.unfold(dimension=1, size=self._chunk_size, step=self._chunk_step)

        # Return raw ecg signal as it will be filtered and moved to gpu by patch masker
        return _torch((ecg_data, label))

    @property
    def data_y(self):
        return self._data_y

    @property
    def n_classes(self):
        if self._label_encoder is None:
            return 0
        if self._classification_type == ClassificationType.BINARY:
            return 1
        return len(self._label_encoder.classes_)

    @property
    def classes(self):
        return self._label_encoder.classes_

    @property
    def loss_func(self):
        return self._loss_func

    @property
    def label_type(self):
        return self._label_type

    @property
    def classification_type(self):
        return self._classification_type

    def transform_lead_selection(self, custom_lead_selection):
        # Parse and remove repetitions
        lead_selection = set([elem.strip() for elem in custom_lead_selection.split(',')])

        if LeadSelection.EIGHT_LEADS.value in lead_selection:
            return [self.ecg_dict[key] for key in self.eight_leads]
        elif LeadSelection.ALL_LEADS.value in lead_selection:
            return [self.ecg_dict[key] for key in self.all_leads]
        else:
            # Ensure all elements are contained within our list of leads
            assert lead_selection.issubset(self.all_leads), f'All leads must be contained within {self.all_leads}'
            return [self.ecg_dict[key] for key in lead_selection]

    def initialize_label_encoder_multi_class(self):
        # Get one hot encoding of everything
        label_encoder = LabelEncoder()
        # We only fit to the classes present in this dataset. After removing undersized classes as well
        label_encoder = label_encoder.fit(self._data_y['label'].unique())
        # Then we transform
        return label_encoder, label_encoder.transform(self._data_y['label'].values)

    def initialize_label_encoder_multi_label(self):
        # Get one hot encoding of everything
        multi_label_encoder = MultiLabelBinarizer()
        # Concatenate all labels into a single list and run fit_transform
        labels = self._data_y['label'].apply(lambda val: literal_eval(val))
        return multi_label_encoder, multi_label_encoder.fit_transform(labels.tolist())

    def get_data_subset_unannotated(self):
        """As we are not concerned with the true labels, we can simply separate the values randomly"""
        if 'strat_fold_unannotated' not in self._data_y.columns or self._reset_strat_folds:
            indices = np.random.permutation(self._data_y.shape[0])
            self._data_y.loc[indices[:round(len(indices) * 0.8)], 'strat_fold_unannotated'] = Split.TRAIN.value
            self._data_y.loc[indices[round(len(indices) * 0.8):], 'strat_fold_unannotated'] = Split.VALIDATION.value

            # Save the stratification for daisy-chained runs
            self._data_y.to_csv(path_or_buf=self._dataset_path, index=False,
                                columns=[column for column in self._data_y.columns if column != 'original_index'])

        if self._split == Split.ALL:
            # Handles the case where we simply want to merge datasets together
            return self._data_y
        elif self._split == Split.VALIDATION:
            return self._data_y[self._data_y['strat_fold_unannotated'] == Split.VALIDATION.value].reset_index(drop=True)
        elif self._split == Split.TEST:
            # For now we do not include a test set as this is simply SSRL pre-training
            return []
        return self._data_y[self._data_y['strat_fold_unannotated'] == Split.TRAIN.value].reset_index(drop=True)

    def get_data_subset_annotated(self, df_indices):
        """
        This method splits the dataset into proportionally balanced folds. Then returns the correct split (train,
        validation or test) for the current dataset.

        In the case of multi-class we use stratified shuffle split. Multi-label uses Iterative Stratification.

        There is a bit of a hack here. As there is no way to ensure the iterative stratification split is deterministic.
        We will instead set the train, test, and validation indices only once. These will be static values shared
        between all instances of the class.

        Returns
        -------
        A subset of the total dataframe, corresponding to the train, validation or test split.
        """
        if 'strat_fold_annotated' not in self._data_y.columns or self._reset_strat_folds:
            if self._classification_type == ClassificationType.MULTI_LABEL:
                train_test_stratifier = IterativeStratification(n_splits=5, order=1)
                test_validation_stratifier = IterativeStratification(n_splits=2, order=1)
            else:
                # Multi-class case
                train_test_stratifier = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
                test_validation_stratifier = StratifiedShuffleSplit(n_splits=5, test_size=0.5)

            # Continue generating splits until we get one with at least one label from each class in train, valid, test
            flag, impatience = False, 5
            while impatience > 0 and not flag:
                df_data = self._data_y.loc[df_indices, :].reset_index(drop=True)
                n_hot_vector = self._n_hot_vector[df_indices, :]

                train_index, test_val = next(train_test_stratifier.split(df_data, n_hot_vector))

                # Get the test/validation subset from the original dataframe and n_hot vector
                test_val_data_subset = df_data.loc[test_val, :].reset_index(drop=True)
                test_val_n_hot_subset = n_hot_vector[test_val]

                validation_index, test_index = next(test_validation_stratifier.split(test_val_data_subset,
                                                                                     test_val_n_hot_subset))
                # Set values in dataframe
                train_index = df_data.loc[train_index, :]['original_index'].to_list()
                val_index = test_val_data_subset.loc[validation_index, :]['original_index'].to_list()
                test_index = test_val_data_subset.loc[test_index, :]['original_index'].to_list()

                impatience -= 1
                flag = np.all(self._n_hot_vector[train_index].sum(axis=0) > 0) and \
                       np.all(self._n_hot_vector[val_index].sum(axis=0) > 0) and \
                       np.all(self._n_hot_vector[test_index].sum(axis=0) > 0)

            if not flag:
                raise Exception("Could not find acceptable train, valid, test split after 5 runs. "
                                "(Increase minimum class size).")

            # Save stratification
            self._data_y['strat_fold_annotated'] = np.nan
            self._data_y.loc[train_index, 'strat_fold_annotated'] = Split.TRAIN.value
            self._data_y.loc[val_index, 'strat_fold_annotated'] = Split.VALIDATION.value
            self._data_y.loc[test_index, 'strat_fold_annotated'] = Split.TEST.value
            self._data_y.to_csv(path_or_buf=self._dataset_path, index=False,
                                columns=[column for column in self._data_y.columns if column != 'original_index'])

        # Return the correct subset based on our current split
        if self._split == Split.VALIDATION:
            val_index = self._data_y['strat_fold_annotated'] == Split.VALIDATION.value
            return self._data_y[val_index].reset_index(drop=True), self._n_hot_vector[val_index]
        elif self._split == Split.TEST:
            test_index = self._data_y['strat_fold_annotated'] == Split.TEST.value
            return self._data_y[test_index].reset_index(drop=True), self._n_hot_vector[test_index]
        train_index = self._data_y['strat_fold_annotated'] == Split.TRAIN.value
        return self._data_y[train_index].reset_index(drop=True), self._n_hot_vector[train_index]

    def get_class_sizes(self):
        if self._classification_type is ClassificationType.MULTI_LABEL:
            return pd.DataFrame({'label': self._label_encoder.classes_, 'count': self._n_hot_vector.sum(axis=0)})
        else:
            return pd.DataFrame({'label': self._label_encoder.classes_, 'count': Counter(self._n_hot_vector).values()})

    def plot_classes(self, plot_path):
        """
        This method plots the count for each class in bar graph. Allowing us to see the class imbalance.

        Returns
        -------
        None
        """
        counts_df = self.get_class_sizes()

        plt.figure(figsize=(30, 20))
        plt.bar(x=counts_df['label'], height=counts_df['count'])
        plt.title(f"Distribution of Diagnosis (Split={self._split})", color="black", fontsize=60)
        plt.tick_params(axis="both", colors="black")
        plt.xlabel("Diagnosis", color="black", fontsize=50)
        plt.ylabel("Count", color="black", fontsize=50)
        plt.xticks(rotation=90, fontsize=30)
        plt.yticks(fontsize=30)

        for x, y in zip(counts_df['label'], counts_df['count']):
            plt.text(x, y, str(y), fontsize=40, rotation=90)

        plt.savefig(f'{plot_path}.png')
        plt.close()

    def print_class_sizes(self):
        counts_df = self.get_class_sizes()
        print(self._split)
        print(counts_df, end='\n\n')

    def multi_label_to_multi_class(self):
        """
        We leave this very simple implementation where we only select the first label for each row.
        Based on future purposes this method can be made more complex to handle filtering for particular classes.
        """
        self._data_y['label'] = self._data_y['label'].apply(lambda labels: list(literal_eval(str(labels)))[0])

    def custom_class_classification(self):
        # Flip null values to other
        self._data_y.loc[self._data_y['label'].isnull(), 'label'] = 'Other'

        self._custom_class_classification = [elem.strip() for elem in self._custom_class_classification.split(',')]

        # Get the columns with labels not including other
        comp_str = '|'.join(self._custom_class_classification)
        indicies = self._data_y['label'].str.contains(comp_str)

        # Flip the labels in the rows not contained in indicies to 'Other'
        self._data_y.loc[~indicies, 'label'] = 'Other'

        # If the other column is contained in the list then return the whole thing
        if 'other' in [elem.lower() for elem in self._custom_class_classification]:
            return self._data_y
        # Otherwise only return the selected indicies
        return self._data_y[indicies].reset_index(drop=True)

    def preprocess_labeled_data(self, label_encoder, n_hot_vector):
        """
        This method removes classes which do not have a sufficient number of labelled samples in the multi-labelled
        case. So we can perform stratification.

        This method will return only those labels with more instances than a given threshold. This will then be handled
        in the method for instantiating the multi-hot vector.
        """
        # Get columns with at least the minimum threshold
        columns = np.where(n_hot_vector.sum(axis=0) >= self._min_class_size)[0]

        # Get n_hot vector subset
        n_hot_vector = n_hot_vector[:, columns]
        # Get only those labels from the label encoder
        label_encoder.classes_ = label_encoder.classes_[columns]
        # Get this subset of columns from n_hot_vector and check which rows contain at least 1-label
        df_indices = np.where(n_hot_vector.sum(axis=1) > 0)[0]

        # Remove indices from dataframe and n_hot vector
        return df_indices, n_hot_vector, label_encoder

    def get_loss(self):
        if self._focal_loss and (self._classification_type in [ClassificationType.BINARY, ClassificationType.MULTI_LABEL]):
            return kornia.losses.BinaryFocalLossWithLogits(alpha=self._focal_alpha, reduction='mean'), torch.float
        elif self._classification_type in [ClassificationType.BINARY, ClassificationType.MULTI_LABEL]:
            return torch.nn.BCEWithLogitsLoss(reduction='mean'), torch.float
        elif self._classification_type == ClassificationType.MULTI_CLASS:
            return torch.nn.CrossEntropyLoss(reduction='mean'), torch.float
        elif self._classification_type == ClassificationType.PRETRAIN:
            return torch.nn.MSELoss(reduction='mean'), torch.float
        else:
            return torch.nn.BCEWithLogitsLoss(reduction='mean'), torch.float

    def __initialize_class__(self):
        # Put the index inside the dataframe, in modify it by dropping rows
        self._data_y = self._data_y.reset_index(drop=False, names=['original_index'])

        if self._classification_type == ClassificationType.PRETRAIN:
            self._loss_func, self._label_type = self.get_loss()
            self._data_y = self.get_data_subset_unannotated()
        elif self._classification_type == ClassificationType.MULTI_LABEL:
            self._loss_func, self._label_type = self.get_loss()
            self.__initialize_multi_label__()
        elif self._classification_type == ClassificationType.MULTI_CLASS:
            self._loss_func, self._label_type = self.get_loss()
            self.__initialize_multi_class__()
        elif self._classification_type == ClassificationType.BINARY:
            self._loss_func, self._label_type = self.get_loss()
            self.__initialize_multi_class__()

    def __initialize_multi_class__(self):
        # Select one label from multi-label set
        self.multi_label_to_multi_class()
        # Initialize custom class classification if specified
        # self._data_y = self.custom_class_classification() if self._custom_class_classification else self._data_y
        # Initialize label encoder
        label_encoder, n_hot_vector = self.initialize_label_encoder_multi_class()
        # Remove undersized classes prior
        df_indices, self._n_hot_vector, self._label_encoder = \
            self.preprocess_labeled_data(label_encoder=label_encoder, n_hot_vector=n_hot_vector)
        # Get current data subset: train, validation, test, and the corresponding labels
        self._data_y, self._n_hot_vector = self.get_data_subset_annotated(df_indices)

    def __initialize_multi_label__(self):
        # Initialize label encoder
        label_encoder, n_hot_vector = self.initialize_label_encoder_multi_label()
        # Remove undersized classes prior
        df_indices, self._n_hot_vector, self._label_encoder = \
            self.preprocess_labeled_data(label_encoder=label_encoder, n_hot_vector=n_hot_vector)
        # Get current data subset: train, validation, test, and the corresponding labels
        self._data_y, self._n_hot_vector = self.get_data_subset_annotated(df_indices)

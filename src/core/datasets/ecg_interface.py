import os
from enum import Enum

from src.core.constants.definitions import CODE_DIR


class Split(Enum):
    ALL = 0
    TRAIN = 1
    TEST = 2
    VALIDATION = 3


class LeadSelection(Enum):
    ALL_LEADS = 'all_leads'
    EIGHT_LEADS = 'eight_leads'


class ClassificationType(Enum):
    MULTI_CLASS = 0
    MULTI_LABEL = 1
    BINARY = 2
    PRETRAIN = 3


class EcgInterface:
    """
    High-level interface for ECG datasets. Meant to store common terms
    """

    # Seed value for reproducibility of stratification during split into train, validation and testing sets.
    stratification_seed = 200
    # Resample frequency
    resample_freq = 500
    # Time window for resampling
    time_window = 10
    # Resample rate. Currently set to 500Hz for a 10 second time window.
    resample_rate = 5000
    # Number of leads
    NUMBER_OF_LEADS = 12
    # Constant for segmented data
    SEGMENTED = 'segmented'
    # default folder for dataset files
    dataset_folder = 'modified_dataset'
    # default csv path
    csv_path = 'modified_dataset.csv'
    # path to snomed ct codes
    snomed_path = os.path.join(CODE_DIR, 'core/constants/SNOMED_mappings.csv')

    # Lead selection
    eight_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    all_leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    alt_order = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'AVR', 'AVL', 'AVF']
    ecg_dict = {lead: index for (index, lead) in enumerate(all_leads)}

    # These following parameters can be used for any datasets that require additional pre-processing (inlcuding CODE).
    # Place to save list of h5py files for reference
    h5py_file_df = 'h5py_file_df.csv'

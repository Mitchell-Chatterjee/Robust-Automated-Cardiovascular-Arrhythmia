import argparse
import logging
import os

from src.core.constants.definitions import CODE_DIR
from src.core.datasets.chapman_dataset.chapman_dataset import ChapmanDataset
from src.core.datasets.chapman_dataset.chapman_preprocessing import ChapmanPreprocessing
from src.core.datasets.cinc_2020_dataset.cinc_2020_dataset import Cinc2020Dataset
from src.core.datasets.cinc_2020_dataset.cinc_2020_preprocessing import Cinc2020Preprocessing
from src.core.datasets.code_dataset.code_dataset_annotated import CodeDatasetAnnotated
from src.core.datasets.code_dataset.code_preprocessing import CodePreprocessing
from src.core.datasets.ecg_interface import Split, EcgInterface, ClassificationType
from src.core.datasets.noise.ecg_noise_preprocessing import EcgNoisePreprocessing
from src.core.datasets.ptb_xl.ptb_xl_dataset import PtbXlDataset
from src.core.datasets.ptb_xl.ptb_xl_preprocessing import PtbXlPreprocessing
from src.core.datasets.unified_annotated_dataset.unified_dataset import UnifiedDataset
from src.core.datasets.unified_annotated_dataset.unified_preprocessing import UnifiedPreprocessing

parser = argparse.ArgumentParser()
# Datasets
parser.add_argument('--root_path_chapman', type=str, help='root file path for the Chapman dataset',
                    default='/home/student/Datasets/100Hz/chapman/')
parser.add_argument('--root_path_cinc', type=str, help='root file path for the Cinc-2020 dataset',
                    default='/home/student/Datasets/100Hz/cinc-2020/')
parser.add_argument('--root_path_code', type=str, help='root file path for the CODE dataset',
                    default='/home/student/Datasets/CODE/')
parser.add_argument('--root_path_ptb_xl', type=str, help='root path for PTB-XL dataset',
                    default='/home/student/Datasets/PTB-XL/')

parser.add_argument('--data_sub_path_chapman', type=str, help='data sub path for the Chapman dataset',
                    default='WFDBRecords')
parser.add_argument('--data_sub_path_cinc', type=str, help='data sub path for the Cinc-2020 dataset',
                    default='training')
parser.add_argument('--data_sub_path_code', type=str, help='data sub path for the CODE dataset',
                    default='training')
parser.add_argument('--data_sub_path_ptb_xl', type=str, help='data sub path for the CODE dataset',
                    default='records500')

parser.add_argument('--snomed_mapping_path', type=str, help='map for snomed labels to abbreviation',
                    default='core/constants/SNOMED_mappings.csv')

parser.add_argument('--root_path_unified', type=str, help='root file path for the new unified dataset',
                    default='/home/student/Datasets/100Hz/Unified_Dataset')
parser.add_argument('--root_path_noise_dataset', type=str, help='root path to the noise dataset',
                    default='/home/student/Datasets/500Hz/noise')

# Whether  to include these datasets in pre-processing and unification
parser.add_argument('--cinc', action=argparse.BooleanOptionalAction, default=False,
                    help='Whether to include cinc-2020 dataset in pre-processing and unification')
parser.add_argument('--chapman', action=argparse.BooleanOptionalAction, default=False,
                    help='Whether to include chapman dataset in pre-processing and unification')
parser.add_argument('--code', action=argparse.BooleanOptionalAction, default=False,
                    help='Whether to include CODE dataset in pre-processing and unification')
parser.add_argument('--ptb_xl', action=argparse.BooleanOptionalAction, default=False,
                    help='Whether to include PTB-XL dataset in pre-processing and unification')
parser.add_argument('--noise', action=argparse.BooleanOptionalAction, default=False,
                    help='Whether to include the noise dataset in pre-processing')

# Data specific arguments
parser.add_argument('--resample_freq', type=int, default=100,
                    help='The sampling frequency to use for resampling the signals. Default 500Hz.')
parser.add_argument('--time_window', type=int, default=10,
                    help='The time window size to use for resampling the signals. Default 10 seconds.')
parser.add_argument('--remove_ptb_xl_from_cinc', action=argparse.BooleanOptionalAction, default=False,
                    help='Whether to include the ptb-xl dataset in the CINC-2020 dataset when unifying.')
parser.add_argument('--remove_segmented_from_cinc', action=argparse.BooleanOptionalAction, default=False,
                    help='Whether to include the segmented data in the CINC-2020 dataset when unifying.')

# Other methods
parser.add_argument('--method', type=str, help='method for unification (preprocess, unify, both)',
                    default='both')
parser.add_argument('--log_level', type=str, default='DEBUG', help='log level')

params = parser.parse_args()
print('args:', params, end='\n\n')


def update_ecg_interface():
    EcgInterface.resample_freq = params.resample_freq
    EcgInterface.time_window = params.time_window
    EcgInterface.resample_rate = params.resample_freq * params.time_window
    EcgInterface.snomed_path = os.path.join(CODE_DIR, params.snomed_mapping_path)

    logging.debug(f'The resample frequency is: {EcgInterface.resample_freq}Hz')
    logging.debug(f'The time window is: {EcgInterface.time_window}s')
    logging.debug(f'This gives a resample rate (signal length) of: {EcgInterface.resample_rate}')
    logging.debug(f'The mapping from snomed codes to labels will be taken from: {EcgInterface.snomed_path}')


def preprocess_dataset():
    # Instantiate the datasets
    if params.chapman:
        logging.debug('Preprocessing Chapman dataset')
        ChapmanPreprocessing(root_path=params.root_path_chapman, data_sub_path=params.data_sub_path_chapman,
                             preprocessing=True)
        logging.debug('Done preprocessing Chapman dataset')

    if params.cinc:
        logging.debug('Preprocessing CINC dataset')
        Cinc2020Preprocessing(root_path=params.root_path_cinc, data_sub_path=params.data_sub_path_cinc,
                              preprocessing=True)
        logging.debug('Done preprocessing CINC dataset')

    if params.ptb_xl:
        logging.debug('Preprocessing PTB-XL dataset.')
        PtbXlPreprocessing(root_path=params.root_path_ptb_xl, data_sub_path=params.data_sub_path_ptb_xl,
                           preprocessing=True)
        logging.debug('Done preprocessing PTB-XL dataset')

    if params.code:
        logging.debug('Preprocessing CODE annotated dataset')
        CodePreprocessing(root_path=params.root_path_code, data_sub_path=params.data_sub_path_code,
                          preprocessing=True)
        logging.debug('Done preprocessing CODE annotated dataset')

    if params.noise:
        logging.debug('Preprocessing MIT-BIH Noise Stress Test dataset')
        EcgNoisePreprocessing(root_path=params.root_path_noise_dataset)
        logging.debug('Done preprocessing MIT-BIH Noise Stress Test dataset')


def unify_datasets():
    logging.debug('Unifying datasets\n\n')

    datasets = []
    h5_name = f'{UnifiedDataset.__name__}.h5'
    # Prepare datasets
    if params.chapman:
        logging.debug('Creating Chapman dataset')
        # Pretrain classification type just ensures the initialization doesn't happen
        chapman_dataset = ChapmanDataset(root_path=params.root_path_chapman, min_class_size=None,
                                         top_n_classes=None, split=Split.ALL,
                                         custom_lead_selection='all_leads',
                                         classification_type=ClassificationType.PRETRAIN)
        datasets.append(chapman_dataset)

    if params.cinc:
        logging.debug('Creating CINC dataset')
        # Pretrain classification type just ensures the initialization doesn't happen
        cinc_dataset = Cinc2020Dataset(root_path=params.root_path_cinc, min_class_size=None,
                                       top_n_classes=None, split=Split.ALL,
                                       custom_lead_selection='all_leads',
                                       classification_type=ClassificationType.PRETRAIN,
                                       remove_ptb_xl=params.remove_ptb_xl_from_cinc,
                                       remove_segmented_data=params.remove_segmented_from_cinc)
        datasets.append(cinc_dataset)

    if params.ptb_xl:
        logging.debug('Creating PTB-XL dataset')
        # Pretrain classification type just ensures the initialization doesn't happen
        ptb_xl_dataset = PtbXlDataset(root_path=params.root_path_ptb_xl, min_class_size=None,
                                      top_n_classes=None, split=Split.ALL,
                                      custom_lead_selection='all_leads',
                                      classification_type=ClassificationType.PRETRAIN)
        h5_name = f'{PtbXlDataset.__name__}.h5'
        datasets.append(ptb_xl_dataset)

    if params.code:
        logging.debug('Creating CODE annotated dataset')
        # Pretrain classification type just ensures the initialization doesn't happen
        code_annotated_dataset = CodeDatasetAnnotated(root_path=params.root_path_code,
                                                      min_class_size=None, top_n_classes=None, split=Split.TRAIN,
                                                      custom_lead_selection='all_leads',
                                                      classification_type=ClassificationType.PRETRAIN)
        datasets.append(code_annotated_dataset)

    # Now unify the datasets into one
    logging.debug('Creating Unified dataset')
    UnifiedPreprocessing(root_path=params.root_path_unified, data_sub_path='training',
                         datasets=datasets, h5_name=h5_name)
    logging.debug('Done unifying datasets')


if __name__ == '__main__':
    numeric_level = getattr(logging, params.log_level.upper(), None)
    assert isinstance(numeric_level, int), f'Invalid log level: {params.log_level}'
    assert params.method in ['preprocess', 'unify', 'both']

    # Set logging level
    logging.basicConfig(level=numeric_level)

    # Execute
    update_ecg_interface()
    if params.method == 'preprocess':
        preprocess_dataset()
    elif params.method == 'unify':
        unify_datasets()
    elif params.method == 'both':
        preprocess_dataset()
        unify_datasets()

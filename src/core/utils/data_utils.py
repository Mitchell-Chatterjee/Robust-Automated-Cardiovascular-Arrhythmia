# Command line imports
from src.core.constants.definitions import DataAugmentation
from src.core.datasets.ptb_xl.ptb_xl_dataset import PtbXlDataset
from src.core.utils.data_module import DataLoaders
from src.core.datasets.chapman_dataset.chapman_dataset import ChapmanDataset
from src.core.datasets.cinc_2020_dataset.cinc_2020_dataset import Cinc2020Dataset
from src.core.datasets.code_dataset.code_dataset_annotated import CodeDatasetAnnotated
from src.core.datasets.code_dataset.code_dataset_unannotated import CodeDatasetUnannotated
from src.core.datasets.unified_annotated_dataset.unified_dataset import UnifiedDataset
from src.core.utils.filters import ButterworthFilter, MedianFilter

DSETS = ['chapman', 'cinc-2020', 'ptb-xl', 'CODE_Annotated', 'CODE_Unannotated', 'unified']


def get_dls(params):

    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params, 'use_time_features'):
        params.use_time_features = False

    dataset_kwargs = {
                        'root_path': params.root_path,
                        'min_class_size': params.min_class_size,
                        'top_n_classes': params.top_n_classes,
                        'classification_type': params.classification_type,
                        'custom_lead_selection': params.custom_lead_selection,
                        'focal_loss': params.focal_loss,
                        'focal_alpha': params.focal_alpha,
                        'alt_lead_ordering': params.alt_lead_ordering
    }
    if hasattr(params, 'custom_class_selection'):
        dataset_kwargs['custom_class_selection'] = params.custom_class_selection

    if hasattr(params, 'data_augmentation') and params.data_augmentation is DataAugmentation.test_time_aug_cpc:
        dataset_kwargs['data_augmentation'] = params.data_augmentation
        dataset_kwargs['chunk_size'] = params.chunk_size
        dataset_kwargs['chunk_step'] = params.chunk_step

    if hasattr(params, 'data_augmentation') and params.data_augmentation is DataAugmentation.pre_train_cpc:
        dataset_kwargs['data_augmentation'] = params.data_augmentation

    if hasattr(params, 'trafos'):
        dataset_kwargs['transformations'] = params.trafos
        dataset_kwargs['t_params'] = {key: value for key, value in params.__dict__.items() if key.startswith('t_')}

    if params.dset == 'CODE_Annotated':

        dls = DataLoaders(
            datasetCls=CodeDatasetAnnotated,
            dataset_kwargs=dataset_kwargs,
            batch_size=params.batch_size,
            workers=params.num_workers,
            reset_strat_folds=params.reset_strat_folds,
            distributed=params.distributed
        )

    elif params.dset == 'CODE_Unannotated':
        dataset_kwargs['data_sub_path'] = params.data_sub_path

        dls = DataLoaders(
            datasetCls=CodeDatasetUnannotated,
            dataset_kwargs=dataset_kwargs,
            batch_size=params.batch_size,
            workers=params.num_workers,
            reset_strat_folds=params.reset_strat_folds,
            distributed=params.distributed
        )

    elif params.dset == 'cinc-2020':
        dls = DataLoaders(
            datasetCls=Cinc2020Dataset,
            dataset_kwargs=dataset_kwargs,
            batch_size=params.batch_size,
            workers=params.num_workers,
            reset_strat_folds=params.reset_strat_folds,
            distributed=params.distributed
        )

    elif params.dset == 'ptb-xl':
        dataset_kwargs['diagnostic_class'] = params.diagnostic_class
        dls = DataLoaders(
            datasetCls=PtbXlDataset,
            dataset_kwargs=dataset_kwargs,
            batch_size=params.batch_size,
            workers=params.num_workers,
            reset_strat_folds=params.reset_strat_folds,
            distributed=params.distributed
        )

    elif params.dset == 'chapman':
        dls = DataLoaders(
            datasetCls=ChapmanDataset,
            dataset_kwargs=dataset_kwargs,
            batch_size=params.batch_size,
            workers=params.num_workers,
            reset_strat_folds=params.reset_strat_folds,
            distributed=params.distributed
        )

    elif params.dset == 'unified':
        dls = DataLoaders(
            datasetCls=UnifiedDataset,
            dataset_kwargs=dataset_kwargs,
            batch_size=params.batch_size,
            workers=params.num_workers,
            reset_strat_folds=params.reset_strat_folds,
            distributed=params.distributed
        )

    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[0], dls.train.dataset[0][0].shape[1]
    return dls

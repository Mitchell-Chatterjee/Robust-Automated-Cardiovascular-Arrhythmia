import os
from enum import Enum
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parents[3]

# Model weight root
MODEL_DIR = os.path.join(ROOT_DIR, 'results/Pre-training')

# Code root
CODE_DIR = os.path.join(ROOT_DIR, 'src')


class DataAugmentation(Enum):
    none = 0
    test_time_aug_cpc = 1
    test_time_aug_transformer = 2
    dropout_aug_transformer = 3
    per_lead_aug = 4
    pre_train_cpc = 5


class ModelSelectionMetric(Enum):
    valid_loss = 1
    valid_AUROC = 2


class Mode(Enum):
    PRETRAIN = 1
    PREDICTION = 2
    REGRESSION = 3
    CLASSIFICATION = 4


class Model(Enum):
    vanilla_vit = 1
    e_d_vit = 2
    cpc = 3
    patch_tst = 4

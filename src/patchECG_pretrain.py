# Command line imports
import argparse
import os

import torch

from src.core.constants.definitions import ROOT_DIR, DataAugmentation, Mode, Model
from src.core.datasets.ecg_interface import ClassificationType
from src.core.models.optim_factory import create_optimizer, prepare_optimizer
from src.core.support.abstract_support_class import AbstractSupportClass
from src.core.support.general import ModelSaver, ReconstructionPlotter
from src.core.support.scheduler import OneCycleLR
from src.core.utils.data_utils import get_dls
from src.core.utils.learner_utils import get_model, plot_recorders, update_ecg_interface, configure_dist_training, \
    terminate_dist_training, get_model_from_module
from src.core.utils.basics import snapshot_exists, load_snapshot
from src.core.learner import Learner
from src.core.support.patch_mask import PatchMaskSupport
from src.core.support.tracking import EvalTracker, PredictionTracker, TrainingTracker
from src.core.support.transforms import RevInSupport
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()

# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='CODE_Unannotated', help='dataset name')
parser.add_argument('--root_path', type=str, help='root path for the dataset',
                    default='/home/student/Datasets/100Hz/CODE')
parser.add_argument('--reset_strat_folds', action=argparse.BooleanOptionalAction, default=False,
                    help='reset stratification folds if set to true')
parser.add_argument('--top_n_classes', type=int, default=None,
                    help='top n classes to consider when performing classification, takes precendence over min'
                         'class size if present')
parser.add_argument('--min_class_size', type=int, default=50,
                    help='minimum class size to consider during classification.')
parser.add_argument('--custom_lead_selection', type=str, default='all_leads',
                    help='Define the leads used for pre-training. Can select preset: eight_leads, all_leads, or'
                         'define a custom list from [I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6]'
                         'using comma separated values')
parser.add_argument('--distributed', action=argparse.BooleanOptionalAction, default=False,
                    help='indicates whether we are running the model in distributed mode')
parser.add_argument('--data_sub_path', type=str, help='sub-path for the data folder', default='training')
parser.add_argument('--context_points', type=int, default=5000, help='sequence length')
parser.add_argument('--target_points', type=int, default=100, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=26, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--alt_lead_ordering', action=argparse.BooleanOptionalAction, default=False,
                    help='Re-order the leads from: [I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6] to'
                         '[I, II, V1, V2, V3, V4, V5, V6, III, AVR, AVL, AVF]')

# Patch
parser.add_argument('--patch_len', type=int, default=50, help='patch length')
parser.add_argument('--stride', type=int, default=50, help='stride between patch')

# RevIN
parser.add_argument('--revin_mode', type=str, default='BN', help='Use batch normalization or ptb-xl stats: [BN, ptb-xl]')

# Model args
parser.add_argument('--model', type=str, default='e_d_vit',
                    help=f'Choose the model. Note model parameters for models other than vit cannot be set dynamically.'
                         f'These include: {[e.name for e in Model]}')
parser.add_argument('--learning_rate', type=float, help='learning rate')
parser.add_argument('--n_layers', type=int, default=6, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=8, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=384, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='Transformer dropout')
parser.add_argument('--head_type', type=str, default='pretrain', help='Task type')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
parser.add_argument('--shared_embedding', action=argparse.BooleanOptionalAction, default=True,
                    help='shares embeddings matrix between each lead if set to true, otherwise separate')
parser.add_argument('--data_augmentation', type=str, default='none',
                    help=f'Choose from data augmentation methods {[e.name for e in DataAugmentation]}. '
                         f'test_time_aug: Trains model on randomly selected chunks from the original signal. '
                         f'During test time an ensemble of predictions is achieved by taking the mean of all predictions on a signal. '
                         f'Given a step size and chunk length'
                         f'per_lead_aug: Creates a classifier per-lead and takes the mean over each lead prediction at test time.')

# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
parser.add_argument('--noise_masking', action=argparse.BooleanOptionalAction, default=False,
                    help='includes noise in the pre-training task, in-place of zero values when masking')
parser.add_argument('--entire_signal_loss', action=argparse.BooleanOptionalAction, default=False,
                    help='allow loss to be calculated on the entire signal, only to be used when using noise masking')
parser.add_argument('--zero_noise_mixing', type=float, default=0.5,
                    help='Value between 0-1. Allows mixing noise masking and patch (zero) masking during pre-training.'
                         '1 is all noise masking, 0 is all zero masking.')

# Data Augmentation
parser.add_argument('--trafos', nargs='+', help='add transformation to data augmentation pipeline',
                    default=None)
# Sampled Noise
parser.add_argument('--t_root_path_noise', type=str, default='/home/student/Datasets/100Hz/noise',
                    help='path to root of noise dataset, if using noise masking')
# GaussianNoise
parser.add_argument('--t_gaussian_scale', help='std param for gaussian noise transformation',
                    default=0.005, type=float)
# RandomResizedCrop
parser.add_argument('--t_rr_crop_ratio_range', help='ratio range for random resized crop transformation',
                    default=[0.5, 1.0], type=float)
parser.add_argument('--t_output_size', help='output size for random resized crop transformation',
                    default=250, type=int)
# DynamicTimeWarp
parser.add_argument('--t_warps', help='number of warps for dynamic time warp transformation',
                    default=3, type=int)
parser.add_argument('--t_radius', help='radius of warps of dynamic time warp transformation',
                    default=10, type=int)
# TimeWarp
parser.add_argument('--t_epsilon', help='epsilon param for time warp',
                    default=10, type=float)
# ChannelResize
parser.add_argument('--t_magnitude_range', nargs='+', help='range for scale param for ChannelResize transformation',
                    default=[0.5, 2], type=float)
# Downsample
parser.add_argument('--t_downsample_ratio', help='downsample ratio for Downsample transformation',
                    default=0.2, type=float)
# TimeOut
parser.add_argument('--t_to_crop_ratio_range', nargs='+', help='ratio range for timeout transformation',
                    default=[0.2, 0.4], type=float)

# Optimization args
parser.add_argument('--save_every', type=int, default=10, help='save model every n epochs')
parser.add_argument('--plot_every_n', type=int, default=5, help='plot reconstruction every n epochs')
parser.add_argument('--n_epochs_pretrain', type=int, default=100, help='number of pre-training epochs')

parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: 0.9, 0.999, use opt default)')
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
       weight decay. We use a cosine schedule for WD. 
       (Set the same value with args.weight_decay to keep weight decay no change)""")
parser.add_argument('--targeted_opt', action=argparse.BooleanOptionalAction, default=False,
                    help='will use optimizer with layer-wise decay if set to true (same as fine-tuning)')

parser.add_argument('--scheduler', action=argparse.BooleanOptionalAction, default=True,
                    help='Will use a learning rate scheduler if set to true. Default is OneCycleLR.')

parser.add_argument('--sam', action=argparse.BooleanOptionalAction, metavar='SAM', default=False,
                    help='Wraps the optimizer in sharpness aware maximization optimizer if applied')
parser.add_argument('--sam_adaptive', action=argparse.BooleanOptionalAction, metavar='ADAPTIVE_SAM', default=True,
                    help='Sets SAM to Adaptive SAM')
parser.add_argument('--sam_rho', type=float, metavar='RHO', default=2.,
                    help='Rho value for SAM optimizer')

parser.add_argument('--lr', type=float, default=0.0015, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--layer_decay', type=float, default=1)

parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')

parser.add_argument('--focal_loss', action=argparse.BooleanOptionalAction, default=False,
                    help='Focal loss only applies to binary and multi-label classification')
parser.add_argument('--focal_alpha', type=float, default=0.25,
                    help='Value between 0 and 1. Defines the strength of the focal loss term.')

# model name to keep track of saved models
parser.add_argument('--model_name', type=str, default='test', help='name of the model (learning rate, etc)')

# Distributed training args
parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist_backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--local_node', action=argparse.BooleanOptionalAction, default=False,
                    help='prepares distributed trainer to run on local gpu instead of SLURM workload managed cluster')

# Pretrained model name
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model name')

params = parser.parse_args()
params.save_path = 'results/pre-train/saved_models/' + params.dset_pretrain + f'/{params.model_name}/'


def pretrain_func(args, global_rank, local_rank):
    if global_rank == 0:
        with open(os.path.join(params.save_path, 'params.txt'), 'w') as f:
            f.write(str(args))

    # get dataloader
    dls = get_dls(args)

    # Update ECG interface
    seq_len = dls.len if args.data_augmentation is DataAugmentation.per_lead_aug else dls.len * dls.vars
    c_in = dls.vars if args.data_augmentation is DataAugmentation.per_lead_aug else 1
    update_ecg_interface(number_of_leads=c_in, resample_rate=dls.len)

    # get model     
    model = get_model(c_in=c_in, seq_len=seq_len, args=args, global_rank=global_rank)
    # If we have a pre-trained model and we are performing another round of SSRL pre-training. Load that model.
    if args.pretrained_model_path is not None and snapshot_exists(path=os.path.join(ROOT_DIR, args.pretrained_model_path)):
        model = load_snapshot(model, os.path.join(ROOT_DIR, args.pretrained_model_path))
    # Create model
    model = DDP(model.to(local_rank), device_ids=[local_rank]) if args.distributed else model.to(local_rank)
    # get optimizer
    optimizer = prepare_optimizer(model=model, args=args) if args.pretrained_model_path is not None else \
        create_optimizer(model=model, args=args, filter_bias_and_bn=False)

    # get loss, label type
    patch_masker = PatchMaskSupport(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio,
                                    per_lead=args.data_augmentation is DataAugmentation.per_lead_aug)
    rec_plotter = ReconstructionPlotter(every_n_epoch=args.plot_every_n, save_path=args.save_path) \
        if args.model in [Model.vanilla_vit, Model.e_d_vit, Model.patch_tst] else AbstractSupportClass()
    label_type = dls.train.dataset.label_type
    loss_func = get_model_from_module(model).cpc_loss if args.model is Model.cpc else patch_masker.loss

    # define learner
    learn = Learner(dls=dls,
                    train_sampler=dls.train_sampler,
                    model=model,
                    mode=args.head_type,
                    opt=optimizer,
                    scheduler=OneCycleLR(lr_max=args.lr, pct_start=0.3) if args.scheduler else AbstractSupportClass(),
                    local_rank=local_rank,
                    global_rank=global_rank,
                    classification_type=dls.train.dataset.classification_type,
                    loss_func=loss_func,
                    label_type=label_type,
                    distributed=args.distributed,
                    lr=args.lr,
                    clip_grad=args.clip_grad,
                    model_saver=ModelSaver(n_epochs=args.n_epochs_pretrain, args=args, every_epoch=args.save_every,
                                           monitor='valid_loss', fname=args.model_name,
                                           path=args.save_path, global_rank=global_rank),
                    eval_tracker=EvalTracker(number_of_classes=0,
                                             metrics_by_class=False,
                                             classification_type=params.classification_type),
                    prediction_tracker=PredictionTracker(),
                    training_tracker=TrainingTracker(loss_func=loss_func, mode=args.head_type,
                                                     number_of_classes=0,
                                                     save_path=os.path.join(args.save_path,
                                                                            f'_training_metrics_gpu{global_rank}_{args.model_name}.csv'),
                                                     classification_type=params.classification_type),
                    reconstruction_plotter=rec_plotter,
                    patch_masker=patch_masker,
                    normalization=RevInSupport(dls.vars, denorm=False, norm_stats=args.revin_mode)
                    )

    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain)
    # Plot training and validation loss curves
    if global_rank == 0:
        plot_recorders(tracker=learn.training_tracker, split='Loss', args=args,
                       metrics=['train_loss', 'valid_loss'],
                       n_epochs=args.n_epochs_pretrain)


# Uncomment this for command line mode
if __name__ == '__main__':
    # Set the dataset
    params.dset = params.dset_pretrain

    # Ensure model has a name
    assert params.model_name is not None, 'Model must have a name'
    # Ensure masked noise mixing is between 0 and 1
    assert 0 <= params.zero_noise_mixing <= 1, 'Masked noise masking must be a value between 0 and 1'
    # Convert head type to mode
    assert params.head_type in ['pretrain', 'prediction', 'regression', 'classification'], \
        'head type should be either pretrain, prediction, regression, or classification'

    params.model = Model[params.model.lower()]
    params.head_type = Mode[params.head_type.upper()]
    params.classification_type = ClassificationType.PRETRAIN

    params.data_augmentation = DataAugmentation[params.data_augmentation.lower()]
    assert params.data_augmentation in [DataAugmentation.none, DataAugmentation.per_lead_aug, DataAugmentation.pre_train_cpc], \
        "Pre-training data augmentation is restricted to per-lead or none"

    if params.model == Model.cpc:
        torch.backends.cudnn.enabled = False

    # Run distributed pretraining
    l_rank, g_rank = configure_dist_training(params=params)
    pretrain_func(args=params, global_rank=g_rank, local_rank=l_rank)
    terminate_dist_training(global_rank=g_rank, stage='pretraining')

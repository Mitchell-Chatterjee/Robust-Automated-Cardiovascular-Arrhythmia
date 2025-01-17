# Command line imports
import functools
import os
import pathlib

import pandas as pd
import peft

from src.core.datasets.ecg_interface import ClassificationType
from src.core.datasets.ptb_xl.ptb_xl_dataset import DiagnosticClass
from src.core.models.optim_factory import prepare_optimizer, create_optimizer
from src.core.models.patchTST import ClassToken
from src.core.learner import Learner
from src.core.support.abstract_support_class import AbstractSupportClass
from src.core.support.patch_mask import FinetunePatchSupport
from src.core.support.general import ModelSaver
from src.core.support.scheduler import OneCycleLR
from src.core.support.tracking import PredictionTracker, EvalTracker, TrainingTracker, compute_scores_multi_class, \
    compute_scores_multi_label, compute_scores_binary
from src.core.support.transforms import RevInSupport
from src.core.utils.basics import snapshot_exists
from src.core.utils.learner_utils import get_model, plot_recorders, update_ecg_interface, transfer_weights, \
    get_bootstrap_samples
from src.core.utils.data_utils import get_dls
from src.core.constants.definitions import ROOT_DIR, DataAugmentation, ModelSelectionMetric, Mode, Model
from torch.nn.parallel import DistributedDataParallel as DDP

import torch, gc
gc.collect()
torch.cuda.empty_cache()

import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()

# Pretraining and Finetuning
parser.add_argument('--linear_probing', action=argparse.BooleanOptionalAction, default=False,
                    help='whether or not to do linear probing (default false)')
parser.add_argument('--finetuning', action=argparse.BooleanOptionalAction, default=True,
                    help='whether or not to do fine-tuning (default true)')
parser.add_argument('--testing', action=argparse.BooleanOptionalAction, default=True,
                    help='whether or not to do testing (default true)')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
parser.add_argument('--metric_by_class', action=argparse.BooleanOptionalAction, default=False,
                    help='prints metrics by class if true')
parser.add_argument('--linear_only', action=argparse.BooleanOptionalAction, default=False,
                    help='Linear fine-tuning head. Used for CPC model')
# Dataset and dataloader
parser.add_argument('--dset_finetune', type=str, default='ptb-xl', help='dataset name')
parser.add_argument('--root_path', type=str, help='root path for the dataset',
                    default='/home/student/Datasets/100Hz/ptb-xl_1.0.3')
parser.add_argument('--reset_strat_folds', action=argparse.BooleanOptionalAction, default=False,
                    help='reset stratification folds if set to true')
parser.add_argument('--top_n_classes', type=int, default=None,
                    help='top n classes to consider when performing classification, takes precendence over min'
                         'class size if present')
parser.add_argument('--min_class_size', type=int, default=10,
                    help='minimum class size to consider during classification.')
parser.add_argument('--custom_class_selection', type=str,
                    help="Custom class selection as comma separated values. If keyword 'Other' is included, then all "
                         "groups not included in the list of classes will be aggregated into the 'Other' class.")
parser.add_argument('--custom_lead_selection', type=str, default='all_leads',
                    help='Define the leads used for pre-training. Can select preset: eight_leads, all_leads, or'
                         'define a custom list from [I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6]'
                         'using comma separated values')
parser.add_argument('--classification_type', type=str, default='MULTI_LABEL',
                    help=f'Defines the classification task. These include: {[e.name for e in ClassificationType]}')
parser.add_argument('--diagnostic_class', type=str, default='all',
                    help=f'Specifically for the PTB-XL dataset. Defines the different diagnostic tasks the dataset may '
                         f'be used to test. These include: {[e.name for e in DiagnosticClass]}')
parser.add_argument('--distributed', action=argparse.BooleanOptionalAction, default=False,
                    help='indicates whether we are running the model in distributed mode')
parser.add_argument('--data_sub_path', type=str, help='sub-path for the data folder', default='WFDBRecords')
parser.add_argument('--context_points', type=int, default=5000, help='sequence length')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
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
parser.add_argument('--model', type=str, default='vanilla_vit',
                    help=f'Choose the model. Note model parameters for models other than vit cannot be set dynamically.'
                         f'These include: {[e.name for e in Model]}')
parser.add_argument('--n_classes', type=int, default=None,
                    help='number of classes during fine-tuning, set automatically')
parser.add_argument('--class_token', type=str, default='cls_token', help='type of classification task'
                                                                         '(global_mean or cls_token)')
parser.add_argument('--n_layers', type=int, default=12, help='number of Transformer layers')
parser.add_argument('--head_type', type=str, default='classification', help='Task type')
parser.add_argument('--n_heads', type=int, default=12, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=768, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=3072, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0., help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0., help='head dropout')
parser.add_argument('--shared_embedding', action=argparse.BooleanOptionalAction, default=True,
                    help='shares embeddings matrix between each lead if set to true, otherwise separate')

# Optimization args
parser.add_argument('--save_every', type=int, default=10, help='save model every n epochs')
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--n_epochs_finetune_head', type=int, default=50, help='number of finetuning epochs (head only)')
parser.add_argument('--n_epochs_probe', type=int, default=0, help='number of linear probing epochs')
parser.add_argument('--focal_loss', action=argparse.BooleanOptionalAction, default=False,
                    help='Focal loss only applies to binary and multi-label classification')
parser.add_argument('--focal_alpha', type=float, default=None,
                    help='Value between 0 and 1. Defines the strength of the focal loss term.')
parser.add_argument('--model_selection_metric', type=str, default='valid_loss',
                    help=f'Determines which metric should be used for selecting the best model among: {[e.name for e in ModelSelectionMetric]}. '
                         f'Default validation_loss.')

parser.add_argument('--data_augmentation', type=str, default='none',
                    help=f'Choose from data augmentation methods {[e.name for e in DataAugmentation]}. '
                         f'test_time_aug_cpc: Trains model on randomly selected chunks from the original signal. (Specific to cpc model)'
                         f'test_time_aug_transformer: Trains model on randomly selected chunks from the original signal. (Specific to transformer model)'
                         f'During test time an ensemble of predictions is achieved by taking the mean of all predictions on a signal. '
                         f'Given a step size and chunk length'
                         f'per_lead_aug: Creates a classifier per-lead and takes the mean over each lead prediction at test time.')
parser.add_argument('--chunk_size', help='Chunk length for test time augmentation',
                    default=250, type=int)
parser.add_argument('--chunk_step', help='Step size for test time augmentation',
                    default=125, type=int)

# LoRA args
parser.add_argument('--lora', action=argparse.BooleanOptionalAction, default=False,
                    help='uses lora when fine-tuning')
parser.add_argument('--lora_alpha', type=int, default=16, help='alpha value for LoRA')
parser.add_argument('--lora_r', type=int, default=16, help='r value for LoRA')
parser.add_argument('--lora_dropout', type=float, default=0.05, help='dropout for LoRA')


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

# Fine-tuning mask for adding noise
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
parser.add_argument('--root_path_noise', type=str, default='/home/student/Datasets/100Hz/noise',
                    help='path to root of noise dataset, if using noise masking')
parser.add_argument('--noise_masking', action=argparse.BooleanOptionalAction, default=False,
                    help='includes noise in the pre-training task, in-place of zero values when masking')

parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
    weight decay. We use a cosine schedule for WD and using a larger decay by
    the end of training improves performance for ViTs.""")

parser.add_argument('--scheduler', action=argparse.BooleanOptionalAction, default=False,
                    help='Will use a learning rate scheduler if set to true. Default is OneCycleLR.')

parser.add_argument('--sam', action=argparse.BooleanOptionalAction, metavar='SAM', default=False,
                    help='Wraps the optimizer in sharpness aware maximization optimizer if applied')
parser.add_argument('--sam_adaptive', action=argparse.BooleanOptionalAction, metavar='ADAPTIVE_SAM', default=True,
                    help='Sets SAM to Adaptive SAM')
parser.add_argument('--sam_rho', type=float, metavar='RHO', default=2.,
                    help='Rho value for SAM optimizer')

parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--layer_decay', type=float, default=0.65)

# Mentor Mixup
parser.add_argument('--mentor_mix', action=argparse.BooleanOptionalAction, default=False,
                    help='performs mentor mixup to increase robustness to label noise')
parser.add_argument('--mentor_mix_start_epoch', type=float, default=2,
                    help='Start epoch for mentor mix, starting later allows model to perform early learning stage'
                         'without additional noise')
parser.add_argument('--mentor_mix_alpha', type=float, default=1,
                    help='Alpha parameter for beta distribution of mentor mix')
parser.add_argument('--mentor_mix_gamma', type=float, default=0.9,
                    help='Gamma parameter for exponential moving average of mentor mix')

parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                    help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

# Confidence intervals
parser.add_argument('--bootstrapping', action=argparse.BooleanOptionalAction, default=False,
                    help='performs empirical bootstrapping with the given model')
parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for computing confidence intervals')


# Pretrained model name
parser.add_argument('--pretrained_model_path', type=str,
                    default=None,
                    help='pretrained model name')

# model name to keep track of saved models
parser.add_argument('--model_name', type=str, default='test', help='name of the model (learning rate, etc)')

args = parser.parse_args()
args.save_path = 'results/fine-tune/saved_models/' + args.dset_finetune + f'/{args.model_name}/'


def prepare_learner(dls, global_rank, local_rank):
    # Assign the number of classes
    args.n_classes = dls.train.dataset.n_classes

    # Update ECG interface
    seq_len = dls.len if args.data_augmentation is DataAugmentation.per_lead_aug else dls.len * dls.vars
    c_in = dls.vars if args.data_augmentation is DataAugmentation.per_lead_aug else 1
    update_ecg_interface(number_of_leads=c_in, resample_rate=dls.len)

    # get model
    model = get_model(c_in=c_in, seq_len=seq_len, args=args, global_rank=global_rank)
    # transfer weights from pre-trained model
    if args.pretrained_model_path is not None and snapshot_exists(path=os.path.join(ROOT_DIR, args.pretrained_model_path)):
        model = transfer_weights(os.path.join(ROOT_DIR, args.pretrained_model_path), model)

    # Prepare PEFT fine-tuning if true
    if args.lora:
        assert hasattr(model, "lora_modules"), "Target Modules must be defined in the model"
        assert hasattr(model, "ft_modules"), "Modules to save must be defined in the model"
        config = peft.LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=model.lora_modules,
            modules_to_save=model.ft_modules,
        )
        model = peft.get_peft_model(model, config)
        model.print_trainable_parameters()

    # Prepare model for distributed data parallel if distributed training is specified
    model = DDP(model.to(local_rank), device_ids=[local_rank]) if args.distributed else model.to(local_rank)
    optimizer = prepare_optimizer(model=model, args=args) if args.pretrained_model_path is not None else\
        create_optimizer(model=model, args=args, filter_bias_and_bn=False)

    # get loss, label type
    loss_func, label_type = dls.train.dataset.loss_func, dls.train.dataset.label_type

    # Plot classes from training dataloader for interpretation
    dls.train.dataset.plot_classes(plot_path=os.path.join(args.save_path, f'training_data_dist_{args.model_name}'))
    # print class sizes
    for elem in [dls.train, dls.valid, dls.test]:
        elem.dataset.print_class_sizes()

    # define learner
    learn = Learner(dls=dls,
                    train_sampler=dls.train_sampler,
                    model=model,
                    mode=args.head_type,
                    opt=optimizer,
                    scheduler=OneCycleLR(lr_max=args.lr, pct_start=0.3) if args.scheduler else AbstractSupportClass(),
                    mentor_mix=args.mentor_mix,
                    mentor_mix_start_epoch=args.mentor_mix_start_epoch,
                    mentor_mix_alpha=args.mentor_mix_alpha,
                    mentor_mix_gamma=args.mentor_mix_gamma,
                    local_rank=local_rank,
                    global_rank=global_rank,
                    classification_type=dls.train.dataset.classification_type,
                    loss_func=loss_func,
                    label_type=label_type,
                    distributed=args.distributed,
                    lr=args.lr,
                    clip_grad=args.clip_grad,
                    # metrics=[mse],
                    model_saver=ModelSaver(n_epochs=args.n_epochs_finetune, args=args, every_epoch=args.save_every,
                                           monitor=args.model_selection_metric.name, fname=args.model_name,
                                           path=args.save_path, global_rank=global_rank),
                    eval_tracker=EvalTracker(number_of_classes=args.n_classes,
                                             metrics_by_class=args.metric_by_class,
                                             classification_type=dls.train.dataset.classification_type),
                    prediction_tracker=PredictionTracker(),
                    training_tracker=TrainingTracker(loss_func=loss_func, mode=args.head_type,
                                                     number_of_classes=args.n_classes,
                                                     classification_type=dls.train.dataset.classification_type,
                                                     save_path=os.path.join(args.save_path,
                                                                            f'_training_metrics_gpu{global_rank}_{args.model_name}')),
                    patch_masker=FinetunePatchSupport(patch_len=args.patch_len, stride=args.stride,
                                                      per_lead=args.data_augmentation is DataAugmentation.per_lead_aug),
                    normalization=RevInSupport(dls.vars, denorm=False, norm_stats=args.revin_mode),
                    data_augmentation=args.data_augmentation
                    )
    return learn


def finetune_func(learn):

    # fit the data to the model
    learn.fine_tune(n_epochs=args.n_epochs_finetune, freeze_epochs=args.n_epochs_finetune_head,
                    lora=args.lora)

    # Plot training and validation loss curves
    plot_recorders(tracker=learn.training_tracker, split='Loss', args=args,
                   metrics=['train_loss', 'valid_loss'],
                   n_epochs=args.n_epochs_finetune + args.n_epochs_finetune_head)

    # Plot training metrics
    plot_recorders(tracker=learn.training_tracker, split='Training', args=args,
                   metrics=['train_accuracy', 'train_precision', 'train_recall', 'train_f1_score'],
                   n_epochs=args.n_epochs_finetune + args.n_epochs_finetune_head)

    # Plot validation metrics
    plot_recorders(tracker=learn.training_tracker, split='Validation', args=args,
                   metrics=['valid_accuracy', 'valid_precision', 'valid_recall', 'valid_f1_score'],
                   n_epochs=args.n_epochs_finetune + args.n_epochs_finetune_head,)

    # Plot validation metrics
    plot_recorders(tracker=learn.training_tracker, split='AUROC-AUPRC', args=args,
                   metrics=['valid_AUROC', 'valid_AUPRC'],
                   n_epochs=args.n_epochs_finetune + args.n_epochs_finetune_head)


def test_func(learn, dls):
    # Get the weight path for the fine-tuned model
    weight_path = os.path.join(ROOT_DIR, args.save_path, f'{args.model_name}_best.pt')

    # Test the best performing model and get the per-class results on train, validation and test set
    if args.data_augmentation not in [DataAugmentation.test_time_aug_cpc, DataAugmentation.test_time_aug_transformer]:
        learn.eval(dataloader=dls.train,
                   save_path=os.path.join(args.save_path, f'_train_metrics_{args.model_name}'),
                   class_sizes=dls.train.dataset.get_class_sizes(),
                   split='train',
                   per_class_only=True,
                   weight_path=weight_path)
    learn.eval(dataloader=dls.valid,
               save_path=os.path.join(args.save_path, f'_valid_metrics_{args.model_name}'),
               class_sizes=dls.valid.dataset.get_class_sizes(),
               split='validation',
               per_class_only=True,
               weight_path=weight_path)
    learn.eval(dataloader=dls.test,
               save_path=os.path.join(args.save_path, f'_test_metrics_{args.model_name}'),
               class_sizes=dls.test.dataset.get_class_sizes(),
               split='test',
               per_class_only=False,
               weight_path=weight_path)


def bootstrap_function(learn, dls):
    # Get best model
    learn.load_model(weight_path=os.path.join(ROOT_DIR, args.save_path, f'{args.model_name}_best.pt'))

    # Get correct subsets
    scores = {'AUROC': [], 'AUPRC': []}
    data, labels = (torch.stack(elem) for elem in zip(*[dls.test.dataset[i] for i in range(len(dls.test.dataset))]))
    bootstrap_samples = get_bootstrap_samples(labels, args.iterations)

    # Get prediction for each
    for indices in bootstrap_samples:
        # Feed in as batches
        pred_list, yb_list = torch.empty((0, args.n_classes)).squeeze(), torch.empty((0, args.n_classes)).squeeze()
        batches = [indices[i:i + args.batch_size] for i in range(0, len(indices), args.batch_size)]

        for batch in batches:
            with torch.no_grad():
                pred, _ = learn.batch_predict([data[batch], labels[batch]])
                pred_list = torch.concat((pred_list, pred.detach().cpu()), dim=0)

        # Apply sigmoid if not already applied
        if torch.max(abs(pred_list)) > 1:
            pred_list = torch.sigmoid(pred_list)

        # Calculate AUROC score (append to list)
        if dls.train.dataset.classification_type is ClassificationType.MULTI_CLASS:
            temp_vals, _ = compute_scores_multi_class(raw_predictions=pred_list, values={},
                                                      per_class_values=False, labels=labels[indices],
                                                      number_of_classes=args.n_classes, metrics_by_class=False)
        elif dls.train.dataset.classification_type is ClassificationType.MULTI_LABEL:
            temp_vals, _ = compute_scores_multi_label(raw_predictions=pred_list, values={},
                                                      per_class_values=False, labels=labels[indices],
                                                      number_of_labels=args.n_classes, metrics_by_class=False)
        else:
            # dls.train.dataset.classification_type is ClassificationType.BINARY
            temp_vals, _ = compute_scores_binary(raw_predictions=pred_list, values={},
                                                 per_class_values=False, labels=labels[indices])

        scores['AUROC'].append(temp_vals['AUROC'])
        scores['AUPRC'].append(temp_vals['AUPRC'])

    # Create dataframe and compute scores
    df = pd.DataFrame(scores)
    pd.DataFrame({'lower_quantile': df.quantile(0.05), 'mean': df.mean(), 'upper_quantile': df.quantile(0.95)}) \
        .to_csv(pathlib.Path(args.save_path) / 'confidence_bounds.csv')
    df.to_csv(pathlib.Path(args.save_path) / 'auroc_auprc_scores.csv')


def linear_probing_func(learn):
    # Test the model on the test data
    learn.linear_probe(n_epochs=args.n_epochs_probe, cls_token=False)


if __name__ == '__main__':
    # Set the dataset
    args.dset = args.dset_finetune

    # Convert head type to mode
    assert args.head_type in ['pretrain', 'prediction', 'regression', 'classification'], \
        'head type should be either pretrain, prediction, regression, or classification'
    args.head_type = Mode[args.head_type.upper()]
    # Convert classification token to class_token enum
    assert args.class_token in ['cls_token', 'global_mean'], \
        'classification token should either be cls_token or global_mean (global mean pooling)'
    # Ensure model has a name
    assert args.model_name is not None, 'Model must have a name'

    # Convert strings to enums
    args.model = Model[args.model.lower()]
    args.class_token = ClassToken[args.class_token.upper()]
    args.diagnostic_class = DiagnosticClass[args.diagnostic_class.upper()]
    args.classification_type = ClassificationType[args.classification_type.upper()]
    args.data_augmentation = DataAugmentation[args.data_augmentation.lower()]
    args.model_selection_metric = ModelSelectionMetric[args.model_selection_metric]

    if args.model == Model.cpc:
        torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    print(args)

    # Save params to text
    with open(os.path.join(args.save_path, 'params.txt'), 'w') as f:
        f.write(str(args))

    # Get the dataloaders
    dataloaders = get_dls(args)

    # l_rank, g_rank = configure_dist_training(params=args)
    learner = prepare_learner(dls=dataloaders, global_rank=0, local_rank=0)
    # For now only allow linear probing or fine-tuning
    if args.bootstrapping:
        bootstrap_function(learn=learner, dls=dataloaders)
    elif args.linear_probing:
        linear_probing_func(learn=learner)
    elif args.finetuning:
        # Run fine-tuning
        finetune_func(learn=learner)

    # Run testing
    if args.testing:
        test_func(learn=learner, dls=dataloaders)
    # terminate_dist_training(global_rank=g_rank, stage='fine-tuning')

    # TODO: Amalgamate recorders into one for post-training graphics

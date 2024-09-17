import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel

from src.core.constants.definitions import Mode, Model
from src.core.datasets.ecg_interface import EcgInterface
from src.core.models.encoder_decoder_vit import EncoderDecoderViT
from src.core.models.lstm.models.cpc import CPCModel
from src.core.models.patchTST import PatchTST
from src.core.models.vit_1d import ViT


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)


def get_model(c_in, seq_len, args, global_rank):
    """
    c_in: number of variables
    """

    # get number of patches
    num_patch = (max(seq_len, args.patch_len) - args.patch_len) // args.stride + 1

    target_dim = args.n_classes if args.head_type is Mode.CLASSIFICATION else args.patch_len
    class_token = args.class_token if args.head_type is Mode.CLASSIFICATION else None

    if args.model == Model.e_d_vit:
        model = EncoderDecoderViT(
            seq_len=seq_len,
            patch_size=args.patch_len,
            num_classes=target_dim,
            dim=args.d_model,
            depth=args.n_layers,
            heads=args.n_heads,
            mlp_dim=args.d_ff,
            mode=args.head_type,
            channels=c_in,
            dim_head=args.d_model // args.n_heads,
            dropout=args.head_dropout,
            emb_dropout=args.dropout
        )
        model.apply(init_weights)
    elif args.model == Model.patch_tst:
        model = PatchTST(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=args.patch_len,
            stride=args.stride,
            target_dim=target_dim,
            n_layers=args.n_layers,
            d_model=args.d_model,
            head_type=args.head_type,
            class_token=class_token,
            n_heads=args.n_heads,
            shared_embedding=args.shared_embedding,
            d_ff=args.d_ff,
            attn_dropout=args.dropout,
            dropout=args.dropout,
            head_dropout=args.head_dropout
        )
    elif args.model == Model.cpc:
        strides = [1] * 4
        kss = [1] * 4
        lin_ftrs_head = [512]
        ps_head = 0.5
        bn_head = True
        model = CPCModel(input_channels=12, mode=args.head_type, strides=strides, kss=kss, features=[512]*4, n_hidden=512, n_layers=2, mlp=False,
                         lstm=True, bias_proj=False, num_classes=target_dim, skip_encoder=False, bn_encoder=True,
                         lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_head=bn_head, linear_only=args.linear_only)
    else:
        # vanilla vit case
        model = ViT(
            seq_len=seq_len,
            patch_size=args.patch_len,
            stride=args.stride,
            num_classes=target_dim,
            dim=args.d_model,
            depth=args.n_layers,
            heads=args.n_heads,
            mlp_dim=args.d_ff,
            mode=args.head_type,
            channels=c_in,
            dim_head=args.d_model // args.n_heads,
            dropout=args.head_dropout,
            emb_dropout=args.dropout
        )

    # print out the model size
    if global_rank == 0:
        print('number of patches:', num_patch)
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def configure_dist_training(params):
    """"
    Prepares learner to run in Distributed Data Parallel mode. Can either be done on a single node or across multiple
    nodes on a cluster. As SLURM workload managed clusters differ in how variables are set, we must separate the
    two training cases.
    """

    def configure_local_node_training():
        # Initialize distributed training
        init_process_group(backend=params.dist_backend)
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # These values are instantiated by torch run
        return int(os.environ['LOCAL_RANK']), int(os.environ['RANK']), torch.cuda.device_count()

    def configure_cluster_training():
        ngpus_per_node = torch.cuda.device_count()

        l_rank = int(os.environ.get("SLURM_LOCALID"))
        g_rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + l_rank

        torch.cuda.set_device(l_rank)

        init_process_group(backend=params.dist_backend, init_method=params.init_method, world_size=params.world_size,
                           rank=g_rank)  # Initialize Data Parallelism communications
        return l_rank, g_rank, ngpus_per_node

    if params.local_node:
        local_rank, global_rank, world_size = configure_local_node_training()
    else:
        local_rank, global_rank, world_size = configure_cluster_training()

    if not os.path.exists(params.save_path) and global_rank == 0:
        os.makedirs(params.save_path)

    if global_rank == 0:
        print(f'\n\nExecuting program in Distributed Data Parallel mode')
        print(f'Number of GPUs available on each node: {world_size}\n')
        print('args:', params, end='\n\n')

    return local_rank, global_rank


def terminate_dist_training(global_rank, stage):
    if global_rank == 0:
        print(f'{stage} complete')
    destroy_process_group()


def plot_recorders(tracker, split, args, metrics, n_epochs):
    # Plot data
    for metric in metrics:
        plt.plot(range(1, n_epochs + 1), tracker.recorder[metric], 'o-', label=metric)

    # Add a legend
    plt.legend()

    # Add labels
    plt.xlabel("Epoch Number")
    plt.ylabel(f"{split} metrics")
    plt.title(f"{split} Values")

    # Save the plot
    plt.savefig(os.path.join(args.save_path, f'{split}_plot_{args.model_name}.png'))
    plt.close()


def get_model_from_module(model):
    "Return the model maybe wrapped inside `model`."
    if isinstance(model, (DistributedDataParallel, nn.DataParallel)):
        model = model.module
    return model


def update_ecg_interface(number_of_leads, resample_rate, time_window=10, strat_seed=200):
    EcgInterface.NUMBER_OF_LEADS = number_of_leads
    EcgInterface.resample_rate = resample_rate
    EcgInterface.resample_freq = resample_rate // time_window
    EcgInterface.stratification_seed = strat_seed


def transfer_weights(weights_path, model, exclude_head=True, device='cpu'):
    # state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device)
    if isinstance(model, CPCModel) and 'state_dict' in new_state_dict:
        new_state_dict = new_state_dict['state_dict']
        new_state_dict = {key.replace('model_cpc.', ''): v for key, v in new_state_dict.items()}
    else:
        new_state_dict = new_state_dict['model']

    # print('new_state_dict',new_state_dict)
    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items():
        if exclude_head and 'head' in name:
            continue
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            elif 'pos' in name:
                param.copy_(input_param[0, param.shape[1], :])
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass  # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device)
    return model


def get_bootstrap_samples(test_dataset, iterations):
    test_ids, temp = [], 0

    while iterations - temp > 0:
        # Get random sample from test set (with replacement)
        indices = np.sort(np.random.choice(a=len(test_dataset), size=len(test_dataset), replace=True))

        # Ensure there is at least one sample from each class
        label_counts = torch.sum(test_dataset[indices], dim=0)
        if not torch.all(label_counts):
            continue

        test_ids.append(indices)
        temp += 1

    print(f"\nGenerated {iterations} splits for computing confidence intervals\n")
    return test_ids

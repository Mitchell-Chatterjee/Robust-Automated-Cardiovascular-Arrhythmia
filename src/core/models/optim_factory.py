# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------'
import torch
from torch import optim as optim, nn

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

from src.core.models.lstm.models.cpc import CPCModel
from src.core.models.sam import SAM

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    # TODO: Revert this back if using PatchTST
    if any([elem in var_name for elem in ['CLS_TOKEN', 'W_pos', 'W_P', 'to_patch_embedding', 'pos_embedding',
                                          'cls_token', 'mask_token']]):
        return 0
    elif 'layers' in var_name:
        layer_id = int(var_name.split('layers.')[1].split('.')[0])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    if hasattr(args, 'finetuning') and args.finetuning and isinstance(model, CPCModel):
        parameters = [
            {"params": model.encoder.parameters(), "lr_scale": args.layer_decay * args.layer_decay},
            {"params": model.rnn.parameters(), "lr_scale": args.layer_decay},
            {"params": model.head.parameters(), "lr_scale": 1}
        ]
    elif weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = [{'params': [param for _, param in model.named_parameters() if param.requires_grad],
                       'lr_scale': 1,
                       'weight_decay': weight_decay}]

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        opt_args['momentum'] = args.momentum
        opt_args['nesterov'] = True
        base_optimizer = optim.SGD
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        opt_args['momentum'] = args.momentum
        opt_args['nesterov'] = False
        base_optimizer = optim.SGD
    elif opt_lower == 'adam':
        base_optimizer = optim.Adam
    elif opt_lower == 'adamw':
        base_optimizer = optim.AdamW
    elif opt_lower == 'nadam':
        base_optimizer = Nadam
    elif opt_lower == 'radam':
        base_optimizer = RAdam
    elif opt_lower == 'adamp':
        opt_args['wd_ratio'] = 0.01
        opt_args['nesterov'] = True
        base_optimizer = AdamP
    elif opt_lower == 'sgdp':
        opt_args['momentum'] = args.momentum
        opt_args['nesterov'] = True
        base_optimizer = SGDP
    elif opt_lower == 'adadelta':
        base_optimizer = optim.Adadelta
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        base_optimizer = Adafactor
    elif opt_lower == 'adahessian':
        base_optimizer = Adahessian
    elif opt_lower == 'rmsprop':
        opt_args['momentum'] = args.momentum
        opt_args['alpha'] = 0.9
        base_optimizer = optim.RMSprop
    elif opt_lower == 'rmsproptf':
        opt_args['momentum'] = args.momentum
        opt_args['alpha'] = 0.9
        base_optimizer = RMSpropTF
    elif opt_lower == 'nvnovograd':
        base_optimizer = NvNovoGrad
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        opt_args['momentum'] = args.momentum
        opt_args['nesterov'] = True
        base_optimizer = FusedSGD
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        opt_args['momentum'] = args.momentum
        opt_args['nesterov'] = False
        base_optimizer = FusedSGD
    elif opt_lower == 'fusedadam':
        opt_args['adam_w_mode'] = False
        base_optimizer = FusedAdam
    elif opt_lower == 'fusedadamw':
        opt_args['adam_w_mode'] = True
        base_optimizer = FusedAdam
    elif opt_lower == 'fusedlamb':
        base_optimizer = FusedLAMB
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        base_optimizer = FusedNovoGrad
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if args.sam:
        optimizer = SAM(params=parameters, base_optimizer=base_optimizer, rho=args.sam_rho,
                        adaptive=args.sam_adaptive, **opt_args)
    else:
        optimizer = base_optimizer(params=parameters, **opt_args)

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer


def prepare_optimizer(model, args):
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (args.n_layers + 1 - i) for i in range(args.n_layers + 2)))
    else:
        assigner = None

    optimizer = create_optimizer(
        args, model,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    return optimizer

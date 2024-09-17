import math
from cmath import inf

import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

from src.core.models.sam import SAM
from src.core.support.abstract_support_class import AbstractSupportClass


class OneCycleLR(AbstractSupportClass):
    def __init__(self,
                 lr_max=None,
                 lr=None,
                 total_steps=None,
                 steps_per_epoch=None,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 three_phase=False,
                 last_epoch=-1,
                 verbose=False):

        super().__init__()
        self.lr_max = lr_max if lr_max else lr
        self.lrs = []
        self.total_steps, self.steps_per_epoch = total_steps, steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy, self.cycle_momentum = anneal_strategy, cycle_momentum
        self.base_momentum, self.max_momentum = base_momentum, max_momentum
        self.div_factor, self.final_div_factor = div_factor, final_div_factor
        self.three_phase = three_phase
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.scheduler = None

    def before_fit(self, dls, opt, n_epochs, scheduler_dict):
        if not self.steps_per_epoch: self.steps_per_epoch = len(dls.train)
        self.lrs = []  # store lr values

        if isinstance(opt, SAM):
            opt = opt.base_optimizer

        self.scheduler = lr_scheduler.OneCycleLR(optimizer=opt,
                                                 max_lr=self.lr_max,
                                                 total_steps=self.total_steps,
                                                 epochs=n_epochs,
                                                 steps_per_epoch=self.steps_per_epoch,
                                                 pct_start=self.pct_start,
                                                 anneal_strategy=self.anneal_strategy,
                                                 cycle_momentum=self.cycle_momentum,
                                                 base_momentum=self.base_momentum,
                                                 max_momentum=self.max_momentum,
                                                 div_factor=self.div_factor,
                                                 final_div_factor=self.final_div_factor,
                                                 three_phase=self.three_phase,
                                                 last_epoch=self.last_epoch,
                                                 verbose=self.verbose
                                                 )
        if scheduler_dict:
            self.scheduler.load_state_dict(scheduler_dict)

    def after_fit(self, **kwargs):
        # TODO: Move this to abstract method
        pass

    def after_batch_train(self, model, loss):
        if model.training:
            self.scheduler.step()
            self.lrs.append(self.scheduler.get_last_lr()[0])


class LRFinderSupport(AbstractSupportClass):
    def __init__(self, start_lr=1e-7, end_lr=10, num_iter=100, step_mode='exp', beta=0.98, suggestion='valley'):
        self.start_lr, self.end_lr = start_lr, end_lr
        self.num_iter = num_iter
        self.step_mode = step_mode
        self.losses, self.lrs = [], []
        self.best_loss, self.aver_loss = inf, 0
        self.train_iter = 0

        if beta >= 1:
            raise ValueError("`num_iter` must be smaller than 1")
        else:
            self.beta = beta
        self.suggestion = suggestion

    def before_fit(self, dls, opt, n_epochs):
        self.losses, self.lrs = [], []
        self.best_loss, self.aver_loss = inf, 0
        self.train_iter = 0

        # set base_lr for the optimizer
        self.set_lr(self.start_lr, opt)

        # check num_iter 
        if not self.num_iter: self.num_iter = len(dls.train)
        # if self.num_iter > len(self.dls.train): self.num_iter = len(self.dls.train)

        # Initialize the proper learning rate policy
        if self.step_mode.lower() == "exp":
            self.scheduler = ExponentialLR(opt, self.end_lr, self.num_iter)
        elif self.step_mode.lower() == "linear":
            self.scheduler = LinearLR(opt, self.end_lr, self.num_iter)

    def after_batch_train(self, model, loss):
        self.train_iter += 1
        self.scheduler.step()
        self.lrs.append(self.scheduler.get_last_lr()[0])

        # update smooth loss
        self.smoothing(self.beta, loss)
        if self.smoothed_loss < self.best_loss: self.best_loss = self.smoothed_loss
        # Stop if the loss is exploding
        if self.smoothed_loss > 4 * self.best_loss:
            raise KeyboardInterrupt  # stop fit method
        if self.train_iter > self.num_iter:
            raise KeyboardInterrupt  # stop fit method

    def after_fit(self, opt):
        # reset the gradients
        opt.zero_grad()
        if self.suggestion == 'valley':
            self.suggested_lr = valley(self.lrs, self.losses)

    def smoothing(self, beta, loss):
        # Smooth the loss if beta is specified        
        self.aver_loss = beta * self.aver_loss + (1 - beta) * loss.detach().item()
        self.smoothed_loss = self.aver_loss / (1 - beta ** self.train_iter)
        self.losses.append(self.smoothed_loss)

    def set_lr(self, lrs, opt):
        if not isinstance(lrs, list): lrs = [lrs] * len(opt.param_groups)
        if len(lrs) != len(opt.param_groups):
            raise ValueError(
                "Length of `lrs` is not equal to the number of parameter groups "
                + "in the given optimizer")
        # update lr
        for param_group, lr in zip(opt.param_groups, lrs):
            param_group["lr"] = lr

    def plot_lr_find(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.lrs, self.losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        plt.grid()


class LinearLR(LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        if num_iter <= 1: raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = (self.last_epoch + 1) / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.last_epoch = last_epoch
        if num_iter <= 1: raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = (self.last_epoch + 1) / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


def valley(lrs: list, losses: list):
    "Suggests a learning rate from the longest valley and returns its index"
    n = len(losses)
    max_start, max_end = 0, 0

    # find the longest valley
    lds = [1] * n
    for i in range(1, n):
        for j in range(0, i):
            if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]

    sections = (max_end - max_start) / 3
    idx = max_start + int(sections) + int(sections / 2)

    return float(lrs[idx])


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def update_lr_schedule(optimizer):
    if isinstance(optimizer, SAM):
        optimizer = optimizer.base_optimizer

    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = param_group["lr"] * param_group["lr_scale"]

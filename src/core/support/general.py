import os
import random

import numpy as np
import torch
from peft import PeftModel

from src.core.models.optim_factory import prepare_optimizer, create_optimizer
from src.core.utils.basics import convert_signal_to_image_reconstruction
from src.core.support.abstract_support_class import AbstractSupportClass
from src.core.support.tracking import join_path_file
from src.core.utils.learner_utils import get_model_from_module


class ResultPrinter(AbstractSupportClass):
    def __init__(self):
        super().__init__()

    def get_header(self, recorder, include_time):
        "recorder is a dictionary"
        header = list(recorder.keys())
        return header + ['time'] if include_time else header

    def before_fit(self, run_finder, recorder, global_rank, print_format, include_time=True):
        header = self.get_header(recorder, include_time=include_time)

        self.print_header, self.print_format = '{:>25s}' * len(header), print_format

        if global_rank != 0: return
        if run_finder: return  # don't print if lr_finder is called
        if recorder is None: return  # don't print if there is no recorder
        print(self.print_header.format(*header))

    def after_epoch(self, run_finder, recorder, epoch_time):
        if run_finder: return  # don't print if lr_finder is called
        if recorder is None: return  # don't print if there is no recorder
        epoch_logs = []
        for key in recorder:
            value = recorder[key][-1] if recorder[key] else None
            epoch_logs += [value]
        if epoch_time: epoch_logs.append(epoch_time)

        print(self.print_format.format(*epoch_logs))


class Tracker(AbstractSupportClass):
    def __init__(self, monitor='train_loss', comp=None, min_delta=0.):
        super().__init__()
        if comp is None:
            comp = np.less if 'loss' in monitor or 'error' in monitor else np.greater
        if comp == np.less:
            min_delta *= -1
        self.monitor, self.comp, self.min_delta = monitor, comp, min_delta
        self.best = None

    def before_fit(self, run_finder, recorder):
        if run_finder:
            return
        if self.best is None:
            self.best = float('inf') if self.comp == np.less else -float('inf')
        self.monitor_names = list(recorder.keys())
        assert self.monitor in self.monitor_names

    def after_epoch(self, run_finder, recorder, epoch, model, opt, scheduler):
        if run_finder:
            return
        val = recorder[self.monitor][-1]
        if self.comp(val - self.min_delta, self.best):
            self.best, self.new_best = val, True
        else:
            self.new_best = False


class ModelSaver(Tracker):
    def __init__(self, n_epochs, args, monitor='train_loss', comp=None, min_delta=0.,
                 every_epoch=10, fname='model', path=None, save_process_id=0, global_rank=0):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)
        self.n_epochs = n_epochs
        self.every_epoch = every_epoch
        self.path, self.fname = path, fname
        self.save_process_id, self.global_rank = save_process_id, global_rank
        self._best_save_path = join_path_file(f'{fname}_best', path, ext='.pt')
        self._snapshot_filename = join_path_file(f'{self.fname}_snapshot', self.path, ext='.pt')
        self.args = args

    @property
    def best_save_path(self):
        return self._best_save_path

    def after_epoch(self, run_finder, recorder, epoch, model, opt, scheduler):
        # Only require one of the instances to save the model, when running in distributed mode
        if self.global_rank != self.save_process_id:
            return

        if ((epoch % self.every_epoch) == 0) or (epoch == self.n_epochs - 1):
            print(f'Saved model and snapshot at epoch {epoch}')
            self.save(f'{self.fname}_every_{self.every_epoch}_epoch',
                      self.path, model, opt)

        # Either way save the model if we have a new best
        super().after_epoch(run_finder, recorder, epoch, model, opt, scheduler=scheduler)
        if self.new_best:
            print(f'Better model found at epoch {epoch} with {self.monitor} value: {self.best}.')
            self.save(f'{self.fname}_best', self.path, model, opt)

        # Save snapshot after every epoch in case training is interrupted
        self.save_snapshot(epochs_run=epoch, model=model, opt=opt, scheduler=scheduler)

    def after_fit(self):
        # Clear the snapshots so other runs do not simply reload them.
        print('\nRemoving snapshots')
        if self.snapshot_exists() and self.global_rank == self.save_process_id:
            os.remove(self._snapshot_filename)

    def save(self, fname, path, model, opt, pickle_protocol=2):
        """
        Save model and optimizer state (if `with_opt`) to `self.path/file`
        """

        fname = join_path_file(fname, path, ext='.pt')

        state = {}
        if opt is not None:
            state['opt'] = opt.state_dict()

        if isinstance(model, PeftModel):
            model.save_pretrained(os.path.splitext(fname)[0])
        else:
            state['model'] = get_model_from_module(model).state_dict()

        torch.save(state, fname, pickle_protocol=pickle_protocol)

        return fname

    def load(self, fname, model, opt, device, strict=True, is_trainable=False, **kwargs):
        """
        load the model
        """
        state = torch.load(fname, map_location=torch.device(device))

        if isinstance(model, PeftModel):
            # Optimizer must be re-instantiated for PEFT models
            model = PeftModel.from_pretrained(model.base_model.model, os.path.splitext(fname)[0],
                                              is_trainable=is_trainable)
            opt = prepare_optimizer(model=model, args=self.args) if self.args.pretrained_model_path is not None else \
                create_optimizer(model=model, args=self.args, filter_bias_and_bn=False)
        else:
            model_state = state['model']
            get_model_from_module(model).load_state_dict(model_state, strict=strict)

        return model, opt

    def snapshot_exists(self):
        return os.path.exists(self._snapshot_filename)

    def save_snapshot(self, epochs_run, model, opt, scheduler):
        if isinstance(model, PeftModel):
            '''
            Saving and reloading the peft snapshot isn't exactly straightforward as the optimizer state does not load
            properly. It's easier to simply restart training as PEFT is only used for fine-tuning anyways.
            '''
            return

        snapshot = {'model': get_model_from_module(model).state_dict(), 'opt': opt.state_dict(),
                    'epochs_run': epochs_run, 'best_validation': self.best}

        if not isinstance(scheduler, AbstractSupportClass):
            snapshot['scheduler'] = scheduler.state_dict()

        torch.save(snapshot, self._snapshot_filename)

    def load_snapshot(self, model, opt):
        if isinstance(model, PeftModel):
            '''
            Saving and reloading the peft snapshot isn't exactly straightforward as the optimizer state does not load
            properly. It's easier to simply restart training as PEFT is only used for fine-tuning anyways.
            '''
            return model, opt, -1, None

        print('Loading snapshot. Starting from epoch', end=': ')
        snapshot = torch.load(self._snapshot_filename)

        epochs_run = snapshot['epochs_run']
        self.best = snapshot['best_validation']

        get_model_from_module(model).load_state_dict(snapshot['model'])
        opt.load_state_dict(snapshot['opt'])

        scheduler = snapshot['scheduler'] if 'scheduler' in snapshot.keys() else None

        print(f'{epochs_run}')

        return model, opt, epochs_run, scheduler


class EarlyStopper(Tracker):
    def __init__(self, monitor='train_loss', comp=None, min_delta=0,
                 patient=5):
        self.patient = patient
        self.impatient_level = 0

        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)

    def before_fit(self, run_finder, recorder):
        # set the impatient level
        self.impatient_level = 0
        super().before_fit(run_finder, recorder)

    def after_epoch(self, run_finder, recorder, epoch, opt, model=None, scheduler=None):
        super().after_epoch(run_finder, recorder, epoch, model, opt, scheduler)
        if self.new_best:
            self.impatient_level = 0  # reset the impatience
        else:
            self.impatient_level += 1
            if self.impatient_level > self.patient:
                print(f'No improvement since epoch {epoch - self.impatient_level}: early stopping')
                raise KeyboardInterrupt


class ReconstructionPlotter(AbstractSupportClass):
    def __init__(self, save_path, every_n_epoch=1):
        """
        every_n_epoch:  Epoch % every_n_epoch decides when the method will plot the original/reconstruction pair.
                        Default 1.
        """
        self.original, self.reconstruction, self.best_loss, self.batch_mask = None, None, None, None
        self.every_n_epoch = every_n_epoch
        self.save_path = os.path.join(save_path, 'plots')

    def reset(self):
        self.original, self.reconstruction, self.best_loss, self.batch_mask = None, None, None, None

    def before_epoch(self):
        self.reset()

    def after_batch_valid(self, loss, original, reconstruction, mask):
        if not self.best_loss or loss < self.best_loss:
            self.best_loss = loss.detach().cpu()
            self.original = original.detach().cpu()
            self.reconstruction = reconstruction.detach().cpu()
            self.batch_mask = mask.detach().cpu()

    def after_epoch(self, epoch, n_epochs):
        if epoch % self.every_n_epoch != 0 and not epoch == n_epochs - 1:
            return
        # Get a random index from the best batch and plot this original/reconstruction pair
        rand_index = random.randint(0, len(self.original) - 1)

        # Convert the mask to a seq_len numpy array for use in the image plotter function
        mask = (self.batch_mask[rand_index]).reshape(self.batch_mask.shape[1], -1)

        # Splice the original with the reconstruction
        spliced_signal = torch.mul(self.reconstruction[rand_index], (~mask.bool()).int()) \
                         + torch.mul(self.original[rand_index], mask)

        # Convert the original/reconstruction pairs to images
        convert_signal_to_image_reconstruction(self.original[rand_index],
                                               None, title=f"Original Epoch={epoch}", save_path=self.save_path,
                                               mask=mask)
        convert_signal_to_image_reconstruction(self.reconstruction[rand_index],
                                               None, title=f"Reconstructed Epoch={epoch}", save_path=self.save_path,
                                               mask=mask)
        convert_signal_to_image_reconstruction(spliced_signal,
                                               None, title=f"Reconstructed Overlayed with Original Epoch={epoch}",
                                               save_path=self.save_path, mask=mask)

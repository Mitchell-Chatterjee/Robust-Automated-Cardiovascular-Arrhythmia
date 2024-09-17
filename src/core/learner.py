import copy
from enum import Enum

import einops
import torch.nn
from ray.air import session
from torch import nn

from src.core.models.encoder_decoder_vit import EncoderDecoderViT
from src.core.models.lstm.models.cpc import CPCModel
from src.core.utils.basics import *
from src.core.support.general import *
from src.core.support.patch_mask import FinetunePatchSupport, PatchMaskSupport
from src.core.support.scheduler import *
from src.core.support.tracking import *
from src.core.support.tracking import EvalTracker, PredictionTracker
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from src.core.utils.learner_utils import get_model_from_module
from src.core.constants.definitions import DataAugmentation, Mode


class EpochType(Enum):
    TRAIN = 'train'
    VALID = 'validation'
    PREDICT = 'predict'
    TEST = 'test'


class Learner:

    def __init__(self,
                 dls,
                 train_sampler,
                 model,
                 local_rank,
                 global_rank,
                 opt,
                 scheduler,
                 loss_func,
                 label_type,
                 normalization,
                 mode: Mode,
                 classification_type: ClassificationType,
                 model_saver: ModelSaver,
                 training_tracker: TrainingTracker,
                 eval_tracker: EvalTracker,
                 prediction_tracker: PredictionTracker,
                 patch_masker: FinetunePatchSupport | PatchMaskSupport,
                 data_augmentation=DataAugmentation.none,
                 clip_grad=None,
                 distributed=True,
                 run_finder=False,
                 mentor_mix=False,
                 mentor_mix_start_epoch=0,
                 mentor_mix_alpha=1,
                 mentor_mix_gamma=0.9,
                 lr=1e-3,
                 reconstruction_plotter: ReconstructionPlotter = AbstractSupportClass(),
                 early_stopping: EarlyStopper = AbstractSupportClass()):

        # Dataloaders, model and learning rate
        self.dls, self.train_sampler, self.model, self.lr = dls, train_sampler, model.to(local_rank), lr
        # Optimization and loss functions
        self.opt, self.loss_func = opt, loss_func
        self.scheduler, self.scheduler_dict = scheduler, None
        self.label_type, self.classification_type = label_type, classification_type

        # Device (local rank) and global rank (relevant when running on multiple nodes)
        self.local_rank, self.global_rank, self.distributed, self.mode = local_rank, global_rank, distributed, mode

        # Gradient clipping
        self.clip_grad = clip_grad

        # Indicator of running lr_finder
        self.run_finder = run_finder

        # Global bookkeeping variables.
        self.epochs_run, self.n_epochs = 0, None

        # Default support classes for reducing class complexity
        self.timer_tracker = TimeTracker()
        self.result_printer = ResultPrinter()

        # These support class requires instantiation by the calling method
        self.model_saver = model_saver
        self.training_tracker = training_tracker
        self.eval_tracker = eval_tracker
        self.prediction_tracker = prediction_tracker

        # MentorMixup
        self.ema = 0
        self.beta_dist = torch.distributions.beta.Beta(mentor_mix_alpha, mentor_mix_alpha)
        self.gamma = mentor_mix_gamma
        self.mentor_mix = mentor_mix
        self.mentor_mix_start_epoch = mentor_mix_start_epoch

        # Test time augmentation
        self.data_augmentation = data_augmentation

        # Potentially null support classes. At the discretion of the calling methods.
        self.early_stopping = early_stopping
        self.patch_masker = patch_masker
        self.normalization = normalization
        self.reconstruction_plotter = reconstruction_plotter if global_rank == 0 else AbstractSupportClass()

        # Initialization
        assert isinstance(self.model_saver, ModelSaver), "Must instantiate the support class to indicate how to save" \
                                                         "and load the model"

        # If we have a previous snapshot, reload that
        if self.model_saver.snapshot_exists():
            self.model, self.opt, self.epochs_run, self.scheduler_dict = \
                self.model_saver.load_snapshot(self.model, self.opt)
            self.model = self.model.to(self.local_rank)
            # Increment the epochs run by one from where it was saved
            self.epochs_run += 1

        # Print the device we are training the current process on
        print(f'Device: {torch.device(self.global_rank)}')
        if self.global_rank == 0:
            # Print statement letting us know the batch size per process and the number of steps required
            print(f"\nBatchsize: {len(next(iter(self.dls.train))[0])} | Steps: {len(self.dls.train)}\n")

    # Begin helper methods
    def before_fit(self):
        # Ensure not already complete
        if self.epochs_run == self.n_epochs:
            print("Training complete")
            exit()

        self.timer_tracker.before_fit()
        self.training_tracker.before_fit()

        # Tracker sub-classes
        self.early_stopping.before_fit(run_finder=self.run_finder, recorder=self.training_tracker.recorder)
        self.model_saver.before_fit(run_finder=self.run_finder, recorder=self.training_tracker.recorder)
        self.result_printer.before_fit(run_finder=self.run_finder, recorder=self.training_tracker.recorder,
                                       global_rank=self.global_rank, print_format=self.training_tracker.print_format)
        self.scheduler.before_fit(dls=self.dls, opt=self.opt, n_epochs=self.n_epochs,
                                  scheduler_dict=self.scheduler_dict)

    def after_epoch(self):
        self.training_tracker.after_epoch(epoch=self.epochs_run, gpu_id=self.global_rank)

        # Tracker sub-classes
        self.early_stopping.after_epoch(run_finder=self.run_finder, recorder=self.training_tracker.recorder,
                                        epoch=self.epochs_run, model=self.model, opt=self.opt)
        self.result_printer.after_epoch(run_finder=self.run_finder, recorder=self.training_tracker.recorder,
                                        epoch_time=self.timer_tracker.epoch_time)
        self.model_saver.after_epoch(run_finder=self.run_finder, recorder=self.training_tracker.recorder,
                                     epoch=self.epochs_run, model=self.model, opt=self.opt,
                                     scheduler=self.scheduler)

    def _to_device(self, batch):
        xb, yb = (b.to(self.local_rank, non_blocking=False) for b in batch)
        return xb, yb

    # End of helper methods

    def _fit(self, do_valid=True):
        """fit the model"""
        if not self.dls.valid:
            do_valid = False

        self.before_fit()

        for self.epochs_run in range(self.epochs_run, self.n_epochs):
            if self.distributed:
                self.train_sampler.set_epoch(self.epochs_run)
            # Do a training epoch
            self.epoch_train()
            # Do a validation epoch
            if do_valid:
                self.epoch_validate()
            self.after_epoch()

    def fit_one_cycle(self, n_epochs):
        self.n_epochs = n_epochs
        self._fit()

    def epoch_train(self):
        # Before epoch train
        self.timer_tracker.before_epoch_train()
        self.training_tracker.before_epoch_train()

        self.model.train()
        self.all_batches(EpochType.TRAIN, self.dls.train)

        # After epoch train
        self.timer_tracker.after_epoch_train()
        self.training_tracker.after_epoch_train()

    def epoch_validate(self):
        # Before epoch validation
        self.training_tracker.before_epoch_valid()
        self.reconstruction_plotter.before_epoch()

        # model at evaluation mode
        self.model.eval()
        with torch.no_grad():
            self.all_batches(EpochType.VALID, self.dls.valid)

        # After epoch validation
        self.training_tracker.after_epoch_valid()
        self.reconstruction_plotter.after_epoch(epoch=self.epochs_run, n_epochs=self.n_epochs)

    def all_batches(self, type_, dataloader):
        for batch in dataloader:
            if type_ == EpochType.TRAIN:
                self.batch_train(batch=batch)
            elif type_ == EpochType.VALID:
                self.batch_validate(batch=batch)
            elif type_ == EpochType.PREDICT:
                self.batch_predict(batch=batch)
            elif type_ == EpochType.TEST:
                self.batch_test(batch=batch)

    def mentormix(self, xb, yb, indices=None):
        # Weight (lower values are given higher weight)
        pred, yb_2 = self.model_forward(xb=xb, yb=yb, indices=indices)
        weights = torch.nn.functional.binary_cross_entropy_with_logits(pred, yb_2.type(self.label_type),
                                                                       reduction='none')
        weights = torch.mean(weights, dim=1)

        self.ema = self.gamma * torch.mean(weights, dim=0) + (1 - self.gamma) * self.ema

        # MentorNet implemented by thresholding
        weights = torch.where(weights <= self.ema, 1., 0.)
        weights = torch.nn.functional.softmax(weights, dim=0).cpu()

        # Importance sampling
        sampling_indices = torch.multinomial(weights, xb.shape[0], replacement=True).cpu()

        # Mixup
        mixup_lambda = self.beta_dist.sample(sample_shape=weights.shape).cpu()
        mixup_lambda = (weights * mixup_lambda.apply_(lambda x: max(x, 1 - x))
                        + (1 - weights) * mixup_lambda.apply_(lambda x: min(x, 1 - x))).to(xb.device)

        mixup_samples, mixup_labels = xb[sampling_indices], yb[sampling_indices]

        xb = mixup_lambda[:, None, None] * xb + (1 - mixup_lambda[:, None, None]) * mixup_samples
        yb = mixup_lambda[:, None] * yb + (1 - mixup_lambda[:, None]) * mixup_labels

        return xb, yb

    def batch_train(self, batch):
        # forward + get loss + backward + optimize
        if not isinstance(self.scheduler, AbstractSupportClass):
            update_lr_schedule(optimizer=self.opt)
        # get the inputs and move to device
        xb, yb = self._to_device(batch=batch)
        # Before forward
        if not isinstance(get_model_from_module(self.model), EncoderDecoderViT):
            xb = self.normalization.before_forward(xb=xb)
        indices, yb_temp = None, None

        if self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.per_lead_aug:
            yb = einops.repeat(yb, 'm k -> m n k', n=xb.shape[1]).reshape(xb.shape[0] * xb.shape[1], yb.shape[1])
        elif self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.test_time_aug_transformer:
            indices = self.patch_masker.get_chunk_patch_indices(xb=xb)
        elif self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.dropout_aug_transformer:
            indices = self.patch_masker.get_random_patch_indices(xb=xb)

        # Perform basic mentor mix
        if self.mentor_mix and self.epochs_run >= self.mentor_mix_start_epoch:
            with torch.no_grad():
                yb_temp = copy.deepcopy(yb) # Need this value for tracking training stats
                xb, yb = self.mentormix(xb, yb, indices=indices)

        if isinstance(self.opt, SAM):
            # first forward-backward pass
            pred, target = self.model_forward(xb=xb, yb=yb, indices=indices)
            loss = self.loss_func(pred, target.type(self.label_type))  # use this loss for any training statistics
            loss.backward()
            self.opt.first_step(zero_grad=True)

            # second forward-backward pass
            pred2, target2 = self.model_forward(xb=xb, yb=yb, indices=indices)
            self.loss_func(pred2, target2.type(self.label_type)).backward()  # make sure to do a full forward pass
            self.opt.second_step(zero_grad=True)
        else:
            # forward
            pred, yb = self.model_forward(xb=xb, yb=yb, indices=indices)
            # compute loss
            loss = self.loss_func(pred, yb.type(self.label_type))
            # zero the parameter gradients
            self.opt.zero_grad()
            # gradient
            loss.backward()
            # clip gradients if parameter is included
            if self.clip_grad:
                nn.utils.clip_grad_norm_(get_model_from_module(self.model).parameters(), self.clip_grad)
            # update weights
            self.opt.step()

        # If mentor mix is not yet being run, still update the moving average
        if self.mentor_mix and self.epochs_run < self.mentor_mix_start_epoch:
            self.ema = self.gamma * torch.mean(loss, dim=0) + (1 - self.gamma) * self.ema

        self.training_tracker.after_batch_train(batch=(xb, yb_temp if yb_temp is not None else yb), loss=loss, pred=pred)
        self.scheduler.after_batch_train(model=self.model, loss=loss)

    def batch_validate(self, batch):
        # forward + calculate loss
        # get the inputs and move to device
        xb, yb = self._to_device(batch=batch)
        # Before forward
        if not isinstance(get_model_from_module(self.model), EncoderDecoderViT):
            xb = self.normalization.before_forward(xb=xb)

        if self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.per_lead_aug:
            # Test time augmentation over multiple leads
            pred = torch.mean(torch.sigmoid(self.model_forward(xb=xb,
                                                               yb=einops.repeat(yb, 'm k -> m n k', n=xb.shape[1]))[0]),
                              dim=1)
            # TODO: Fix this to be dynamic. For now we know it will be a bce loss
            loss = F.binary_cross_entropy(pred, yb.type(self.label_type))
        elif self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.test_time_aug_transformer:
            indices = self.patch_masker.get_test_time_augmentation_indices(xb=xb)
            pred = torch.mean(
                torch.stack([torch.sigmoid(self.model_forward(xb=xb, yb=yb, indices=indices[i])[0])
                             for i in range(len(indices))]),
                dim=0)
            # TODO: Fix this to be dynamic. For now we know it will be a bce loss
            loss = F.binary_cross_entropy(pred, yb.type(self.label_type))
        elif self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.test_time_aug_cpc:
            # Aggregate predictions (Test time augmentation)
            pred = torch.mean(
                torch.stack([torch.sigmoid(self.model_forward(xb=xb[..., i, :].squeeze(), yb=yb)[0])
                             for i in range(xb.shape[xb.ndim - 2])]),
                dim=0)
            # TODO: Fix this to be dynamic. For now we know it will be a bce loss
            loss = F.binary_cross_entropy(pred, yb.type(self.label_type))
        else:
            # forward
            pred, yb = self.model_forward(xb=xb, yb=yb)
            # compute loss
            loss = self.loss_func(pred, yb.type(self.label_type))

        # After batch validation
        self.training_tracker.after_batch_valid(batch=batch, loss=loss, pred=pred)

        if self.classification_type == ClassificationType.PRETRAIN \
                and not isinstance(get_model_from_module(self.model), CPCModel):
            yb, pred = self.patch_masker.after_batch_valid(yb=yb, pred=pred)  # [bs x n_vars x seq_len]
            self.reconstruction_plotter.after_batch_valid(loss=loss, original=yb, reconstruction=pred,
                                                          mask=self.patch_masker.binary_mask)

    def batch_predict(self, batch):
        # get the inputs and move to device
        xb, yb = self._to_device(batch=batch)
        # Before forward
        if not isinstance(get_model_from_module(self.model), EncoderDecoderViT):
            xb = self.normalization.before_forward(xb=xb)

        if self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.per_lead_aug:
            # Test time augmentation over multiple leads
            pred = torch.mean(torch.sigmoid(self.model_forward(xb=xb,
                                                               yb=einops.repeat(yb, 'm k -> m n k', n=xb.shape[1]))[0]),
                              dim=1)
        elif self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.test_time_aug_transformer:
            indices = self.patch_masker.get_test_time_augmentation_indices(xb=xb)
            pred = torch.mean(
                torch.stack([torch.sigmoid(self.model_forward(xb=xb, yb=yb, indices=indices[i])[0])
                             for i in range(len(indices))]),
                dim=0)
        elif self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.test_time_aug_cpc:
            # Aggregate predictions (Test time augmentation)
            pred = torch.mean(
                torch.stack([torch.sigmoid(self.model_forward(xb=xb[..., i, :].squeeze(), yb=yb)[0])
                             for i in range(xb.shape[xb.ndim - 2])]),
                dim=0)
        else:
            # forward
            pred, yb = self.model_forward(xb=xb, yb=yb)

        return pred, yb

    def batch_test(self, batch):
        # get the inputs and move to device
        xb, yb = self._to_device(batch=batch)
        # Before forward
        if not isinstance(get_model_from_module(self.model), EncoderDecoderViT):
            xb = self.normalization.before_forward(xb=xb)

        if self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.per_lead_aug:
            # Test time augmentation over multiple leads
            pred = torch.mean(torch.sigmoid(self.model_forward(xb=xb,
                                                               yb=einops.repeat(yb, 'm k -> m n k', n=xb.shape[1]))[0]),
                              dim=1)
        elif self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.test_time_aug_transformer:
            indices = self.patch_masker.get_test_time_augmentation_indices(xb=xb)
            pred = torch.mean(
                torch.stack([torch.sigmoid(self.model_forward(xb=xb, yb=yb, indices=indices[i])[0])
                             for i in range(len(indices))]),
                dim=0)
        elif self.mode is Mode.CLASSIFICATION and self.data_augmentation is DataAugmentation.test_time_aug_cpc:
            # Aggregate predictions (Test time augmentation)
            pred = torch.mean(
                torch.stack([torch.sigmoid(self.model_forward(xb=xb[..., i, :].squeeze(), yb=yb)[0])
                             for i in range(xb.shape[xb.ndim - 2])]),
                dim=0)
        else:
            # forward
            pred, yb = self.model_forward(xb=xb, yb=yb)

        # After batch test
        self.eval_tracker.after_batch_eval(pred=pred, yb=yb)

    def model_forward(self, xb, yb, indices=None):
        if self.mode == Mode.PRETRAIN:
            # Normalize reconstruction target
            yb = self.normalization.before_forward(xb=yb)

        if isinstance(get_model_from_module(self.model), CPCModel):
            # Handles the difference between pre-training and fine-tuning
            pred, yb = self.model(xb) if self.mode == Mode.PRETRAIN else (self.model(xb), yb)
        else:
            # Apply patch masking
            xb, yb, mask, inverted_mask = self.patch_masker.before_forward(input=xb, target=yb, device=self.local_rank)
            # Do prediction
            pred = self.model(xb, mask=mask, inverted_mask=inverted_mask, indices=indices,
                              augmentation=self.data_augmentation)
        # Ensure pred and yb are same shape
        pred = pred.reshape(shape=yb.shape)

        return pred, yb

    # External methods
    def predict(self, dataloader, weight_path=None):
        """_summary_
        Args:
            dataloader is one of the dataloaders returned when calling get_dls
        Returns:
            _type_: _description_
        """
        if dataloader is None:
            return
        if weight_path is not None:
            self.model, _ = self.model_saver.load(fname=weight_path, model=self.model, opt=None, device=self.local_rank)

        # Before prediction
        self.prediction_tracker.before_predict()

        self.model.eval()  # model at evaluation mode
        with torch.no_grad():
            self.all_batches(EpochType.PREDICT, dataloader)

        # After prediction
        self.prediction_tracker.after_predict()

        predictions_list = self.prediction_tracker.preds
        return to_numpy(predictions_list)

    def eval(self, dataloader, save_path, class_sizes, split, per_class_only=False, weight_path=None):
        """_summary_
        Args:
            dataloader is one of the dataloaders returned when calling get_dls
        Returns:
            _type_: _description_
        """
        if dataloader is None:
            return
        if weight_path is not None:
            self.model, _ = self.model_saver.load(fname=weight_path, model=self.model, opt=None, device=self.local_rank)

        # Before test
        self.eval_tracker.before_eval(split)
        self.result_printer.before_fit(run_finder=self.run_finder, recorder=self.eval_tracker.recorder,
                                       global_rank=self.global_rank, print_format=self.eval_tracker.print_format,
                                       include_time=False)

        self.model.eval()
        with torch.no_grad():
            self.all_batches(EpochType.TEST, dataloader)

        # After test
        self.eval_tracker.after_eval(save_path=save_path, class_sizes=class_sizes, per_class_only=per_class_only)
        self.result_printer.after_epoch(run_finder=self.run_finder, recorder=self.eval_tracker.recorder,
                                        epoch_time=None)
        print()

    def fine_tune(self, n_epochs, freeze_epochs, lora=False):
        """
        fintune the pretrained model. First the entire model is freezed, only head is trained
        up to a freeze_epochs number. Then the model is unfreezed and the entire model is trained
        """
        assert (n_epochs > 0) | (freeze_epochs > 0), "Either n_epochs or freeze_epochs has to be > 0"

        update_lr_schedule(optimizer=self.opt)

        # TODO: Reimplement this
        # Finetune the head of freeze_epochs > 0:
        if freeze_epochs > 0:
            # Reload model and avoid potential error by avoiding unnessary gradient computations
            self.model = DDP(get_model_from_module(self.model), device_ids=[self.local_rank],
                             find_unused_parameters=True) if self.distributed else self.model

            print('Finetune the head')
            self.freeze()
            self.fit_one_cycle(freeze_epochs)

            print('Loading best model from head finetuning')
            self.model, self.opt = self.model_saver.load(fname=self.model_saver.best_save_path, model=self.model,
                                                         opt=self.opt, device=self.local_rank, is_trainable=True)
            self.epochs_run += 1

        # Finetune the entire network if n_epochs > 0
        if n_epochs > 0:
            print('\n\n')
            print('Finetune the entire network')
            self.unfreeze(lora=lora)
            self.fit_one_cycle(freeze_epochs + n_epochs)

    def hyper_param_tuning(self, n_epochs):
        self.n_epochs = n_epochs

        update_lr_schedule(optimizer=self.opt)

        self.before_fit()

        for self.epochs_run in range(self.epochs_run, self.n_epochs):
            if self.distributed:
                self.train_sampler.set_epoch(self.epochs_run)
            # Do a training epoch
            self.epoch_train()
            # Do a validation epoch
            self.epoch_validate()
            self.after_epoch()

            session.report(
                {"loss": self.training_tracker.recorder['valid_loss'][-1],
                 "AUROC": self.training_tracker.recorder['valid_AUROC'][-1],
                 "epoch": self.epochs_run}
            )


    def linear_probe(self, n_epochs, cls_token=False, base_lr=None, pct_start=0.3):
        """
        linear probing the pretrained model. The model is freeze except the head during finetuning
        """
        assert (n_epochs > 0), "n_epochs has to be > 0"
        if not base_lr:
            base_lr = self.lr
        print('Linear probing the model')
        self.freeze()
        self.fit_one_cycle(n_epochs)

    def lr_finder(self, start_lr=1e-7, end_lr=10, num_iter=100, step_mode='exp', show_plot=True, suggestion='valley'):
        """
        find the learning rate
        """
        self.n_epochs = num_iter // len(self.dls.train) + 1

        # add LRFinderCB to support list and will remove later
        lr_finder_support = LRFinderSupport(start_lr, end_lr, num_iter, step_mode, suggestion=suggestion)

        # fit
        self._fit(do_valid=False)

        if show_plot:
            lr_finder_support.plot_lr_find()
        if suggestion:
            return lr_finder_support.suggested_lr

    def freeze(self):
        """
        freeze the model head
        require the model to have head attribute
        """
        head_list = [param for name, param in get_model_from_module(self.model).named_parameters()
                     if 'head' in name]
        if len(head_list) > 0:
            # print('model head is available')
            for param in get_model_from_module(self.model).parameters():
                param.requires_grad = False
            for param in head_list:
                param.requires_grad = True
            # print('model is frozen except the head')

    def unfreeze(self, lora):
        for name, param in get_model_from_module(self.model).named_parameters():
            if lora and 'lora' not in name.lower():
                continue
            param.requires_grad = True

    def load_model(self, weight_path):
        if weight_path is not None:
            self.model, _ = self.model_saver.load(fname=weight_path, model=self.model, opt=None, device=self.local_rank)
    # End of external methods

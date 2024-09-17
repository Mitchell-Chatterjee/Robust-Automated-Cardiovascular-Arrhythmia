import os
import time
from pathlib import Path

import pandas as pd
import torch

from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelPrecision, \
    MultilabelRecall, MultilabelAccuracy, MultilabelStatScores
from torcheval.metrics import Metric, MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, \
    MulticlassAUROC, MulticlassAUPRC, MultilabelAUPRC, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, \
    BinaryAUROC, BinaryAUPRC
from functools import partial

from src.core.datasets.ecg_interface import ClassificationType
from src.core.support.abstract_support_class import AbstractSupportClass
from src.core.constants.definitions import Mode


class TimeTracker(AbstractSupportClass):
    def __init__(self):
        self.epoch_time, self.start_time = None, None

    def before_fit(self):
        self.epoch_time = None

    def before_epoch_train(self):
        self.start_time = time.time()

    def after_epoch_train(self):
        self.epoch_time = self.format_time(time.time() - self.start_time)

    def format_time(self, t):
        "Format `t` (in seconds) to (h):mm:ss"
        t = int(t)
        h, m, s = t // 3600, (t // 60) % 60, t % 60
        if h != 0:
            return f'{h}:{m:02d}:{s:02d}'
        else:
            return f'{m:02d}:{s:02d}'


class TrainingTracker(AbstractSupportClass):

    def __init__(self, loss_func, mode: Mode, number_of_classes, save_path, classification_type):
        self.loss_func = loss_func
        self.mode = mode
        self.mean_reduction_ = False
        self.recorder, self.batch_recorder = {}, {}
        self.predictions, self.labels = [], []
        self.number_of_classes = number_of_classes
        self.classification_type = classification_type
        self.save_path = f'{save_path}.csv'
        self.print_format = None

    def get_print_format(self):
        len_header = len(list(self.recorder.keys())) + 1  # Add 1 for time
        self.print_format = '{:>25d}' + '{:>25d}' + '{:>25.6f}' * (len_header - 3) + '{:>25}'

    def before_fit(self):
        self.initialize_recorder()
        self.get_print_format()
        if hasattr(self.loss_func, 'reduction'):
            self.mean_reduction_ = True if self.loss_func.reduction == 'mean' else False

    def initialize_recorder(self):
        if os.path.exists(self.save_path):
            recorder = load_recorders(save_path=self.save_path)
        else:
            recorder = {'gpu_id': [], 'epoch': [], 'train_loss': [], 'valid_loss': []}

            if self.mode == Mode.CLASSIFICATION:
                recorder['train_accuracy'], recorder['train_precision'], \
                    recorder['train_recall'], recorder['train_f1_score'], \
                    recorder['train_AUROC'], recorder['train_AUPRC'] = [], [], [], [], [], []
                recorder['valid_accuracy'], recorder['valid_precision'], \
                    recorder['valid_recall'], recorder['valid_f1_score'], \
                    recorder['valid_AUROC'], recorder['valid_AUPRC'] = [], [], [], [], [], []
        self.recorder = recorder

    def reset(self):
        self.batch_recorder = {'n_samples': [], 'batch_losses': []}
        self.labels, self.predictions = [], []

    def after_epoch(self, epoch, gpu_id):
        self.recorder['epoch'].append(epoch)
        self.recorder['gpu_id'].append(gpu_id)
        save_recorders(save_path=self.save_path, recorder=self.recorder)

    def before_epoch_train(self):
        # define storage for batch training loss and metrics
        self.reset()

    def before_epoch_valid(self):
        # if valid data is available, define storage for batch training loss and metrics
        self.reset()

    def after_epoch_train(self):
        values, _ = compute_scores(predictions=self.predictions, labels=self.labels,
                                   number_of_classes=self.number_of_classes, metrics_by_class=False,
                                   classification_type=self.classification_type, batch_recorder=self.batch_recorder,
                                   mode=self.mode)

        # save training loss after one epoch
        for key in values.keys():
            self.recorder[f'train_{key}'].append(values[key])

    def after_epoch_valid(self):
        values, _ = compute_scores(predictions=self.predictions, labels=self.labels,
                                   number_of_classes=self.number_of_classes, metrics_by_class=False,
                                   classification_type=self.classification_type, batch_recorder=self.batch_recorder,
                                   mode=self.mode)

        # save training loss after one epoch
        for key in values.keys():
            self.recorder[f'valid_{key}'].append(values[key])

    def after_batch_train(self, batch, loss, pred):
        self.accumulate(batch, loss, pred)

    def after_batch_valid(self, batch, loss, pred):
        self.accumulate(batch, loss, pred)

    def accumulate(self, batch, loss, pred):
        xb, yb = batch

        bs = len(xb)
        self.batch_recorder['n_samples'].append(bs)

        # get batch loss 
        loss = loss.detach() * bs if self.mean_reduction_ else loss.detach()
        self.batch_recorder['batch_losses'].append(loss)

        if self.mode == Mode.CLASSIFICATION:
            # Append data to list
            self.predictions.append(pred.data.cpu().detach())
            self.labels.append(yb.data.data.cpu().detach())


class EvalTracker(AbstractSupportClass):
    def __init__(self, number_of_classes, classification_type, metrics_by_class=False):
        self.predictions, self.labels = [], []
        self.recorder, self.per_class_recorder = {}, {}
        self.number_of_classes = number_of_classes
        self.classification_type = classification_type
        self.metrics_by_class = metrics_by_class
        self.print_format = None
        self.split = None
        super().__init__()

    def get_print_format(self):
        len_header = len(list(self.recorder.keys()))
        self.print_format = '{:>25.6f}' * len_header

    def before_eval(self, split):
        self.split = split
        self.initialize_recorder()
        self.get_print_format()

    def initialize_recorder(self):
        self.labels, self.predictions = [], []
        keys = [f'{self.split}_accuracy', f'{self.split}_precision', f'{self.split}_recall', f'{self.split}_f1_score',
                f'{self.split}_AUROC', f'{self.split}_AUPRC']
        self.recorder, self.per_class_recorder = {key: [] for key in keys}, {key: [] for key in keys}

    def after_batch_eval(self, pred, yb):
        # append the prediction after each forward batch
        self.predictions.append(pred.data.cpu().detach())
        self.labels.append(yb.data.cpu().detach())

    def after_eval(self, save_path, class_sizes, per_class_only=False):
        values, per_class_values = compute_scores(predictions=self.predictions, labels=self.labels,
                                                  number_of_classes=self.number_of_classes,
                                                  metrics_by_class=self.metrics_by_class,
                                                  classification_type=self.classification_type, batch_recorder=None,
                                                  mode=Mode.CLASSIFICATION)

        # save training loss after one epoch
        for key in values.keys():
            self.recorder[f'{self.split}_{key}'].append(values[key])

        if not per_class_only:
            save_recorders(save_path=f'{save_path}.csv', recorder=self.recorder)

        if per_class_values:
            save_per_class_recorders(save_path=f'{save_path}_per_class.csv', class_sizes=class_sizes,
                                     recorder=per_class_values)


class PredictionTracker(AbstractSupportClass):
    def __init__(self):
        self.preds = []
        super().__init__()

    def before_predict(self):
        self.preds = []

    def after_batch_predict(self, pred):
        # append the prediction after each forward batch
        self.preds.append(pred)

    def after_predict(self):
        self.preds = torch.concat(self.preds)  # .detach().cpu().numpy()


def compute_scores(predictions, labels, number_of_classes, metrics_by_class, batch_recorder, classification_type,
                   mode: Mode):
    values, per_class_values = {}, {}
    if batch_recorder:
        values['loss'] = sum(batch_recorder['batch_losses']).item() / sum(batch_recorder['n_samples'])

    # calculate metrics
    if mode == Mode.CLASSIFICATION:
        raw_predictions = torch.cat(predictions)
        labels = torch.cat(labels)

        if classification_type == ClassificationType.MULTI_CLASS:
            return compute_scores_multi_class(raw_predictions=raw_predictions, values=values,
                                              per_class_values=per_class_values, labels=labels,
                                              number_of_classes=number_of_classes, metrics_by_class=metrics_by_class)
        elif classification_type == ClassificationType.BINARY:
            return compute_scores_binary(raw_predictions=raw_predictions, values=values,
                                         per_class_values=per_class_values, labels=labels)
        elif classification_type == ClassificationType.MULTI_LABEL:
            return compute_scores_multi_label(raw_predictions=raw_predictions, values=values,
                                              per_class_values=per_class_values, labels=labels,
                                              number_of_labels=number_of_classes, metrics_by_class=metrics_by_class)
    else:
        return values, per_class_values


def compute_scores_multi_class(raw_predictions, values, per_class_values, labels, number_of_classes, metrics_by_class):
    def apply_metric(preds, lbls, metric: Metric):
        metric.update(preds, lbls)
        return metric.compute().detach().cpu().numpy()

    f_apply_metric = partial(
        apply_metric,
        lbls=labels
    )
    activated_preds = torch.sigmoid(raw_predictions)
    _, label_predictions = torch.max(raw_predictions, 1)
    # Accuracy overall
    values['accuracy'] = f_apply_metric(
        metric=MulticlassAccuracy(num_classes=number_of_classes, average='macro'),
        preds=label_predictions
    )
    # Precision
    values['precision'] = f_apply_metric(
        metric=MulticlassPrecision(num_classes=number_of_classes, average='macro'),
        preds=label_predictions
    )
    # Recall
    values['recall'] = f_apply_metric(
        metric=MulticlassRecall(num_classes=number_of_classes, average='macro'),
        preds=label_predictions
    )
    # F1 - score
    values['f1_score'] = f_apply_metric(
        metric=MulticlassF1Score(num_classes=number_of_classes, average='macro'),
        preds=label_predictions
    )
    # AUROC
    values['AUROC'] = f_apply_metric(
        metric=MulticlassAUROC(num_classes=number_of_classes, average='macro'),
        preds=activated_preds
    )
    # AUPRC
    values['AUPRC'] = f_apply_metric(
        metric=MulticlassAUPRC(num_classes=number_of_classes, average='macro'),
        preds=activated_preds
    )

    if metrics_by_class:
        # Accuracy by class
        per_class_values['accuracy'] = [
            '{:.4f}'.format(elem) for elem in
            f_apply_metric(
                metric=MulticlassAccuracy(num_classes=number_of_classes, average=None),
                preds=label_predictions)
        ]

        # Precision by class
        per_class_values['precision'] = [
            '{:.4f}'.format(elem) for elem in
            f_apply_metric(
                metric=MulticlassPrecision(num_classes=number_of_classes, average=None),
                preds=label_predictions)
        ]
        # Recall by class
        per_class_values['recall'] = [
            '{:.4f}'.format(elem) for elem in
            f_apply_metric(
                metric=MulticlassRecall(num_classes=number_of_classes, average=None),
                preds=label_predictions)
        ]
        # F1-score by class
        per_class_values['f1_score'] = [
            '{:.4f}'.format(elem) for elem in
            f_apply_metric(
                metric=MulticlassF1Score(num_classes=number_of_classes, average=None),
                preds=label_predictions)
        ]
        # AUROC by class
        per_class_values['AUROC'] = [
            '{:.4f}'.format(elem) for elem in
            f_apply_metric(metric=MulticlassAUROC(num_classes=number_of_classes, average=None),
                           preds=activated_preds)
        ]
        # AUPRC by class
        per_class_values['AUPRC'] = [
            '{:.4f}'.format(elem) for elem in
            f_apply_metric(metric=MulticlassAUPRC(num_classes=number_of_classes, average=None),
                           preds=activated_preds)
        ]
    return values, per_class_values


def compute_scores_binary(raw_predictions, values, per_class_values, labels):
    def apply_metric(preds, lbls, metric: Metric):
        metric.update(preds, lbls)
        return metric.compute().detach().cpu().numpy()

    f_apply_metric = partial(
        apply_metric,
        lbls=labels
    )

    activated_preds = torch.sigmoid(raw_predictions)
    label_predictions = torch.round(activated_preds)
    # Accuracy overall
    values['accuracy'] = f_apply_metric(
        metric=BinaryAccuracy(threshold=0.5),
        preds=label_predictions
    )
    # Precision
    values['precision'] = f_apply_metric(
        metric=BinaryPrecision(threshold=0.5),
        preds=label_predictions
    )
    # Recall
    values['recall'] = f_apply_metric(
        metric=BinaryRecall(threshold=0.5),
        preds=label_predictions
    )
    # F1 - score
    values['f1_score'] = f_apply_metric(
        metric=BinaryF1Score(threshold=0.5),
        preds=label_predictions
    )
    # AUROC
    values['AUROC'] = f_apply_metric(
        metric=BinaryAUROC(),
        preds=activated_preds
    )
    # AUPRC
    values['AUPRC'] = f_apply_metric(
        metric=BinaryAUPRC(),
        preds=activated_preds
    )

    return values, per_class_values


def compute_scores_multi_label(raw_predictions, values, per_class_values, labels, number_of_labels, metrics_by_class):
    def apply_metric(preds, lbls, metric: MultilabelStatScores | MultilabelAUROC):
        return metric(preds, lbls).detach().cpu().numpy()

    # Accuracy overall
    values['accuracy'] = apply_metric(
        preds=raw_predictions,
        lbls=labels,
        metric=MultilabelAccuracy(num_labels=number_of_labels, average='macro', threshold=0.5)
    )
    # Precision
    values['precision'] = apply_metric(
        preds=raw_predictions,
        lbls=labels,
        metric=MultilabelPrecision(num_labels=number_of_labels, average='macro', threshold=0.5)
    )
    # Recall
    values['recall'] = apply_metric(
        preds=raw_predictions,
        lbls=labels,
        metric=MultilabelRecall(num_labels=number_of_labels, average='macro', threshold=0.5)
    )
    # F1 - score
    values['f1_score'] = apply_metric(
        preds=raw_predictions,
        lbls=labels,
        metric=MultilabelF1Score(num_labels=number_of_labels, average='macro', threshold=0.5)
    )
    # AUROC
    values['AUROC'] = apply_metric(
        preds=raw_predictions,
        lbls=labels,
        metric=MultilabelAUROC(num_labels=number_of_labels, average="macro", thresholds=None)
    )
    # AUPRC
    metric = MultilabelAUPRC(num_labels=number_of_labels, average='macro')
    metric.update(raw_predictions, labels)
    values['AUPRC'] = metric.compute().detach().cpu().numpy()

    if metrics_by_class:
        # Accuracy by class
        per_class_values['accuracy'] = [
            '{:.4f}'.format(elem) for elem in
            apply_metric(
                preds=raw_predictions,
                lbls=labels,
                metric=MultilabelAccuracy(num_labels=number_of_labels, average=None, threshold=0.5)
            )
        ]

        # Precision by class
        per_class_values['precision'] = [
            '{:.4f}'.format(elem) for elem in
            apply_metric(
                preds=raw_predictions,
                lbls=labels,
                metric=MultilabelPrecision(num_labels=number_of_labels, average=None, threshold=0.5)
            )
        ]
        # Recall by class
        per_class_values['recall'] = [
            '{:.4f}'.format(elem) for elem in
            apply_metric(
                preds=raw_predictions,
                lbls=labels,
                metric=MultilabelRecall(num_labels=number_of_labels, average=None, threshold=0.5)
            )
        ]
        # F1-score by class
        per_class_values['f1_score'] = [
            '{:.4f}'.format(elem) for elem in
            apply_metric(
                preds=raw_predictions,
                lbls=labels,
                metric=MultilabelF1Score(num_labels=number_of_labels, average=None, threshold=0.5)
            )
        ]
        # AUROC by class
        per_class_values['AUROC'] = [
            '{:.4f}'.format(elem) for elem in
            apply_metric(
                preds=raw_predictions,
                lbls=labels,
                metric=MultilabelAUROC(num_labels=number_of_labels, average=None, thresholds=None)
            )
        ]
        # AUPRC by class
        metric = MultilabelAUPRC(num_labels=number_of_labels, average=None)
        metric.update(raw_predictions, labels)
        per_class_values['AUPRC'] = ['{:.4f}'.format(elem) for elem in metric.compute()]

    return values, per_class_values


def load_recorders(save_path):
    """Reload recorder lists in case we crash, or split training."""
    recorder = {}
    df = pd.read_csv(save_path)

    # Read dataframe back into lists in case we crash during training
    for name, values in df.items():
        recorder[name] = values.to_list()
    return recorder


def save_recorders(save_path, recorder):
    """
    Create dataframe and write to csv.

    Although counter-intuitive it is cheaper to maintain a dictionary of lists and create a new dataframe each time
    than it is to grow a dataframe row-wise.
    """
    df = pd.DataFrame(data=recorder)
    df.to_csv(save_path, float_format='%.6f', index=False)


def save_per_class_recorders(save_path, class_sizes, recorder):
    """
    Create dataframe and write to csv.

    This gives the per-class results.

    Although counter-intuitive it is cheaper to maintain a dictionary of lists and create a new dataframe each time
    than it is to grow a dataframe row-wise.
    """
    df = pd.DataFrame(data=recorder)
    df = pd.concat([class_sizes, df], axis=1)
    df.to_csv(save_path, float_format='%.6f', index=False)


def join_path_file(file, path, ext=''):
    "Return `path/file` if file is a string or a `Path`, file otherwise"
    if not isinstance(file, (str, Path)): return file
    if not isinstance(path, Path): path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path / f'{file}{ext}'

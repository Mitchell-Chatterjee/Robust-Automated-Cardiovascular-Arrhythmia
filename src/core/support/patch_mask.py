import itertools

import torch

import numpy as np

from src.core.support.abstract_support_class import AbstractSupportClass
from src.core.utils.basics import _torch_single


class PatchMaskSupport(AbstractSupportClass):
    def __init__(self, patch_len, stride, mask_ratio, per_lead):
        """
        Support Class used to perform the pretext task of reconstruct the original data after a binary mask
        has been applied.
        Args:
            patch_len:        patch length
            stride:           stride
            mask_ratio:       mask ratio
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self._per_lead = per_lead
        self._mask = None
        self._inverted_mask = None
        self._binary_mask = None

    @property
    def mask(self):
        return self._mask

    @property
    def inverted_mask(self):
        return self._inverted_mask

    @property
    def binary_mask(self):
        return self._binary_mask

    @property
    def loss(self):
        return self._loss

    def before_forward(self, input, target, device):
        """
        Patch masking
        xb: [bs x n_vars x seq_len] -> [bs x n_vars x num_patch x patch_len]
        """
        # Apply regular zero masking
        xb_patch = create_patch(input, self.patch_len, self.stride)
        target = create_patch(target, self.patch_len, self.stride)

        # Move mask to cuda
        self._binary_mask = _torch_single(get_mask(xb_patch, self.mask_ratio, self._per_lead)).to(device)
        self._mask = self._binary_mask[:, :, :, :1].squeeze() != 0
        self._inverted_mask = self._binary_mask[:, :, :, :1].squeeze() == 0

        return xb_patch, target, self._mask, self._inverted_mask

    def after_batch_valid(self, yb, pred):
        return un_patch(batch=yb), un_patch(batch=pred)

    def _loss(self, preds, target):
        """
        preds:   [bs x n_vars x num_patch x patch_len]
        targets: [bs x n_vars x num_patch x patch_len]
        """
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self._inverted_mask).sum() / self._inverted_mask.sum()
        return loss

    def get_chunk_patch_indices(self, xb):
        """
        Gets a continuous chunk of data. Corresponding to one quarter the length of the entire signal.
        """
        seq_len = xb.shape[xb.ndim - 1]
        num_patch = (max(seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        subset_size = round(num_patch // 4)

        start_idx = np.random.randint(0, num_patch - subset_size + 1)
        return list(range(start_idx, start_idx + subset_size))

    def get_random_patch_indices(self, xb):
        """
        Get random subset of patches. Between 0.25 and 0.75 of the original signal.
        """
        seq_len = xb.shape[xb.ndim - 1]
        num_patch = ((max(seq_len, self.patch_len) - self.patch_len) // self.stride + 1) * xb.shape[1]

        # Get random value between 0.25 and 0.75 of signal
        len_keep = np.random.randint(round(num_patch // 4), round((num_patch // 4) * 3))

        return np.random.choice(num_patch, size=len_keep, replace=False)

    def get_test_time_augmentation_indices(self, xb):
        """
        Gets continuous chunks of data. Provides overlapping indices for test time augmentation.
        """
        seq_len = xb.shape[xb.ndim - 1]
        num_patch = (max(seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        subset_size = round(num_patch // 4)
        step = round(subset_size // 2)

        indices = [list(range(i, i + subset_size)) for i in range(0, num_patch - subset_size, step)] + \
                  [list(range(num_patch - subset_size, num_patch))]
        indices.sort()
        return list(k for k, _ in itertools.groupby(indices))


class FinetunePatchSupport(PatchMaskSupport):

    def __init__(self, patch_len, stride, per_lead):
        """
        Support Class used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        super().__init__(patch_len=patch_len, stride=stride, mask_ratio=0, per_lead=per_lead)

    def before_forward(self, input, target, device):
        """
        Set patch
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch = create_patch(input, self.patch_len, self.stride)
        return xb_patch, target, None, None


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x n_vars x seq_len]
    """
    seq_len = xb.shape[xb.ndim-1]

    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len

    xb = xb[..., s_begin:]  # xb: [bs x nvars x tgt_len]

    # For torch tensors
    xb = xb.unfold(dimension=xb.ndim-1, size=patch_len, step=stride).squeeze()  # xb: [bs x n_vars x num_patch x patch_len]

    # For numpy arrays
    # xb = np.stack(np.split(xb, seq_len // patch_len, axis=2), axis=2)

    return xb


def un_patch(batch):
    """
    batch:     [bs x n_vars x num_patch x patch_len]
    """
    bs, n_vars, num_patch, patch_len = batch.shape
    if torch.is_tensor(batch):
        return torch.reshape(batch, shape=(bs, n_vars, num_patch * patch_len))
    return np.reshape(batch, newshape=(bs, n_vars, num_patch * patch_len))


def get_mask(xb, mask_ratio, per_lead):
    # xb: [bs x n_vars x num_patch x patch_len]
    bs, n_vars, num_patch, patch_len = xb.shape

    len_keep = int(num_patch * (1 - mask_ratio))

    if per_lead:
        # Get a random set of indices
        third_1_mask = np.zeros(shape=(bs // 2, n_vars, num_patch, patch_len))
        third_2_mask = np.zeros(shape=(bs // 2 + bs % 2, n_vars, num_patch, patch_len))

        # Standard masking per-lead (first third)
        indices_to_keep = np.random.choice(num_patch, size=len_keep, replace=False)
        third_1_mask[:, :, indices_to_keep, :] += 1

        # Different masking per-lead (second third)
        for lead in range(n_vars):
            indices_to_keep = np.random.choice(num_patch, size=len_keep, replace=False)
            third_2_mask[:, lead, indices_to_keep, :] += 1

        # Stack separate masks together
        binary_mask = np.concatenate((third_1_mask, third_2_mask), axis=0)

        return binary_mask
    else:
        # Get a random set of indices
        third_1_mask = np.zeros(shape=(bs // 3, n_vars, num_patch, patch_len))
        third_2_mask = np.zeros(shape=(bs // 3 + bs % 3, n_vars, num_patch, patch_len))
        third_3_mask = np.zeros(shape=(bs // 3, n_vars * num_patch, patch_len))

        # Standard masking per-lead (first third)
        indices_to_keep = np.random.choice(num_patch, size=len_keep, replace=False)
        third_1_mask[:, :, indices_to_keep, :] += 1

        # Different masking per-lead (second third)
        for lead in range(n_vars):
            indices_to_keep = np.random.choice(num_patch, size=len_keep, replace=False)
            third_2_mask[:, lead, indices_to_keep, :] += 1

        # Masking across all leads (last third)
        indices_to_keep = np.random.choice(num_patch * n_vars, size=len_keep * n_vars, replace=False)
        third_3_mask[:, indices_to_keep, :] += 1
        third_3_mask = third_3_mask.reshape((-1, n_vars, num_patch, patch_len))

        # Stack separate masks together
        binary_mask = np.concatenate((third_1_mask, third_2_mask, third_3_mask), axis=0)

        return binary_mask

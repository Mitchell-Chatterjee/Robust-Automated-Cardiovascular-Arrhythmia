import os.path

import torch
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from src.core.utils.learner_utils import get_model_from_module


def get_device(use_cuda=True, device_id=None, usage=5):
    "Return or set default device; `use_cuda`: None - CUDA if available; True - error if not available; False - CPU"
    if not torch.cuda.is_available():
        use_cuda = False
    else:
        if device_id is None:
            device_ids = get_available_cuda(usage=usage)
            device_id = device_ids[0]  # get the first available device
        torch.cuda.set_device(device_id)
    return torch.device(torch.cuda.current_device()) if use_cuda else torch.device('cpu')


def set_device(usage=5):
    "set the device that has usage < default usage  "
    device_ids = get_available_cuda(usage=usage)
    torch.cuda.set_device(device_ids[0])  # get the first available device


def default_device(use_cuda=True):
    "Return or set default device; `use_cuda`: None - CUDA if available; True - error if not available; False - CPU"
    if not torch.cuda.is_available():
        use_cuda = False
    return torch.device(torch.cuda.current_device()) if use_cuda else torch.device('cpu')


def get_available_cuda(usage=10):
    if not torch.cuda.is_available(): return
    # collect available cuda devices, only collect devices that has less that 'usage' percent 
    device_ids = []
    for device in range(torch.cuda.device_count()):
        if torch.cuda.utilization(device) < usage: device_ids.append(device)
    return device_ids


def to_device(b, device=None, non_blocking=False):
    """
    Recursively put `b` on `device`
    components of b are torch tensors
    """
    if device is None:
        device = default_device(use_cuda=True)

    if isinstance(b, dict):
        return {key: to_device(val, device) for key, val in b.items()}

    if isinstance(b, (list, tuple)):
        return type(b)(to_device(o, device) for o in b)

    return b.to(device, non_blocking=non_blocking)


def to_numpy(b):
    """
    Components of b are torch tensors
    """
    if isinstance(b, dict):
        return {key: to_numpy(val) for key, val in b.items()}

    if isinstance(b, (list, tuple)):
        return type(b)(to_numpy(o) for o in b)

    return b.detach().cpu().numpy()


def convert_signal_to_image(record, sig_filter, title):
    t = np.arange(0, 1, 1 / record.shape[1])
    # Generate the plot for this image
    fig, axs = plt.subplots(nrows=record.shape[0], ncols=1, figsize=(10, 10))
    for j, lead in enumerate(record):
        # Filter and plot the signal
        if sig_filter is None:
            axs[j].plot(t, lead, color='C0')
        else:
            axs[j].plot(t, sig_filter(lead), color='C0')
        axs[j].axis('off')
    plt.title(title)
    plt.tight_layout()

    # Save the figure
    plot_path = 'results/plots'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fig.savefig(f'{plot_path}/{title}.png')
    plt.close()


def convert_signal_to_image_reconstruction(record, sig_filter, save_path, title, mask):
    t = np.arange(0, 1, 1 / record.shape[1])

    # Generate the plot for this image
    fig, axs = plt.subplots(nrows=record.shape[0], ncols=1, figsize=(10, 10))
    for j, lead in enumerate(record):
        lead_mask = mask[j]
        # Filter the signal if filter provided
        lead = sig_filter(lead) if sig_filter is not None else lead
        # Ensure these are numpy arrays
        lead, lead_mask = lead.numpy(), lead_mask

        # Create the points and segments
        points = np.array([t, lead]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Decide on the order of colours based on whether t/f is in first position
        cmap = ListedColormap(['r', 'C0'])
        # Boundary norm decides which colours are applied to which values based on the mask
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        # Colour the line segments using the mask
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lead_mask = lead_mask[:-1]
        lc.set_array(lead_mask)
        lc.set_linewidth(2)

        # Plot the collection of segments
        axs[j].add_collection(lc)
        axs[j].set_ylim(lead.min(), lead.max())
        axs[j].axis('off')

    red_patch = mpatches.Patch(color='red', label='Masked data')
    blue_patch = mpatches.Patch(color='C0', label='Unmasked data')
    plt.legend(handles=[red_patch, blue_patch], loc='upper right')

    plt.title(title)
    plt.tight_layout()

    # Save the figure
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(f'{save_path}/{title}.png')
    plt.close()


def snapshot_exists(path):
    return path is not None and os.path.exists(path)


def load_snapshot(model, path):
    print(f'Loading pre-trained model: {path}')
    snapshot = torch.load(path)
    get_model_from_module(model).load_state_dict(snapshot['model'])

    return model


def _torch_single(x):
    return torch.from_numpy(x) if not torch.is_tensor(x) else x


def _torch(dfs):
    return tuple(torch.from_numpy(x) if not torch.is_tensor(x) else x for x in dfs)

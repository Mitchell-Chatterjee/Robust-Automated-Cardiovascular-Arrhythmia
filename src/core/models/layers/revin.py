import torch
from torch import nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, norm_stats='BN'):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        assert norm_stats in ['BN', 'ptb-xl'], "RevIn must use batch normalization or ptb-xl stats"
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.norm_stats = norm_stats

        # PTB-XL stats
        self.mean = torch.tensor(
            [-0.00184586, -0.00130277, 0.00017031, -0.00091313, -0.00148835, -0.00174687, -0.00077071, -0.00207407,
             0.00054329, 0.00155546, -0.00114379, -0.00035649])
        self.stdev = torch.tensor(
            [0.16401004, 0.1647168, 0.23374124, 0.33767231, 0.33362807, 0.30583013, 0.2731171, 0.27554379, 0.17128962,
             0.14030828, 0.14606956, 0.14656108])

        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            x = x.transpose(-1, -2)
            self._get_statistics(x)
            x = self._normalize(x).transpose(-1, -2)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        if self.norm_stats == 'BN':
            dim2reduce = tuple(range(0, x.ndim-1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=False).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=False, unbiased=False) + self.eps).detach()
        else:
            self.mean = self.mean.to(x.device)
            self.stdev = self.stdev.to(x.device)

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

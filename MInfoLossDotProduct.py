import numpy as np
import torch
import torch.nn as nn
import logging
import math
from torch.nn import init
import functools

from dim_loss import fenchel_dual_loss, nce_loss, donsker_varadhan_loss, get_positive_expectation, \
    get_negative_expectation

####################
# initialize
####################
logger = logging.getLogger('base')

class Permute(torch.nn.Module):
    """Module for permuting axes.

    """
    def __init__(self, *perm):
        """

        Args:
            *perm: Permute axes.
        """
        super().__init__()
        self.perm = perm

    def forward(self, input):
        """Permutes axes of tensor.

        Args:
            input: Input tensor.

        Returns:
            torch.Tensor: permuted tensor.

        """
        return input.permute(*self.perm)

class MIFCNet(nn.Module):
    def __init__(self, n_input, n_units, bn =False):
        super().__init__()

        self.bn = bn

        assert(n_units >= n_input)

        self.linear_shortcut = nn.Linear(n_input, n_units)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(n_input, n_units, bias=False),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units)
        )

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((n_units, n_input), dtype=np.uint8)
        for i in range(n_input):
            eye_mask[i, i] = 1

        self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)


        self.block_ln = nn.LayerNorm(n_units)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h = self.block_nonlinear(x) + self.linear_shortcut(x)

        if self.bn:
            h = self.block_ln(h)

        return h

class MI1x1ConvNet(nn.Module):
    """Simple custorm 1x1 convnet.

    """
    def __init__(self, n_input, n_units,):
        super().__init__()

        self.block_nonlinear = nn.Sequential(
            nn.Conv2d(n_input, n_units, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_units),
            nn.ReLU(),
            nn.Conv2d(n_units, n_units, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.block_ln = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(n_units),
            Permute(0, 3, 1, 2)
        )

        self.linear_shortcut = nn.Conv2d(n_input, n_units, kernel_size=1,
                                         stride=1, padding=0, bias=False)

        # initialize shortcut to be like identity (if possible)
        if n_units >= n_input:
            eye_mask = np.zeros((n_units, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
            self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        h = self.block_ln(self.block_nonlinear(x) + self.linear_shortcut(x))
        return h

class MutualInfoLoss(nn.Module):
    def __init__(self, opt):
        super(MutualInfoLoss, self).__init__()

        self.opt = opt

    def forward(self, L, G):
        '''
        Args:
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.

        '''
        if G is not None:
            # Add a global-local loss.
            local_global_loss = self.local_global_loss(L, G, self.opt.measure, self.opt.MIMode)
        else:
            local_global_loss = 0.

        return local_global_loss

    def local_global_loss(self, l, g, measure, mode):
        '''

        Args:
            l: Local feature map.
            g: Global features.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.

        Returns:
            torch.Tensor: Loss.

        '''
        N, local_units, dim_x, dim_y = l.size()
        l = l.view(N, local_units, -1)
        # g_enc = g_enc.view(N, local_units, -1)

        if mode == 'fd':
            loss = fenchel_dual_loss(l, g, measure=measure)
        elif mode == 'nce':
            loss = nce_loss(l, g)
        elif mode == 'dv':
            loss = donsker_varadhan_loss(l, g)
        else:
            raise NotImplementedError(mode)

        return loss

    def fenchel_dual_loss(self, l, g, measure=None):
        '''Computes the f-divergence distance between positive and negative joint distributions.

        Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
        Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.

        Args:
            l: Local feature map.
            g: Global features.
            measure: f-divergence measure.

        Returns:
            torch.Tensor: Loss.

        '''
        N, local_units, n_locs = l.size()
        l = l.permute(0, 2, 1)
        l = l.reshape(-1, local_units)

        u = torch.mm(g, l.t())
        u = u.reshape(N, N, -1)
        mask = torch.eye(N).cuda()
        n_mask = 1 - mask

        E_pos = get_positive_expectation(u, measure, average=False).mean(2)
        E_neg = get_negative_expectation(u, measure, average=False).mean(2)
        E_pos = (E_pos * mask).sum() / mask.sum()
        E_neg = (E_neg * n_mask).sum() / n_mask.sum()
        loss = E_neg - E_pos
        return loss

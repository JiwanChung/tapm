# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

# from fairseq
import math

from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class Loss(nn.CrossEntropyLoss):
    def __init__(self, padding_idx=0):
        super(Loss, self).__init__(ignore_index=padding_idx)

    def forward(self, hypo, tgt):
        hypo = hypo.contiguous()
        tgt = tgt.contiguous()
        if hypo.nelement() == 0 or tgt.nelement() == 0:  # failsafe for empty tensor
            loss = None
        else:
            loss = super().forward(hypo.view(-1, hypo.shape[-1]),
                                tgt.view(-1))
        return loss, {}


# label smoothed cross entropy
class SmoothLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', eps=0, padding_idx=0):
        super(SmoothLoss, self).__init__(size_average, reduce, reduction)

        self.eps = eps
        self.padding_idx = padding_idx

    def forward(self, hypo, tgt):
        hypo = hypo.contiguous()
        tgt = tgt.contiguous()
        lprobs = F.log_softmax(hypo, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = tgt.view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

        ppl = (2 ** nll_loss).mean().item()
        nll_loss_report = nll_loss.mean().item()

        nll_loss = self._reduce(nll_loss)
        smooth_loss = self._reduce(smooth_loss)
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, {'nll_loss': nll_loss_report, 'ppl': ppl}

    def _reduce(self, t):
        func = {
            'none': lambda x: x,
            'mean': lambda x: x.mean(),
            'sum': lambda x: x.sum()
        }[self.reduction]

        return func(t)


class FocalLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', gamma=2):
        super(FocalLoss, self).__init__(size_average, reduce, reduction)

        self.gamma = gamma

    def forward(self, hypo, tgt):
        hypo = hypo.contiguous()
        tgt = tgt.contiguous().byte().float()

        # hypo = (tgt=1: hypo, tgt=0: (1-hypo))
        p_t = hypo * (2 * tgt - 1) + (1 - tgt)
        loss = -((1 - p_t) ** self.gamma) * p_t.log()
        loss = loss.mean(dim=-1)
        loss = self._reduce(loss)
        return loss, {}

    def _reduce(self, t):
        func = {
            'none': lambda x: x,
            'mean': lambda x: x.mean(),
            'sum': lambda x: x.sum()
        }[self.reduction]

        return func(t)

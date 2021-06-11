from functools import partial

import torch
# import torch.nn.functional as F
from entmax import sparsemax_loss, entmax15_loss, entmax_bisect_loss

from sampler import get_normalizer
from .base import SmoothLoss


class SparseLoss(SmoothLoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean',
                 eps=0, padding_idx=0, sparsity=2):
        super(SparseLoss, self).__init__(size_average, reduce, reduction,
                                         eps=eps, padding_idx=padding_idx)

        self.normalizer, _ = get_normalizer(sparsity)
        sparsity = str(sparsity)
        eps = 1e-4
        lossers = {
            '1': partial(entmax_bisect_loss, alpha=1 + eps),
            '2': sparsemax_loss,
            '1.5': entmax15_loss,
            'adaptive': entmax_bisect_loss
        }
        # assert sparsity in lossers, f"invalid sparsity value: {sparsity}"
        losser = lossers[sparsity]
        if sparsity == 'adaptive':
            f = losser
        else:
            def f(hypo, target, alpha=None):
                return losser(hypo, target)
        self.losser = f

    def get_log_probs(self, hypo):
        if getattr(hypo, 'if_log_softmax', False):
            probs = hypo
        else:
            probs = self.prober(hypo, dim=-1)
        probs[probs == 0] = 1e-20
        lprobs = probs.log()
        return lprobs

    def get_nll(self, hypo, target, non_pad_mask, alpha=None):
        hypo = hypo.detach()
        if torch.is_tensor(alpha):
            alpha = alpha.detach()
        with torch.no_grad():
            probs = self.normalizer(hypo, dim=-1, alpha=alpha)
            # import ipdb; ipdb.set_trace()
            probs = probs.gather(dim=-1, index=target.unsqueeze(-1))[non_pad_mask]
            lprobs = -(probs + self.eps).log()
            nll_loss = lprobs.mean().item()
        return nll_loss

    def forward(self, hypo, tgt, Model):
        alpha = Model.normalizer_alpha

        hypo = hypo.contiguous()
        tgt = tgt.contiguous()
        hypo = hypo.view(-1, hypo.size(-1))  # (BL)V
        target = tgt.view(-1)  # (BL)

        non_pad_mask = target.ne(self.padding_idx)
        sparse_loss = self.losser(hypo, target, alpha=alpha)  # (BL)
        sparse_loss = sparse_loss[non_pad_mask]

        nll_loss = self.get_nll(hypo, target, non_pad_mask, alpha)
        ppl = 2 ** nll_loss
        sparse_loss_report = sparse_loss.mean().item()

        loss = self._reduce(sparse_loss)
        return loss, {'nll_loss': nll_loss,
                      'sparse_loss': sparse_loss_report, 'ppl': ppl}

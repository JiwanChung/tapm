import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from data.batcher import decode_tensor
from utils import remove_sep, mean
from metric.metric import Metric


class Loss(nn.CrossEntropyLoss):
    def __init__(self, padding_idx=0):
        super(Loss, self).__init__(ignore_index=padding_idx, reduction='mean')

    def forward(self, hypo, tgt, Model=None):
        hypo = hypo.contiguous()
        tgt = tgt.contiguous()
        if hypo.nelement() == 0 or tgt.nelement() == 0:  # failsafe for empty tensor
            loss = None
        else:
            loss = super().forward(hypo.view(-1, hypo.shape[-1]),
                                tgt.view(-1))
        return loss, {}


class RLLoss(_Loss):
    def __init__(self, tokenizer, metrics=['meteor'], use_vist=False,
                 reinforce_group=False):
        super(RLLoss, self).__init__(reduction='mean')

        self.group = reinforce_group
        self.use_vist = use_vist
        self.tokenizer = tokenizer
        self.tokenizer.whitespace = getattr(self.tokenizer, 'whitespace', b'\xc4\xa0'.decode())
        self.metric = Metric(metrics, use_vist=self.use_vist)

    def decode(self, tensor):
        tensor = decode_tensor(self.tokenizer, tensor, use_vist=self.use_vist,
                                remove_past_sep=True)
        tensor = remove_sep(tensor, self.tokenizer.sep_token)
        return tensor

    def calc_score(self, hypo, tgt):
        hypo = self.decode(hypo)
        tgt = self.decode(tgt)
        score_texts = {'temp': (hypo, tgt)}
        score_stats = self.metric.calculate(score_texts)
        return mean(list(score_stats.values()))  # TODO: account for bleu

    def calc_score_group(self, hypos, tgts):
        hypos = [self.decode(hypo) for hypo in hypos]
        tgts = [self.decode(tgt) for tgt in tgts]
        hypo = ' '.join(hypos)
        tgt = ' '.join(tgts)
        score_texts = {'temp': (hypo, tgt)}
        score_stats = self.metric.calculate(score_texts)
        return mean(list(score_stats.values()))  # TODO: account for bleu

    def v_loss(self, reward, baseline, mask):
        o = (reward - baseline) ** 2 * mask
        o = o.sum() / mask.sum()

        return o

    def a_loss(self, log_prob, reward, baseline, mask):
        advantage = reward - baseline.detach()
        o = - log_prob * advantage * mask
        o = o.sum() / mask.sum()

        return o

    def forward(self, hypos, tgts, logits, baseline):
        # BGL
        G = hypos.shape[1]
        mask = hypos != self.tokenizer.pad_id
        mask = mask.float()
        log_prob = F.log_softmax(logits, dim=-1)
        log_prob = log_prob.gather(dim=-1, index=hypos.unsqueeze(-1)).squeeze(-1)
        if self.group:
            reward = torch.Tensor(
                [self.calc_score_group(group_hypo, group_tgt)
                 for group_hypo, group_tgt in zip(hypos, tgts)]
            ).float().to(log_prob.device).unsqueeze(1).repeat(1, G)
        else:
            reward = torch.Tensor(
                [[self.calc_score(hypo, tgt) for hypo, tgt in zip(group_hypo, group_tgt)]
                  for group_hypo, group_tgt in zip(hypos, tgts)]
            ).float().to(log_prob.device)

        reward = reward.float().to(log_prob.device).unsqueeze(-1)

        v_loss = self.v_loss(reward, baseline, mask)
        a_loss = self.a_loss(log_prob, reward, baseline, mask)

        loss = v_loss + a_loss

        stats = {
            'reward': reward.mean().item(),
            'a_loss': a_loss.item(),
            'v_loss': v_loss.item(),
            'rl_loss': loss.item()
        }

        return loss, stats


# from fairseq (github.com/facebook/fairseq)
# with all rights reserved
# label smoothed cross entropy
class SmoothLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean',
                 eps=0, padding_idx=0):
        super(SmoothLoss, self).__init__(size_average, reduce, reduction)

        self.eps = eps
        self.padding_idx = padding_idx

    def get_log_probs(self, hypo):
        if getattr(hypo, 'if_log_softmax', False):
            lprobs = hypo
        else:
            lprobs = F.log_softmax(hypo, dim=-1)
        return lprobs

    def forward(self, hypo, tgt, model=None):
        hypo = hypo.contiguous()
        tgt = tgt.contiguous()
        lprobs = self.get_log_probs(hypo)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = tgt.view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

        ppl = 2 ** nll_loss.mean().item()
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

    def forward(self, hypo_logit, tgt):
        hypo = F.sigmoid(hypo_logit)
        hypo = hypo.contiguous()
        tgt = tgt.contiguous().bool().float()

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


class BinaryCELoss(nn.BCEWithLogitsLoss):
    def __init__(self, reduction='mean'):
        super(BinaryCELoss, self).__init__(reduction=reduction)

    def forward(self, hypo, tgt):
        hypo = hypo.contiguous()
        tgt = tgt.contiguous().bool().float()

        loss = super().forward(hypo, tgt)

        return loss, {}


class BinaryCERecallLoss(_Loss):
    def __init__(self, recall_weight=1, reduction='mean'):
        super(BinaryCERecallLoss, self).__init__(reduction=reduction)

        self.reduction = reduction
        self.weight = recall_weight

    def forward(self, hypo, tgt):
        # BK
        hypo = hypo.contiguous()
        tgt = tgt.contiguous().bool().float()
        recall_weight = torch.FloatTensor([self.weight]).to(hypo.device).expand(hypo.shape[-1])

        loss = F.binary_cross_entropy_with_logits(hypo, tgt,
                                                  pos_weight=recall_weight,
                                                  reduction=self.reduction)

        return loss, {}


class SortedLoss(_Loss):
    def __init__(self, top_k=20, reduction='mean', normalized=True):
        super(SortedLoss, self).__init__(reduction=reduction)

        self.reduction = reduction
        self.top_k = top_k

        self.normalized = normalized

    def get_loss(self, hs, ts, max_val=None):
        losses = []
        for i, (h, t) in enumerate(zip(hs, ts)):
            h = h.masked_select(t.bool())  # get keywords outside topk

            if h.nelement() > 0:
                if self.normalized:
                    loss = (1 - torch.sigmoid(h)).mean()
                else:
                    loss = (max_val[i].item() - h).mean()
                losses.append(loss)
        return sum(losses) / len(losses)

    def forward(self, hypo, tgt, Model=None):
        # BK
        hypo = hypo.contiguous().view(-1, hypo.shape[-1]).contiguous()
        tgt = tgt.contiguous().view(-1, tgt.shape[-1]).contiguous().bool().float()

        h_sorted, h_ids = hypo.sort(dim=-1, descending=True)
        t_sorted = tgt.gather(dim=-1, index=h_ids)

        reg_loss = (h_sorted.sum(dim=-1) - self.top_k).abs().mean()
        max_val = h_sorted[:, 0]
        h_sorted = h_sorted[:, self.top_k:]
        t_sorted = t_sorted[:, self.top_k:]

        sort_loss = self.get_loss(h_sorted, t_sorted, max_val=max_val)
        loss = reg_loss + sort_loss

        return loss, {'sort_loss': sort_loss.item(), 'reg_loss': reg_loss.item()}

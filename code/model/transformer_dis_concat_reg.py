from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from utils import mean
from .transformer_dis_concat import TransformerDisConcat
from .collaborative_experts import calc_ranking_loss


class TransformerDisConcatReg(TransformerDisConcat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.margin = 1

        self.roll_left_linear = nn.Linear(self.gpt_dim, self.gpt_dim)
        self.roll_right_linear = nn.Linear(self.gpt_dim, self.gpt_dim)

        self.l2_loss = nn.MSELoss(reduction='none')

    def l2(self, x, y, group_mask):
        loss = self.l2_loss(x, y)  # BG
        group_mask = group_mask.unsqueeze(-1).expand_as(loss)
        loss = loss.masked_select(group_mask)
        return loss.mean()

    def get_reg_loss(self, h_mean, c, group_mask):
        # BGLC, BGRC
        c = OrderedDict(c)
        c = [v.mean(dim=-2) for v in c.values()]
        c = mean(c)
        h = h_mean.detach()
        # h = h.mean(dim=-2)

        roll_loss = self.get_roll_loss(h, c, group_mask)
        rank_loss, stats = self.get_rank_loss(h, c, group_mask)

        loss = roll_loss + rank_loss

        stats = {**stats,
                 'roll_loss': roll_loss.item(),
                 'rank_loss': rank_loss.item()}

        return loss, stats

    def get_roll_loss(self, h, c, group_mask):
        h_left = torch.roll(h, 1, 1)
        h_right = torch.roll(h, -1, 1)

        left_loss = self.l2(self.roll_left_linear(c), h_left, group_mask)
        right_loss = self.l2(self.roll_right_linear(c), h_right, group_mask)

        return (left_loss + right_loss)

    def get_rank_loss(self, h, c, group_mask):
        x1 = F.normalize(h).view(-1, h.shape[-1])
        x2 = F.normalize(c).view(-1, c.shape[-1])
        group_mask = group_mask.view(-1)

        loss1, acc1 = calc_ranking_loss(x1, x2, group_mask, margin=self.margin, pool='mean')
        loss2, acc2 = calc_ranking_loss(x2, x1, group_mask, margin=self.margin, pool='mean')

        stats = {'ranking_accuracy': (acc1 + acc2) / 2}
        if loss1 is None or loss2 is None:
            loss = None
        else:
            loss = (loss1 + loss2).mean()
            if loss is not None:
                stats = {**stats, 'ranking_loss': loss.item()}
            else:
                stats = {**stats, 'ranking_loss': None}
        return loss, stats

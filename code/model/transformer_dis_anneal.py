import torch
from torch import nn
import torch.nn.functional as F

from exp import ex
from .transformer_dis_group import TransformerDisGroupReg, DeepEncoder


class TransformerDisGroupRegAnneal(TransformerDisGroupReg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reg_coeff_threshold = 0.03

    def anneal_coeff(self, acc):
        # acc:1 -> coeff: 0, acc:0 -> coeff: 1
        # linear?
        coeff = 1 - acc
        if coeff < self.reg_coeff_threshold:
            coeff = 0
        return coeff

    def get_reg_loss(self, h, c, group_mask):
        roll_loss, roll_stats = self.get_roll_loss(h, c, group_mask)
        rank_loss, rank_stats = self.get_rank_loss(h, c, group_mask)

        stats = {**roll_stats,
                 **rank_stats,
                 'roll_loss': roll_loss.item(),
                 'rank_loss': rank_loss.item()}

        roll_coeff = self.anneal_coeff(roll_stats['roll_accuracy'])
        roll_loss = roll_coeff * roll_loss
        rank_coeff = self.anneal_coeff(rank_stats['rank_accuracy'])
        rank_loss = rank_coeff * rank_loss

        stats = {**stats,
                 'roll_coeff': roll_coeff,
                 'rank_coeff': rank_coeff}

        loss = (roll_loss + rank_loss).mean()

        return loss, stats


class TransformerDisAnnealFeat(TransformerDisGroupRegAnneal):
    @classmethod
    def get_args(cls):
        return {
            **super().get_args(),
            'fix_gpt_epoch': 5
        }

    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super(TransformerDisAnnealFeat, self).__init__(transformer, tokenizer, dropout_before)

        self.fix_gpt_epoch = fix_gpt_epoch
        self.net.transformer.weight_freezed = True
        self._fix_gpt()

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[DeepEncoder(dim, self.gpt_dim)]))

    def fix_gpt(self, epoch):
        if epoch < self.fix_gpt_epoch:
            if not self.net.transformer.weight_freezed:
                self._fix_gpt()
                self.net.transformer.weight_freezed = True
        else:
            if self.net.transformer.weight_freezed:
                self._fix_gpt(False)
                self.net.transformer.weight_freezed = False
                self.reset_optimizer = True

    def _fix_gpt(self, flag=True):
        print(f"gpt param freezed: {flag}")
        for name, param in self.net.transformer.named_parameters():
            param.requires_grad_(not flag)

    def forward(self, batch, **kwargs):
        if self.training:
            self.fix_gpt(kwargs.get('epoch', 0))
        return self._forward(batch, **kwargs)

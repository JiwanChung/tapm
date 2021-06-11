from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from exp import ex
from .modules import SelfAttention
from .transformer_dis_concat_reg import TransformerDisConcatReg
from .transformer_dis_group import TransformerDisGroupReg
from .encoders import DeepEncoder
from .ss_loss import calc_loss_group


class ConcatAux(TransformerDisConcatReg):
    @classmethod
    def get_args(cls):
        return {
            **super().get_args(),
            'fix_gpt_epoch': 5
        }

    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before)

        self.fix_gpt_epoch = fix_gpt_epoch
        self.net.transformer.weight_freezed = False

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[DeepEncoder(dim, self.gpt_dim)]))

    def pretrain(self, flag=True):
        self.fix_gpt(flag)

    def fix_gpt(self, flag=True):
        if flag != self.net.transformer.weight_freezed:
            print(f"generator param freezed: {flag}")
            self._fix_gpt(flag)

    def _fix_gpt(self, flag=True):
        for name, param in self.net.transformer.named_parameters():
            param.requires_grad_(not flag)
        self.net.transformer.weight_freezed = flag

    def forward(self, batch, **kwargs):
        reinforce = kwargs.get('reinforce', False)
        hypo, logit, target, reg_loss, stats, batch = self._forward(batch, **kwargs)
        if self.training:
            if self.net.transformer.weight_freezed \
                    and not reinforce:
                logit = None
            else:
                reg_loss = None
        return hypo, logit, target, reg_loss, stats, batch


class PretrainAux(TransformerDisGroupReg):
    @classmethod
    def get_args(cls):
        return {
            **super().get_args(),
            'fix_gpt_epoch': 5
        }

    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before)

        self.fix_gpt_epoch = fix_gpt_epoch
        self.net.transformer.weight_freezed = False

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[DeepEncoder(dim, self.gpt_dim)]))

    def pretrain(self, flag=True):
        self.fix_gpt(flag)

    def fix_gpt(self, flag=True):
        if flag != self.net.transformer.weight_freezed:
            print(f"generator param freezed: {flag}")
            self._fix_gpt(flag)

    def _fix_gpt(self, flag=True):
        for name, param in self.net.transformer.named_parameters():
            param.requires_grad_(not flag)
        self.net.transformer.weight_freezed = flag

    def forward(self, batch, **kwargs):
        reinforce = kwargs.get('reinforce', False)
        hypo, logit, target, reg_loss, stats, batch = self._forward(batch, **kwargs)
        if self.training:
            if self.net.transformer.weight_freezed \
                    and not reinforce:
                logit = None
            else:
                reg_loss = None
        return hypo, logit, target, reg_loss, stats, batch


class PretrainAuxContext(PretrainAux):
    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        c = context.mean(dim=-2)

        reg_loss, stats = self.get_reg_loss(o.detach(), c, group_mask)
        return reg_loss, stats


class PretrainAuxContextFeature(PretrainAux):
    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        c = context.mean(dim=-2)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss_context, stats_context = self.get_reg_loss(o.detach(), c, group_mask)
        reg_loss_feature, stats_feature = self.get_reg_loss(o.detach(), feature, group_mask)

        reg_loss = (reg_loss_context + reg_loss_feature) / 2
        stats = {
            **{f"{k}_context": v for k, v in stats_context.items()},
            **{f"{k}_feature": v for k, v in stats_feature.items()}
        }
        return reg_loss, stats


class PretrainAuxContextAttn(PretrainAuxContext):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[AttnEncoder(dim, self.gpt_dim)]))


class PretrainAuxGroup(PretrainAuxContext):
    def run_rank_loss(self, x1, x2, group_mask, skip_idx=0):
        loss1, acc1 = calc_loss_group(x1, x2, group_mask)
        loss2, acc2 = calc_loss_group(x2, x1, group_mask)

        return loss1, acc1, loss2, acc2


class PretrainAuxGroupRoll(PretrainAuxContext):
    # roll loss fix?
    def run_rank_loss(self, x1, x2, group_mask, skip_idx=0):
        loss1, acc1 = calc_loss_group(x1, x2, group_mask,
                                              skip_idx=skip_idx)
        loss2, acc2 = calc_loss_group(x2, x1, group_mask,
                                              skip_idx=-skip_idx)

        return loss1, acc1, loss2, acc2


class AttnEncoder(DeepEncoder):
    def __init__(self, in_dim, dim):
        super().__init__(in_dim, dim)

        self.attn = SelfAttention(dim)

    def forward(self, feature, h=None):
        feature = super().forward(feature)
        shape = feature.shape
        feature = feature.view(-1, *shape[-2:]).contiguous()
        feature = self.attn(feature)
        feature = feature.view(*shape).contiguous()
        return feature

from itertools import product
from random import randint
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from .modules import SelfAttention, MLP


class CollaborativeExperts(nn.Module):
    def __init__(self, feature_names, feature_dims, dim, feature_encoder=None):
        super(CollaborativeExperts, self).__init__()

        self.dim = dim
        self.feature_dims = feature_dims
        self.feature_names = feature_names

        if feature_encoder is None:
            feature_encoder = FeatureEncoder

        for feature in self.feature_names:
            setattr(self, feature, feature_encoder(self.feature_dims[feature], self.dim))
            setattr(self, f'{feature}_2', FeatureEncoderOut(self.dim))

        self.gate = CollaborativeGating(self.dim)

    def _forward(self, features):
        for feature in self.feature_names:
            features[feature] = getattr(self, feature)(features[feature])
        features = self.gate(features)
        for feature in self.feature_names:
            features[feature] = getattr(self, f'{feature}_2')(features[feature])

        # feature = torch.cat([features[k] for k in self.feature_names], dim=-1)

        return features


class CollaborativeExpertsWrapper(CollaborativeExperts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.margin = kwargs.get('margin', 1)
        self.pool = lambda x: x.mean(dim=-2)

        self.expand_o = nn.Linear(self.dim, self.dim * len(self.feature_names))

    def get_shifted_and_mask(self, mask, shift, dim=0):
        shifted_mask = torch.roll(mask, shift, dim)
        return mask & shifted_mask

    def get_max_margin_ranking_loss(self, s_true, s_wrong, mask=None):
        loss = torch.max(torch.zeros(1).to(s_true.device), self.margin + s_wrong - s_true)
        if mask is not None:
            loss = loss.masked_select(mask)
            if loss is None:
                return loss
        return loss.mean()

    def forward(self, o, features, group_mask=None):
        feature = super()._forward(features)  # BC
        o = self.pool(o)
        o = self.expand_o(o)  # BC
        o = F.normalize(o)

        loss1, acc1 = calc_ranking_loss(o, feature, group_mask, margin=self.margin)
        loss2, acc2 = calc_ranking_loss(feature, o, group_mask, margin=self.margin)  # bidirectional

        if loss1 is None or loss2 is None:
            loss = None
        else:
            loss = (loss1 + loss2).mean()
            with torch.no_grad():
                stats = {
                    'ranking_loss': loss.item(),
                    'ranking_accuracy': (acc1 + acc2) / 2,
                }

        return loss, stats


class FeatureEncoder(nn.Module):
    def __init__(self, video_dim, dim):
        super(FeatureEncoder, self).__init__()

        self.self_attention = SelfAttention(video_dim, m_dim=dim, heads=4)

    def forward(self, feature):
        feature = self.self_attention(feature)
        return feature.mean(dim=-2)


class CollaborativeGating(nn.Module):
    def __init__(self, dim):
        super(CollaborativeGating, self).__init__()

        self.g = MLP(dim * 2, dim)
        self.h = MLP(dim, dim)

    def forward(self, features):
        features = OrderedDict(features)
        keys = [(key1, key2) for key1, key2 in product(features.keys(), features.keys()) if key1 != key2]
        res = {}
        for key1, key2 in keys:
            f1 = features[key1]
            f2 = features[key2]
            if f1.shape != f2.shape:
                shape = [max(s1, s2) for s1, s2 in zip(f1.shape, f2.shape)]
                f1 = f1.expand(*shape)
                f2 = f2.expand(*shape)
            res[(key1, key2)] = F.relu(self.g(torch.cat((f1, f2), dim=-1)))
        weights = {}
        for key in features.keys():
            pairs = [res[pair] for pair in keys if key in pair]
            weight = sum(pairs)
            weight = self.h(weight)
            weight = torch.sigmoid(weight)
            weights[key] = weight

        for key in features.keys():
            features[key] * weights[key]

        return features


class FeatureEncoderOut(nn.Module):
    def __init__(self, dim):
        super(FeatureEncoderOut, self).__init__()

        self.linear = nn.Linear(dim, dim)

    def forward(self, feature):
        return self.linear(feature)


def get_max_margin_ranking_loss(s, margin, mask=None, pool='mean'):
    # i b (i==0 -> true), i b

    def pool_loss(x):
        if pool == 'mean':
            x = x.mean()
        elif pool == 'max':
            x, _ = x.max(dim=0)
            x = x.mean()
        return x

    s_true = s[0].unsqueeze(0)
    s_false = s[1:]
    loss = torch.max(torch.zeros(1).to(s_true.device), margin + s_false - s_true)  # (i-1) b
    num_element = loss.nelement()
    if num_element == 0:
        return None
    if mask is not None:
        mask = mask[1:]
        res = []
        for i in range(mask.shape[1]):
            mask_i = mask[:, i]
            loss_i = loss[:, i].masked_select(mask_i)
            num_element_i = mask_i.float().sum()
            if num_element_i > 0:
                res.append(pool_loss(loss_i))
        if len(res) > 0:
            loss = torch.stack(res, dim=-1)
            loss = loss.mean()
        else:
            return None
    else:
        loss = pool_loss(loss)
    return loss


def get_batch_max_margin_ranking_loss(s, margin, mask=None, pool='mean'):
    def pool_loss(x):
        if pool == 'mean':
            x = x.mean()
        elif pool == 'max':
            x, _ = x.max(dim=0)
            x = x.mean()
        return x

    s_true = s[:, 0].unsqueeze(1)
    s_false = s[:, 1:]
    loss = torch.max(torch.zeros(1).to(s_true.device), margin + s_false - s_true)  # (i-1) b
    num_element = loss.nelement()
    if num_element == 0:
        return None
    if mask is not None:
        eps = 1e-09
        mask = mask[:, 1:]
        mask = mask.float()
        loss = loss * mask
        loss = loss.sum(dim=-1) / (mask.sum(dim=-1) + eps)
    loss = pool_loss(loss)
    return loss


def get_ranking_accuracy(s, mask):
    # i b
    res = []
    for i in range(s.shape[-1]):
        s_i = s[:, i]
        m_i = mask[:, i]
        s_i = s_i.masked_select(m_i)
        if s_i.sum() != 0:
            idx = s_i.argmax(dim=0)
            res.append(int((idx == 0).any()))
    return sum(res) / len(res)


def get_batch_ranking_accuracy(s, mask):
    # BGG
    mask = mask.float()
    s = s * mask
    acc = s.argmax(dim=1) == 0  # BG
    nonzero = mask.sum(dim=1) >= 1  # BG
    acc = acc.masked_select(nonzero)
    acc = acc.float().mean()
    return acc.item()


def calc_ranking_loss(x1, x2, group_mask=None, margin=1, pool='mean',
                      skip_idx=0):
    x2 = x2.unsqueeze(0).repeat(x2.shape[0], 1, 1)  # BBC
    group_mask_rolled = group_mask.unsqueeze(0).repeat(group_mask.shape[0], 1)  # BB
    group_mask = group_mask.unsqueeze(0).clone()

    # get permutations
    for i in range(x2.shape[0]):
        x2[i] = torch.roll(x2[i], i, 0)
        group_mask_rolled[i] = torch.roll(group_mask_rolled[i].byte(), i, 0).bool()

    s = torch.einsum('bc,ibc->ib', x1, x2)  # BB
    mask = group_mask_rolled & group_mask
    loss = get_max_margin_ranking_loss(s, margin, mask, pool)
    with torch.no_grad():
        acc = get_ranking_accuracy(s, mask)

    return loss, acc


def calc_ranking_loss_group(x1, x2, group_mask=None, margin=1, pool='mean',
                            skip_idx=0):
    # BGC, BG
    x2 = x2.unsqueeze(1).repeat(1, x2.shape[1], 1, 1)  # BGGC
    group_mask_rolled = group_mask.unsqueeze(1).repeat(1, group_mask.shape[1], 1)  # BGG
    group_mask = group_mask.unsqueeze(1).clone()

    # get permutations
    for i in range(x2.shape[1]):
        x2[:, i] = torch.roll(x2[:, i], i, 1)
        group_mask_rolled[:, i] = torch.roll(group_mask_rolled[:, i].byte(), i, 1).bool()

    s = torch.einsum('bgc,bigc->big', x1, x2)  # BGG
    mask = group_mask_rolled & group_mask
    if skip_idx != 0:
        if mask.shape[1] > skip_idx:
            mask[:, skip_idx] = 0
    loss = get_batch_max_margin_ranking_loss(s, margin, mask, pool)
    with torch.no_grad():
        acc = get_batch_ranking_accuracy(s, mask)

    return loss, acc

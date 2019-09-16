from itertools import product
from random import randint

import torch
from torch import nn
import torch.nn.functional as F

from .modules import SelfAttention, MLP


class CollaborativeExperts(nn.Module):
    def __init__(self, feature_names, video_dim, image_dim, flow_dim, dim):
        super(CollaborativeExperts, self).__init__()

        self.dim = dim
        self.video_dim = video_dim
        self.image_dim = image_dim
        self.flow_dim = flow_dim
        self.feature_names = feature_names

        for feature in self.feature_names:
            setattr(self, feature, FeatureEncoder(getattr(self, f"{feature}_dim"), self.dim))
            setattr(self, f'{feature}_2', FeatureEncoderOut(self.dim))

        self.gate = CollaborativeGating(self.dim)

    def _forward(self, features):
        for feature in self.feature_names:
            features[feature] = getattr(self, feature)(features[feature])
        features = self.gate(features)
        for feature in self.feature_names:
            features[feature] = getattr(self, f'{feature}_2')(features[feature])

        feature = torch.cat([features[k] for k in self.feature_names], dim=-1)

        return F.normalize(feature)


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
        return loss.mean()

    def get_batch_max_margin_ranking_loss(self, s, mask=None):
        # i b (i==0 -> true), i b
        s_true = s[0].unsqueeze(0)
        s_false = s[1:]
        loss = torch.max(torch.zeros(1).to(s_true.device), self.margin + s_false - s_true)  # (i-1) b
        num_element = loss.nelement()
        if mask is not None:
            mask = mask[1:]
            loss = loss.masked_select(mask)
            num_element = mask.float().sum()
        return loss.sum() / num_element

    def calc_loss(self, x1, x2, group_mask):
        x2 = x2.unsqueeze(0).repeat(x2.shape[0], 1, 1)  # BBC
        group_mask_rolled = group_mask.unsqueeze(0).repeat(group_mask.shape[0], 1)  # BB

        # get permutations
        for i in range(x2.shape[0]):
            x2[i] = torch.roll(x2[i], i, 0)
            group_mask_rolled[i] = torch.roll(group_mask_rolled[i], i, 0)

        s = torch.einsum('bc,ibc->ib', x1, x2)
        mask = group_mask_rolled & group_mask.unsqueeze(0)
        loss = self.get_batch_max_margin_ranking_loss(s, mask)

        return loss

    def forward(self, o, features, group_mask=None):
        feature = super()._forward(features)  # BC
        o = self.pool(o)
        o = self.expand_o(o)  # BC
        o = F.normalize(o)

        loss1 = self.calc_loss(o, feature, group_mask)
        loss2 = self.calc_loss(feature, o, group_mask)  # bidirectional

        loss = (loss1 + loss2).mean()

        with torch.no_grad():
            stats = {
                'ranking_loss': loss.item(),
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
        keys = [(key1, key2) for key1, key2 in product(features.keys(), features.keys()) if key1 != key2]
        res = {}
        for key1, key2 in keys:
                res[(key1, key2)] = F.relu(self.g(torch.cat((features[key1], features[key2]), dim=-1)))
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
        feature = self.linear(feature)
        return F.normalize(feature)

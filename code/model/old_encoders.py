import torch
from torch import nn

from .encoders import DeepEncoder


# for backward compat
class OldTemporalEncoder(DeepEncoder):
    def __init__(self, in_dim, dim, Encoder=DeepEncoder):
        super().__init__(in_dim, dim)

        self.directions = {'prev': 1, 'next': -1}
        self.empty = {'prev': 0, 'next': -1}

        self.context_linear = nn.ModuleDict()
        for direction in self.directions.keys():
            self.context_linear[direction] = nn.Linear(dim, dim)

    def forward(self, feature, h=None):
        feature = super().forward(feature)
        # BGLC
        # build inter-group correlation
        f_prev = self.build_context(feature, 'prev')
        f_next = self.build_context(feature, 'next')
        feature = torch.cat((f_prev, f_next, feature), dim=-2)

        return feature

    def build_context(self, feature, direction='prev'):
        feature = self.context_linear[direction](feature)
        feature = feature.mean(dim=-2)  # BGC
        feature = feature.clone()
        feature = feature.roll(self.directions[direction], 1)
        feature[:, self.empty[direction]] = 0

        return feature.unsqueeze(2)


class OldTemporalEncoderEx(DeepEncoder):
    def __init__(self, in_dim, dim, Encoder=DeepEncoder):
        super().__init__(in_dim, dim)

        self.key_order = ['pprev', 'prev', 'next', 'nnext']
        self.directions = {'prev': 1, 'next': -1, 'pprev': 2, 'nnext': -2}
        self.empties = {'prev': [0], 'next': [-1], 'pprev': [0, 1], 'nnext': [-2, -1]}

        self.context_linear = nn.ModuleDict()
        for direction in self.directions.keys():
            self.context_linear[direction] = nn.Linear(dim, dim)

    def forward(self, feature, h=None):
        feature = super().forward(feature)
        # BGLC
        # build inter-group correlation
        fs = []
        for direction in self.key_order:
            fs.append(self.build_context(feature, direction))
        feature = torch.cat((*fs, feature), dim=-2)

        return feature

    def build_context(self, feature, direction='prev'):
        feature = self.context_linear[direction](feature)
        feature = feature.mean(dim=-2)  # BGC
        feature = feature.clone()
        feature = feature.roll(self.directions[direction], 1)
        for empty in self.empties[direction]:
            size = empty + 1 if empty >= 0 else -empty
            if feature.shape[1] >= size:
                feature[:, empty] = 0

        return feature.unsqueeze(2)

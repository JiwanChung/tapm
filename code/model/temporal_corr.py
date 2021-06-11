import torch
from torch import nn

from exp import ex
from .pretrain_aux import PretrainAuxGroupRoll
from .encoders import DeepEncoder

from .old_encoders import OldTemporalEncoderEx


class TemporalCorr(PretrainAuxGroupRoll):
    # include prev, next
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[TemporalEncoder(dim, self.gpt_dim)]))

    def _get_frame_loss(self, c, frame):
        # BGLC
        c = c[:, :, 2:]  # remove prev and next prediction
        return super()._get_frame_loss(c, frame)


class TemporalCorrEx(TemporalCorr):
    # include all five
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[TemporalEncoderEx(dim, self.gpt_dim)]))

    def _get_frame_loss(self, c, frame):
        # BGLC
        c = c[:, :, 4:]  # remove prev and next prediction
        return super()._get_frame_loss(c, frame)


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, dim, Encoder=DeepEncoder):
        super().__init__()
        self.encoder = Encoder(in_dim, dim)

        self.directions = {'prev': 1, 'next': -1}
        self.empty = {'prev': 0, 'next': -1}

        self.context_linear = nn.ModuleDict()
        for direction in self.directions.keys():
            self.context_linear[direction] = nn.Linear(dim, dim)

    def forward(self, feature, h=None):
        feature = self.encoder(feature)
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


class TemporalEncoderEx(nn.Module):
    def __init__(self, in_dim, dim, Encoder=DeepEncoder):
        super().__init__()
        self.encoder = Encoder(in_dim, dim)

        self.key_order = ['pprev', 'prev', 'next', 'nnext']
        self.directions = {'prev': 1, 'next': -1, 'pprev': 2, 'nnext': -2}
        self.empties = {'prev': [0], 'next': [-1], 'pprev': [0, 1], 'nnext': [-2, -1]}

        self.context_linear = nn.ModuleDict()
        for direction in self.directions.keys():
            self.context_linear[direction] = nn.Linear(dim, dim)

    def forward(self, feature, h=None):
        feature = self.encoder(feature)
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


class TemporalCorrGlobal(TemporalCorrEx):
    # include all five
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[TemporalEncoderGlobal(dim, self.gpt_dim)]))

    def _get_frame_loss(self, c, frame):
        # BGLC
        c = c[:, :, 5:]  # remove global, prev and next prediction
        return super()._get_frame_loss(c, frame)


class TemporalEncoderGlobal(TemporalEncoderEx):
# class TemporalEncoderGlobal(OldTemporalEncoderEx):
    def __init__(self, in_dim, dim, Encoder=DeepEncoder):
        super().__init__(in_dim, dim, Encoder)

        self.context_linear['global'] = nn.Linear(dim, dim)

    def forward(self, feature, h=None):
        feature = self.encoder(feature)
        # BGLC
        # build inter-group correlation
        fs = [self.build_context_global(feature)]
        for direction in self.key_order:
            fs.append(self.build_context(feature, direction))
        feature = torch.cat((*fs, feature), dim=-2)

        return feature

    def global_pool(self, f):
        f = self.context_linear['global'](f)
        return f.mean(dim=-2)

    def build_context_global(self, feature):
        B, G, _, C = feature.shape
        feature = self.global_pool(feature)  # BGC
        feature = feature.mean(dim=-2)  # BC

        feature = feature.contiguous().view(B, 1, 1, C)
        feature = feature.repeat(1, G, 1, 1)
        return feature


class TemporalEncoderGlobalAR(TemporalEncoderEx):
    def __init__(self, in_dim, dim, Encoder=DeepEncoder):
        super().__init__(in_dim, dim, Encoder)

        self.context_linear['global'] = nn.Linear(dim, dim)

    def forward(self, feature, h=None):
        feature = super().forward(feature)
        # BGLC
        # build inter-group correlation
        fs = [self.build_context_global(feature)]
        for direction in self.key_order:
            fs.append(self.build_context(feature, direction))
        feature = torch.cat((*fs, feature), dim=-2)

        return feature

    def global_pool(self, f):
        f = self.context_linear['global'](f)
        return f.mean(dim=-2)

    def build_context_global(self, feature):
        B, G, L, C = feature.shape
        feature = self.global_pool(feature)  # BGC
        feature = feature.mean(dim=-2)  # BC

        feature = feature.contiguous().view(B, 1, 1, C)
        feature = feature.repeat(1, G, 1, 1)
        return feature

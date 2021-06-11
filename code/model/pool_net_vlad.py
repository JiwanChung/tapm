from torch import nn

from exp import ex
from .temporal_corr import (
    TemporalCorrEx, TemporalEncoderGlobal, TemporalCorrGlobal
)
from .NetVLAD import NetVLADLinear
from .net_vlad_like import NetVLADLike


class PoolNetVLAD(TemporalCorrEx):
    @classmethod
    def get_args(cls):
        return {
            **super().get_args(),
            'nvlad_size': 256,
        }

    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch,
                 nvlad_size):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        self.feature_pooler = NetVLADLinear(nvlad_size, self.gpt_dim)
        self.text_pooler = NetVLADLinear(nvlad_size, self.gpt_dim)

    def mean_pool_text(self, o):
        o = self.text_pooler(o)
        return o

    def mean_pool_features(self, features):
        return sum([self.feature_pooler(x) for x in features.values()])


class GlobalNetVLAD(TemporalCorrGlobal):
    @classmethod
    def get_args(cls):
        return {
            **super().get_args(),
            'nvlad_size': 256,
        }

    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch,
                         nvlad_size):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[TemporalEncoderGlobalNetVLAD(dim, self.gpt_dim,
                                                                 nvlad_size)]))


class TemporalEncoderGlobalNetVLAD(TemporalEncoderGlobal):
    def __init__(self, in_dim, dim, nvlad_size):
        super().__init__(in_dim, dim)

        self.context_linear['global'] = NetVLADLinear(nvlad_size, dim)

    def global_pool(self, f):
        f = self.context_linear['global'](f)
        return f


class GlobalNetVLADLike(TemporalCorrGlobal):
    @classmethod
    def get_args(cls):
        return {
            **super().get_args(),
            'nvlad_size': 256,
        }

    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch,
                         nvlad_size, concentration):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[TemporalEncoderGlobalNetVLADLike(dim, self.gpt_dim,
                                                                 nvlad_size,
                                                                 concentration)]))


class TemporalEncoderGlobalNetVLADLike(TemporalEncoderGlobal):
    def __init__(self, in_dim, dim, nvlad_size, concentration):
        super().__init__(in_dim, dim)

        self.context_linear['global'] = NetVLADLike(nvlad_size, dim, concentration)

    def global_pool(self, f):
        f = self.context_linear['global'](f)
        return f

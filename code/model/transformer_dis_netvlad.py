import torch
from torch import nn
import torch.nn.functional as F

from .transformer_dis import FeatureEncoder
from .transformer_dis_group import TransformerDisGroupReg
from .NetVLAD import NetVLADWrapper


class TransformerDisGroupNetVLAD(TransformerDisGroupReg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                nn.Sequential(*[NetVLADWrapper(feature_size=dim, cluster_size=48),
                                FeatureEncoder(dim * 48, self.gpt_dim)]))

    @classmethod
    def get_args(cls, args):
        args.segment_pool_type = None
        return args

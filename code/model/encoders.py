import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .modules import Residual, MLP, TransformerAttention


class DeepEncoder(nn.Module):
    def __init__(self, in_dim, dim):
        super(DeepEncoder, self).__init__()

        self.num_layers = 3

        self.linear_in = nn.Linear(in_dim, dim)
        self.linear2 = nn.Linear(dim, dim)

        self.res_layers = nn.Sequential(
            *[Residual(MLP(dim)) for i in range(self.num_layers)]
        )

    def forward(self, feature, h=None):
        feature = self.linear_in(feature)
        feature = F.leaky_relu(feature)
        feature = self.linear2(feature)
        feature = F.leaky_relu(feature)
        feature = self.res_layers(feature)
        return feature


class AttEncoder(nn.Module):
    def __init__(self, in_dim, dim):
        super(AttEncoder, self).__init__()

        self.num_layers = 3

        self.linear_in = nn.Linear(in_dim, dim)
        self.sas = nn.Sequential(
            *[TransformerAttention(dim) for i in range(self.num_layers)]
        )

    def forward(self, feature, h=None):
        feature = self.linear_in(feature)  # BGLC
        G = feature.shape[1]
        feature = rearrange(feature, 'b g l c -> b (g l) c').contiguous()
        feature = self.sas(feature)
        feature = rearrange(feature, 'b (g l) c -> b g l c', g=G).contiguous()
        return feature


class NoPositionEncoder(nn.Module):
    def __init__(self, in_dim, dim, Encoder=DeepEncoder):
        super().__init__()
        self.encoder = Encoder(in_dim, dim)

    def forward(self, feature, h=None):
        # e.g. for 2nd picture, [1,2,3,4,5,2]
        feature = self.encoder(feature)  # BGLC
        G = feature.shape[1]
        context = rearrange(feature.contiguous(), 'b g l c -> b (g l) c')
        context = context.unsqueeze(1).expand(-1, G, -1, -1)
        feature = torch.cat((context, feature), dim=2)  # BG(GL+L)C

        return feature

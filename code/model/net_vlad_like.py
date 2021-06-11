import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class NetVLADLike(nn.Module):
    def __init__(self, dim, num_clusters, concentration):
        super().__init__()

        self.dim = dim
        self.num_clusters = num_clusters
        self.concentration = concentration

        self._conv = nn.Conv1d(dim, num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(dim, num_clusters))
        self.eps = nn.Parameter(torch.FloatTensor([1e-12]), requires_grad=False)
        self._init_params()

    def _init_params(self):
        self._conv.weight = nn.Parameter(
            (2.0 * self.concentration * self.centroids).unsqueeze(-1)
        )
        self._conv.bias = nn.Parameter(
            - self.concentration * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        # BGLC
        if x.dim() == 4:
            G = x.shape[1]
            x = rearrange(x.contiguous(), 'b g l c -> (b g) l c')
            x = self._forward(x)
            x = rearrange(x.contiguous(), '(b g) c -> b g c', g=G)
        elif x.dim() == 3:
            x = self._forward(x)
        return x

    def conv(self, x):
        x = rearrange(x, 'b l c -> b c l')
        x = self._conv(x)
        x = rearrange(x, 'b c k -> b k c')
        return x

    def _forward(self, x):
        # BLC
        scale = (x ** 2).mean(dim=1, keepdim=True)
        direction = x / torch.max(scale, self.eps)
        direction = self.run_vlad(direction)
        x = scale * direction
        return x.mean(dim=1)  # BC

    def run_vlad(self, x):
        # BLC
        B, L = x.shape[:2]
        K = self.centroids.shape[0]
        soft_assign = self.conv(x)  # BKC
        soft_assign = F.softmax(soft_assign, dim=1)

        centroids = self.centroids.view(1, 1, *self.centroids.shape).repeat(B, L, 1, 1)
        x = x.unsqueeze(2).repeat(1, 1, K, 1)
        residual = x - centroids  # BLKC
        residual *= soft_assign.unsqueeze(-1)
        vlad = residual.sum(dim=1)  # BKC

        return vlad

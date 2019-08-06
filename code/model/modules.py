from torch import nn


class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x

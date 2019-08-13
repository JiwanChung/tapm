import torch
from torch import nn


class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


class BinaryLayer(torch.autograd.Function):
    def forward(self, net_input):
        return (net_input > 0).float().to(net_input.device)

    def backward(self, grad_output):
        return grad_output.clamp(0, 1)


def saturating_sigmoid(x):
    return (1.2 * torch.sigmoid(x) - 0.1).clamp(0, 1)


def l_n_norm(x, dim=0, n=1):
    if n > 0:
        x = torch.abs(x)
        return (x ** n).sum(dim=dim) ** (1 / n)
    elif n == 0:
        x = torch.abs(x)
        x = x.clamp(0, 1)
        f = BinaryLayer()
        x = f(x)
        return x.sum(dim=dim)


class LSTMDecoder(nn.Module):
    def __init__(self, embedding):
        super(LSTMDecoder, self).__init__()

        self.num_layers = 2

        self.wte = embedding
        self.dim = embedding.weight.shape[1]
        self.decoder = nn.GRU(self.dim, self.dim, self.num_layers,
                              bidirectional=False,
                              batch_first=True)

    def out(self, x):
        return torch.matmul(x, self.wte.weight.t())

    def forward(self, h, targets):
        # BC -> NBC
        h = h.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        # BL -> BLC
        targets = self.wte(targets)
        logits, _ = self.decode(targets, h)
        return logits

    def decode(self, s, h):
        o, h = self.decoder(s, h)
        o = self.out(o)

        return o, h

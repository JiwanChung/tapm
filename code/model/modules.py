import torch
from torch import nn
import torch.nn.functional as F


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


class CutInfLayer(torch.autograd.Function):
    def forward(self, net_input):
        return net_input

    def backward(self, grad_output):
        grad_output[grad_output == float('inf')] = 0
        return grad_output


def saturating_sigmoid(x):
    return (1.2 * torch.sigmoid(x) - 0.1).clamp(0, 1)


def l_n_norm(x, dim=0, n=1, normalize=True):
    if n > 0:
        f = CutInfLayer()
        x = f(x)
        x = (x ** n).sum(dim=dim)
        if normalize:
            x = x ** (1 / n)
        return x
    elif n == 0:
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

    def forward(self, h, targets, embedded=False):
        # BC -> NBC
        h = h.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        # BL -> BLC
        if not embedded:
            targets = self.wte(targets)
        logits, _ = self.decode(targets, h)
        return logits

    def decode(self, s, h):
        o, h = self.decoder(s, h)
        o = self.out(o)

        return o, h


class GRU(nn.Module):
    '''
    batch_first GRU
    '''
    def __init__(self, num_layers, in_dim, out_dim, dropout=0.1, bidirectional=False):
        super(GRU, self).__init__()

        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.decoder = nn.GRU(self.in_dim, self.out_dim, self.num_layers,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              batch_first=True)

    def init_c(self, B, C, device=0):
        return torch.zeros(B, self.num_layers, C).float().to(device)

    def init_h(self, B, device=0):
        if isinstance(self.decoder, nn.LSTM):
            h = (torch.zeros(B, self.num_layers, self.out_dim).float().to(device),
                    torch.zeros(B, self.num_layers, self.out_dim).float().to(device))
        else:
            h = torch.zeros(B, self.num_layers, self.out_dim).float().to(device)
        return h

    @staticmethod
    def transpose(h):
        if isinstance(h, tuple):
            h = [i.transpose(0, 1) for i in h]
        else:
            h = h.transpose(0, 1)
        return h

    def forward(self, s, h, **kwargs):
        h = self.transpose(h)
        o, h = self.decoder(s, h)
        h = self.transpose(h)  # BLC, BNC
        return o, h


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim
        Returns:
            Variable -- context vector of size batch_size x dim
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state),
                           2).view(-1, self.dim * 2)
        o = self.linear2(torch.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0):
        super(ResBlock, self).__init__()

        self.dim = dim

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.dim, self.dim)
        self.linear2 = nn.Linear(self.dim, self.dim)
        self.layer_norm1 = nn.LayerNorm(self.dim)
        self.layer_norm2 = nn.LayerNorm(self.dim)

        self.reset_parameters()

    def reset_parameters(self):
        initScale = 0.1

        self.linear1.weight.data.uniform_(-initScale, initScale)
        self.linear1.bias.data.zero_()

        self.linear2.weight.data.uniform_(-initScale, initScale)
        self.linear2.bias.data.zero_()

    def forward(self, x):
        x_prev = x
        x = self.layer_norm1(x)
        x = torch.tanh(x)
        x = self.linear1(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x_prev + x

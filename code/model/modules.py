import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


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


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super(MLP, self).__init__()

        if out_dim is None:
            out_dim = in_dim
        self.dim = out_dim

        self.l1 = nn.Linear(in_dim, self.dim)
        self.l2 = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim=None, v_dim=None, m_dim=None, heads=1):
        super().__init__()

        if k_dim is None:
            k_dim = q_dim
        if v_dim is None:
            v_dim = k_dim
        if m_dim is None:
            m_dim = q_dim

        heads = 1 if q_dim < heads else heads
        heads = 1 if k_dim < heads else heads
        heads = 1 if v_dim < heads else heads

        assert q_dim % heads == 0, f"q_dim: {q_dim} / n_heads: {heads} must be divisible"
        assert k_dim % heads == 0, f"k_dim: {k_dim} / n_heads: {heads} must be divisible"
        assert v_dim % heads == 0, f"v_dim: {v_dim} / n_heads: {heads} must be divisible"
        assert m_dim % heads == 0, f"m_dim: {m_dim} / n_heads: {heads} must be divisible"

        self.q = nn.Linear(q_dim // heads, m_dim // heads)
        self.k = nn.Linear(k_dim // heads, m_dim // heads)
        self.v = nn.Linear(v_dim // heads, m_dim // heads)
        self.heads = heads

    def forward(self, q, k=None, v=None, bidirectional=False):
        if k is None:
            k = q.clone()
        if v is None:
            v = k.clone()
        # BLC

        q = rearrange(q, 'b q (h c) -> b h q c', h=self.heads)
        k = rearrange(k, 'b k (h c) -> b h k c', h=self.heads)
        v = rearrange(v, 'b k (h c) -> b h k c', h=self.heads)

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        a = torch.einsum('bhqc,bhkc->bhqk', q, k)
        a = a / math.sqrt(k.shape[-1])
        a_q = F.softmax(a, dim=-1)  # bhqk
        q_new = torch.einsum('bhqk,bhkc->bhqc', a_q, v)
        q_new = rearrange(q_new, 'b h q c -> b q (h c)')

        if bidirectional:
            a_v = F.softmax(a, dim=-2)  # bhqk
            v = torch.einsum('bhqk,bhqc->bhkc', a_v, q)
            v = rearrange(v, 'b h k c -> b k (h c)')
            return q_new, v
        else:
            return q_new


class SelfAttention(MultiHeadAttention):
    def __init__(self, q_dim, m_dim=None, heads=1):
        super().__init__(q_dim, k_dim=q_dim, v_dim=q_dim, m_dim=m_dim, heads=heads)

    def forward(self, q, bidirectional=False):
        return super().forward(q, q, q, bidirectional=bidirectional)


class Pffn(nn.Module):
    def __init__(self, dim, large_dim):
        super().__init__()

        self.in_linear = nn.Linear(dim, large_dim)
        self.out_linear = nn.Linear(large_dim, dim)

    def forward(self, x):
        x = self.in_linear(x)
        x = F.relu(x)
        x = self.out_linear(x)
        return x


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.m = module

    def forward(self, x):
        return x + self.m(x)


class TransformerAttention(nn.Module):
    def __init__(self, q_dim, m_dim=None, heads=8):
        super().__init__()

        self.att = SelfAttention(q_dim, m_dim=m_dim, heads=heads)

        self.layer_norm = nn.LayerNorm(q_dim)

    def forward(self, q):
        q_new = self.att(q, bidirectional=False)
        return self.layer_norm(q + q_new)

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch
#from torchqrnn import QRNN

from exp import ex


class PretrainedLSTM(nn.Module):
    rnn_type = 'LSTM'

    def __init__(self, tokenizer, pretrained=True):
        super().__init__()

        if not hasattr(self, 'args'):
            self.args = {
                'ninp': 200,
                'nhid': 200,
                'nlayers': 2,
                'dropout': 0.2,
                'ntoken': len(tokenizer.prev_vocab),
            }
        self.args['rnn_type'] = self.rnn_type
        self.rnn = RNNModel(**self.args)
        self.pretrained = pretrained
        if self.pretrained:
            pretrained_weight = self.load(self.rnn_type)
            self.rnn.load_state_dict(pretrained_weight)
        self.rnn.ninp = self.args['ninp']
        self.tokenizer = tokenizer

        self.extend_word_embedding()

        self.lm_head = self.rnn.decoder
        self.rnn.wte = self.rnn.encoder
        self.transformer = self.rnn

    @ex.capture
    def load(self, rnn_type, root):
        with open(root / 'data' / 'pretrained' / f"{rnn_type.lower()}.pt", 'rb') as f:
            weight = torch.load(f)
        return weight

    def extend_word_embedding(self):
        prev_w = self.rnn.encoder.weight.data  # prev_len
        self.rnn.encoder = nn.Embedding(len(self.tokenizer), self.rnn.ninp)
        print(f"extending vocab: {prev_w.shape[0]} -> {self.rnn.encoder.weight.shape[0]}")
        prev_vocab = list(self.tokenizer.prev_vocab.values())
        ma, mi = max(prev_vocab), min(prev_vocab)
        self.rnn.encoder.weight[mi: ma].data = prev_w[mi: ma]

        prev_w = self.rnn.decoder.weight.data
        self.rnn.decoder = nn.Linear(self.rnn.ninp, len(self.tokenizer))
        self.rnn.decoder.weight[mi: ma].data = prev_w[mi: ma]


'''
class PretrainedQRNN(PretrainedLSTM):
    rnn_type = 'QRNN'
'''


@ex.capture
def load_vocab(rnn_type, root):
    with open(root / 'data' / 'pretrained' / f"vocab_{rnn_type.lower()}.json", 'r') as f:
        w2i = json.load(f)
    return w2i


'''
from pytorch/examples
'''


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        # elif rnn_type is 'QRNN':
        #     self.rnn = QRNN(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.config = Munch({'d_model': ninp})

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def run_cell(self, input):
        input = input.transpose(0, 1)
        emb = self.drop(input)
        hidden = self.init_hidden(emb.shape[1])
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output = output.transpose(0, 1)
        return output


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

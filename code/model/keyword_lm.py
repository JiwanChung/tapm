import torch
from torch import nn

from data.batcher import make_keyword_batch

from .transformer_model import TransformerModel


class LSTMKeywordLM(TransformerModel):
    transformer_name = 'bert'

    def __init__(self, args, transformer, tokenizer):
        super(LSTMKeywordLM, self).__init__()

        self.V = len(tokenizer)
        self.dim = 768
        self.num_layers = 2

        self.wte = nn.Embedding(self.V, self.dim)
        self.encoder = nn.GRU(self.dim, self.dim, self.num_layers,
                              bidirectional=False,
                              batch_first=True)
        self.decoder = nn.GRU(self.dim, self.dim, self.num_layers,
                              bidirectional=False,
                              batch_first=True)

    @staticmethod
    def make_batch(*args, **kwargs):
        return make_keyword_batch(*args, **kwargs, concat=False)

    def out(self, x):
        return torch.matmul(x, self.wte.weight)

    def forward(self, sentences, lengths, targets, keywords):
        k = self.wte(keywords)
        s = self.wte(sentences)
        _, h_k = self.encoder(k)
        o, _ = self.decoder(s, h_k)
        o = self.out(o)

        return o, targets, None, None, None

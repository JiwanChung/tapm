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
        self.use_keyword = args.use_keyword

        self.tokenizer = tokenizer

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
        return torch.matmul(x, self.wte.weight.t())

    def forward(self, sentences, lengths, targets, keywords):
        h = self.encode(keywords)
        s = self.wte(sentences)

        logits, _ = self.decode(s, h)
        return logits, targets, None, None, keywords

    def encode(self, keywords):
        h_k = None
        if self.use_keyword:
            k = self.wte(keywords)
            _, h_k = self.encoder(k)
        return h_k

    def decode(self, s, h):
        o, h = self.decoder(s, h)
        o = self.out(o)

        return o, h

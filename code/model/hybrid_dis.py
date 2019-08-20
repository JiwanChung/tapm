from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import make_feature_lm_batch

from .modules import Attention, GRU
from .transformer_model import TransformerModel


class HybridDis(TransformerModel):
    transformer_name = 'bert'

    def __init__(self, args, transformer, tokenizer):
        super(HybridDis, self).__init__()

        self.dim = args.get('dim', 512)
        self.video_dim = args.get('video_dim', 1024)
        self.image_dim = args.get('image_dim', 2048)
        self.dropout_ratio = args.get('dropout', 0.5)
        self.feature_names = args.get('feature_names',
                                      ['video', 'image', 'box'])
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)

        for feature in self.feature_names:
            setattr(self, feature, FeatureEncoder(getattr(self, f"{feature}_dim"), self.dim))
        self.encoder = nn.Linear(len(self.feature_names) * self.dim, self.dim)
        self.wte = nn.Embedding(self.vocab_size, self.dim)
        self.rnn = GRU(1, 3 * self.dim, self.dim, dropout=self.dropout_ratio)
        self.prev_encoder = PrevEncoder(self.dim)
        self.dropout = nn.Dropout(self.dropout_ratio)

    def make_batch(self, *args, **kwargs):
        return make_feature_lm_batch(*args, **kwargs)

    def out(self, x):
        return torch.matmul(x, self.wte.weight.t())

    def forward(self, batch, **kwargs):
        # BVLC, BVL
        sentences = batch.sentences
        B, V, L = sentences.shape[:3]
        outputs = []

        def run(h, c, v, w):
            features = OrderedDict({})
            for feature in self.feature_names:
                features[feature] = getattr(self, feature)(getattr(batch, feature)[:,v], h)
            features = self.encoder(torch.cat(list(features.values()), dim=-1))
            s = sentences[:,v,w].clone()
            s = self.wte(s).unsqueeze(1)  # B1C
            s = torch.cat((features, c, s), dim=-1)
            o, h = self.rnn(s, h)
            logits = self.out(o)  # BV
            return h, c, logits

        res = []
        eos_flags = torch.LongTensor([0] * B).byte().to(sentences.device)
        empty = torch.zeros(B, self.vocab_size).to(sentences.device)
        h, c = self.rnn.init_hiddens(sentences)
        for v in range(V):
            sent = []
            for w in range(L):
                if eos_flags.all():
                    logits = empty.clone()
                else:
                    h, c, logits = run(h, c, v, w)
                    if self.training:
                        eos_flags = eos_flags | (sentences[:, v, min(L - 1, w + 1)] == self.tokenizer.sep_id)
                    else:
                        eos_flags = eos_flags | (logits.argmax(dim=-1) == self.tokenizer.sep_id)
                sent.append(logits)

            c = self.prev_encoder(h)
            res.append(torch.stack(sent, 1).contiguous())  # BLV
        return torch.stack(res, 1).contiguous(), batch.targets, None, {}, None


class FeatureEncoder(nn.Module):
    def __init__(self, video_dim, dim):
        super(FeatureEncoder, self).__init__()

        self.linear = nn.Linear(video_dim, dim)
        self.attention = Attention(dim)

    def forward(self, feature, h):
        # BLC
        if isinstance(h, tuple):  # check LSTM/GRU
            h = h[0]
        feature = self.linear(feature)
        h = h.mean(dim=1)
        return self.attention(h, feature).unsqueeze(1)


class PrevEncoder(nn.Module):
    def __init__(self, dim):
        super(PrevEncoder, self).__init__()

        self.linear = nn.Linear(dim, dim)

    def forward(self, h):
        # BLC
        if isinstance(h, tuple):  # check LSTM/GRU
            h = h[0]
        return self.linear(h)

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import make_feature_lm_batch_with_keywords

from .modules import Attention, GRU
from .scn_rnn import SCNLSTM
from .transformer_model import TransformerModel


'''
currently, this implementation deviates from the original repo
in the following regards:
    1. GRU instead of LSTM
    2. An (deactivated) option to share in_out embeddings
Aside from the above, I tried to closely follow the given details.
'''
class HybridDis(TransformerModel):
    transformer_name = 'none'  # assign transformer_name = 'bert' to use BPE
    model_type = 'caption'
    use_keyword = False

    def __init__(self, args, transformer, tokenizer):
        super(HybridDis, self).__init__()

        self.dim = args.get('dim', 512)
        self.video_dim = args.get('video_dim', 1024)
        self.image_dim = args.get('image_dim', 2048)
        self.keyword_num = args.get('keyword_num', 1000)
        self.dropout_ratio = args.get('dropout', 0.5)
        self.feature_names = args.get('feature_names',
                                      ['video', 'image', 'box'])
        self.share_in_out = args.get('share_in_out',
                                     False)
        self.max_target_len = args.max_target_len
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)

        for feature in self.feature_names:
            setattr(self, feature, FeatureEncoder(getattr(self, f"{feature}_dim"), self.dim))
        self.encoder = nn.Linear(len(self.feature_names) * self.dim, self.dim)
        self.wte = nn.Embedding(self.vocab_size, self.dim)
        self.context_dim = self.dim // 4
        num_layers = 1
        self.rnn = {
            'rnn': GRU(num_layers, 2 * self.dim + self.context_dim, self.dim, dropout=self.dropout_ratio),
            'scn': SCNLSTM(2 * self.dim + self.context_dim, self.keyword_num, self.dim,
                           num_layers, batch_first=True, dropout=self.dropout_ratio)
        }[args.get('decoder_type', 'rnn')]
        self.context_encoder = PrevEncoder(self.dim, self.context_dim)
        self.dropout = nn.Dropout(self.dropout_ratio)

        if self.share_in_out:
            self.out = self.out_shared
        else:
            self.out = nn.Linear(self.dim, self.vocab_size)
        self.init_weights()
        self.use_context = False

    def make_batch(self, *args, **kwargs):
        return make_feature_lm_batch_with_keywords(*args, **kwargs)

    def epoch_update(self, epoch):
        if epoch > 10:
            self.context = True

    def init_weights(self):
        init_range = 0.1
        for feature in self.feature_names:
            getattr(self, feature).linear.weight.data.uniform_(-init_range, init_range)
        if not self.share_in_out:
            self.out.bias.data.fill_(0)
            self.out.weight.data.uniform_(-init_range, init_range)

    def out_shared(self, x):
        return torch.matmul(x, self.wte.weight.t())

    def run_token(self, features, s, h, c, keyword):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        for feature in self.feature_names:
            features[feature] = getattr(self, feature)(features[feature], h)
        features = self.encoder(torch.cat(list(features.values()), dim=-1))
        s = self.wte(s).unsqueeze(1)  # B1C
        s = torch.cat((features, c, s), dim=-1)
        o, h = self.rnn(s, h, keyword=keyword)
        logits = self.out(o)  # BV
        return h, c, logits

    def run_video(self, features, c, v, L, sentences=None, sampler=None, keyword=None):
        video = features['video']
        B = video.shape[0]
        empty = torch.full((B, self.vocab_size), float('-inf')).to(video.device)
        sent = []
        eos_flags = torch.LongTensor([0] * B).byte().to(video.device)
        h = self.rnn.init_h(B, device=video.device)
        c = self.rnn.init_c(B, self.context_dim, device=video.device)
        s0 = sentences[:, v, 0] if sentences is not None \
            else torch.Tensor([self.tokenizer.cls_id]).long().to(video.device)
        s = s0
        hypo = s0.unsqueeze(-1)

        for w in range(L):
            if eos_flags.all():
                logits = empty.clone()
            else:
                h, c, logits = self.run_token(features, s, h, c, keyword=keyword)
                if sentences is not None:  # training
                    s = sentences[:, v, min(L - 1, w + 1)].clone()
                    eos_flags = eos_flags | (sentences[:, v, min(L - 1, w + 1)] == self.tokenizer.sep_id)
                else:
                    s, probs = sampler(logits, hypo)
                    eos_flags = eos_flags | (logits.argmax(dim=-1) == self.tokenizer.pad_id)
            hypo = torch.cat((hypo, s.unsqueeze(-1)), dim=1)
            sent.append(logits)
        if sentences is None:
            hypo = hypo[:, 1:][probs.argmax(dim=-1)]
        else:
            sent = torch.stack(sent, 1).contiguous()
        c = self.context_encoder(h)
        if not self.use_context:
            c = torch.full_like(c.detach(), 0)
            c.requires_grad_(False)
        return c, sent, hypo

    def forward(self, batch, **kwargs):
        # BVLC, BVL
        video = batch.video
        B, V = video.shape[:2]
        L = batch.sentences.shape[2] if hasattr(batch, 'sentences') else self.max_target_len
        sent_gt = batch.sentences if hasattr(batch, 'sentences') else None

        res = []
        for v in range(V):
            features = {k: val[:, v] for k, val \
                        in {f: getattr(batch, f) for f \
                            in self.feature_names}.items()}
            c = self.rnn.init_c(B, self.context_dim, device=video.device)
            keyword = batch.keywords[:, v] if hasattr(batch, 'keywords') else None
            c, sent, _ = self.run_video(features, c, v, L, sentences=sent_gt, keyword=keyword)
            res.append(sent)  # BLV
        del batch.sentences  # for generation
        return torch.stack(res, 1).contiguous(), batch.targets, None, {}, batch


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
    def __init__(self, in_dim, out_dim):
        super(PrevEncoder, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h):
        # BLC
        if isinstance(h, tuple):  # check LSTM/GRU
            h = h[0]
        return self.linear(h)

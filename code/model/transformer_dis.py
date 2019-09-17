from collections import OrderedDict
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F

from transformers import transformer_embed, transformer_run_cells
from .hybrid_dis import HybridDis
from debug_utils import timeit
from .keyword_classifier import KeywordClassifier


class TransformerDis(HybridDis):
    transformer_name = 'gpt2'  # assign transformer_name = 'bert' to use BPE

    def __init__(self, args, transformer, tokenizer):
        super(TransformerDis, self).__init__(args, transformer, tokenizer)

        self.dropout_ratio = args.get('dropout', 0.5)

        self.net = transformer
        self.net.train()
        self.gpt_dim = self.net.transformer.config.n_embd

        self.keyword_classifier = KeywordClassifier(
            self.net.transformer.wte,
            self.keyword_num, self.dim, self.feature_names,
            self.video_dim, self.image_dim, self.flow_dim, self.dropout_ratio,
            recall_k=self.k,
            loss_type=self.keyword_loss_type
        )

        for feature in self.feature_names:
            setattr(self, feature, FeatureEncoder(getattr(self, f"{feature}_dim"), self.gpt_dim))

        self.reduce_cat = nn.Linear(self.gpt_dim + self.keyword_num, self.gpt_dim)
        self.reduce_c = nn.Linear(self.gpt_dim, self.dim)

        self.dropout = nn.Dropout(self.dropout_ratio)

    def add_keyword(self, h, keyword):
        h = torch.cat((h, # features.unsqueeze(1).expand(-1, h.shape[1], -1),
                       keyword.unsqueeze(1).expand(-1, h.shape[1], -1)), dim=-1)
        h = self.reduce_cat(h)
        return h

    def get_logits(self, o, keyword, gt=None):
        return self.net.lm_head(o), None, {}

    def merge_context(self, features, cls_embd, sep_embd):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        return torch.cat((cls_embd, *chain(*[(feature, sep_embd) for feature in features.values()])), dim=1)

    def run_transformer(self, hypo, features, keyword):
        h, past, head_mask = transformer_embed(self.net.transformer, hypo)
        h = self.add_keyword(h, keyword)

        cls_embd = self.net.transformer.wte(torch.LongTensor([self.tokenizer.cls_id]).to(h.device))
        sep_embd = self.net.transformer.wte(torch.LongTensor([self.tokenizer.sep_id]).to(h.device))
        B, L, C = h.shape
        cls_embd = cls_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        sep_embd = sep_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        context = self.merge_context(features, cls_embd, sep_embd)
        h = torch.cat((context, h), dim=1)

        o = transformer_run_cells(self.net.transformer, h, past=past, head_mask=head_mask)[0]
        context_embedded = o[:, :context.shape[1]]
        o = o[:, context.shape[1]:]

        return o, context_embedded

    def run_transformer_get_loss(self, hypo, features, keyword, group_mask=None, gt=None):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        res = OrderedDict()
        for feature in self.feature_names:
            res[feature] = getattr(self, feature)(features[feature], None)
        o, _ = self.run_transformer(hypo, res, keyword)

        o = self.dropout(o)
        c = o.mean(dim=1)
        c = self.reduce_c(c)
        logits, loss, stats = self.get_logits(o, keyword, gt)
        return logits, c, loss, stats

    def run_token(self, features, hypo, h, c, group_mask, keyword):
        logits, h = self.generate_token(hypo, features, c, h, group_mask, keyword)
        return h, c, logits

    def run_train(self, hypo, features, keyword, group_mask=None):
        return self.run_transformer_get_loss(hypo, features, keyword, group_mask, gt=hypo)

    def generate_token(self, hypo, features, c, h, group_mask, keyword):
        logits, h, _, _ = self.run_transformer_get_loss(hypo, features, keyword, group_mask)
        return logits, h

    def run_video(self, features, c, v, L, sentences=None, sampler=None,
                  keyword=None, reduce_hypo=True, group_mask=None):
        video = features['video']
        B = video.shape[0]
        empty = torch.full((B, self.vocab_size), float('-inf')).to(video.device)
        sent = []
        eos_flags = torch.LongTensor([0] * B).byte().to(video.device)
        if c is None:
            c = self.rnn.init_c(B, self.context_dim, device=video.device) if hasattr(self, 'rnn') else None
        s0 = sentences[:, v, 0] if sentences is not None \
            else torch.Tensor([self.tokenizer.cls_id]).long().to(video.device).expand(B)
        s = s0
        hypo = s0.unsqueeze(-1)

        stats = {}
        small_loss = None
        if sentences is not None:  # training
            sent, h, small_loss, stats = self.run_train(sentences[:, v], features, keyword, group_mask)
        else:
            for w in range(L):
                if eos_flags.all():
                    logits = empty.clone()
                else:
                    h = None
                    h, c, logits = self.run_token(features, hypo, h, c, group_mask, keyword=keyword)
                    s, probs = sampler(logits)
                    eos_flags = eos_flags | (logits[:, -1].argmax(dim=-1) == self.tokenizer.pad_id)
                hypo = torch.cat((hypo, s.unsqueeze(-1)), dim=1)
                sent.append(logits)

            hypo = hypo[:, 1:]
            if reduce_hypo:
                hypo = hypo[probs.argmax(dim=-1)]
        c = self.context_encoder(h)
        if not self.use_context:
            c = torch.full_like(c.detach(), 0)
            c.requires_grad_(False)
        return c, sent, hypo, small_loss, stats


class FeatureEncoder(nn.Module):
    def __init__(self, video_dim, dim):
        super(FeatureEncoder, self).__init__()

        self.linear = nn.Linear(video_dim, dim)

    def forward(self, feature, h):
        # BLC
        feature = self.linear(feature)
        return feature

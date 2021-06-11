from collections import OrderedDict
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F

from exp import ex

from run_transformer import transformer_embed, transformer_run_cells
from .hybrid_dis import HybridDis
from .keyword_classifier import KeywordClassifier
from .NetVLAD import NetVLADWrapper


class TransformerDis(HybridDis):
    transformer_name = 'gpt2'  # assign transformer_name = 'bert' to use BPE

    @classmethod
    def get_args(cls):
        return {
            **super().get_args(),
            'dropout_before': False,
        }

    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before):
        super(TransformerDis, self).__init__(transformer, tokenizer)

        self.dropout_before = dropout_before

        self.net = transformer
        self.net.train()
        config = self.net.transformer.config
        for name in ['emb_dim', 'n_embd', 'd_model', 'hidden_size']:
            if hasattr(config, name):
                self.gpt_dim = getattr(config, name)
                break
        assert hasattr(self, 'gpt_dim'), "no dimension specified"

        if not hasattr(self.net.transformer, 'word_embedding'):
            if not hasattr(self.net.transformer, 'wte'):
                if not hasattr(self.net.transformer, 'word_emb'):
                    if not hasattr(self.net.transformer, 'w'):
                        self.net.transformer.w = self.net.transformer.embeddings
                    self.net.transformer.word_emb = self.net.transformer.w
                self.net.transformer.wte = self.net.transformer.word_emb
            self.net.transformer.word_embedding = self.net.transformer.wte
            '''
            self.net.transformer.wte = self.net.transformer.word_embedding

        if hasattr(self.net.transformer, 'word_embedding'):
            del self.net.transformer.word_embedding
            '''

        '''
        self.keyword_classifier = KeywordClassifier(
            self.net.transformer.word_embedding,
            self.keyword_num, self.dim, self.feature_names,
            self.feature_dims,
            self.dropout_ratio,
            recall_k=self.k,
            loss_type=self.keyword_loss_type
        )
        '''
        def chain(input_, f_list):
            for op in f_list:
                input_ = op(input_)
            return input_
        for feature in self.feature_names:
            if feature in ['human_i3d']:
                setattr(self, feature,
                        nn.Sequential(*[NetVLADWrapper(feature_size=1536, cluster_size=24),
                                  FeatureEncoder(1536 * 24, self.gpt_dim)]))
                continue
            setattr(self, feature, FeatureEncoder(self.feature_dims[feature], self.gpt_dim))

        self.reduce_cat = nn.Linear(self.gpt_dim + self.keyword_num, self.gpt_dim)
        self.reduce_c = nn.Linear(self.gpt_dim, self.dim)

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.visual_dropout = nn.Dropout(self.visual_dropout_ratio)

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

    def get_embedding(self, name, device):
        x = torch.LongTensor([getattr(self.tokenizer, name)]).to(device)
        x = x.unsqueeze(0)
        x = self.net.transformer.word_embedding(x)
        return x.squeeze(0)

    def run_transformer(self, B, hypo, features, keyword, infer=False):
        h, inputs = transformer_embed(self.net.transformer, hypo,
                                      skip_ids=[self.tokenizer.pad_id, self.tokenizer.sep_id],
                                      infer=infer)
        h = self.add_keyword(h, keyword)

        cls_embd = self.get_embedding('cls_id', h.device)
        sep_embd = self.get_embedding('sep_id', h.device)
        B, L, C = h.shape
        cls_embd = cls_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        sep_embd = sep_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        context = self.merge_context(features, cls_embd, sep_embd)

        o, context_embedded = transformer_run_cells(self.net.transformer, context, h, hypo=hypo,
                                                    pad_id=self.tokenizer.pad_id, **inputs)
        o = self.dropout(o)
        context_embedded = self.visual_dropout(context_embedded)

        return o, context_embedded

    def run_transformer_get_loss(self, hypo, features, keyword, group_mask=None, gt=None):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        res = OrderedDict()
        for feature in self.feature_names:
            res[feature] = getattr(self, feature)(features[feature])
        o, _ = self.run_transformer(hypo, res, keyword)

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
        eos_flags = torch.LongTensor([0] * B).bool().to(video.device)
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
                    s, probs = sampler(logits, alpha=self.normalizer_alpha)
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

    def forward(self, feature, h=None):
        # BLC
        feature = self.linear(feature)
        feature = F.leaky_relu(feature)
        return feature

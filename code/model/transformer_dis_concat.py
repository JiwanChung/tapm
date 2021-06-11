from collections import OrderedDict
from itertools import chain
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from exp import ex

from data.batcher import pad, pad_tensor
from tensor_utils import find_first
from utils import mean
from run_transformer import transformer_embed, transformer_run_cells
from .hybrid_dis import HybridDis
from debug_utils import timeit
from .transformer_dis import TransformerDis, FeatureEncoder


class TransformerDisConcat(TransformerDis):
    model_type = 'caption_single'

    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, num_samples):
        super().__init__(transformer, tokenizer, dropout_before)

        self.max_target_len = math.ceil(self.max_target_len * 3)  # mean group size
        self.num_samples = num_samples

    @classmethod
    def get_args(cls):
        return {
            **super().get_args(),
            'concat_group': True,
        }

    def process_features(self, features):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        res = OrderedDict()
        for feature in self.feature_names:
            res[feature] = getattr(self, feature)(features[feature])
        return res

    def forward(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    def _forward(self, batch, **kwargs):
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.feature_names}.items()}
        sample_feature = features[list(features.keys())[0]]

        features = self.process_features(features)
        context_infos = self.prepare_context(features)

        B = sample_feature.shape[0]  # (BG)
        device = sample_feature.device
        empty = torch.full((B, self.num_samples, self.vocab_size), float('-inf')).to(device)
        s0 = torch.Tensor([self.tokenizer.cls_id]).long().to(device).expand(B)
        s0 = s0.unsqueeze(-1)
        s = s0
        hypo = s0.unsqueeze(-1)
        eos_flags = torch.LongTensor([0] * B).bool().to(device).unsqueeze(1).expand(B, self.num_samples)

        sampler = kwargs.get('sampler', None)
        reduce_hypo = kwargs.get('reduce_hypo', True)
        # hypo = None
        logit = None
        if sampler is None:  # training
            hypo = batch.sentences
            hypo = rearrange(hypo.contiguous(), 'b g l -> (b g) l')
            logit, reg_loss, stats = self.run_token(hypo, context_infos, batch.lengths, batch.group_mask)
            logit = logit.unsqueeze(1)  # add group_dim
        else:
            reg_loss = None,
            stats = {}
            for i, w in enumerate(range(self.max_target_len)):
                if eos_flags.all():
                    logits = empty.clone()
                else:
                    logits = []
                    for n in range(hypo.shape[1]):
                        logit, _, _ = self.run_token(hypo[:, n], context_infos, None, batch.group_mask)
                        logits.append(logit[:, -1])  # get last token
                    logits = torch.stack(logits, dim=1)

                    s, probs, hypo_keys = sampler(logits, self.normalizer_alpha)  # BN, BN, BN
                    if i == 0:  # first iter
                        hypo = hypo.repeat(1, s.shape[1], 1)
                    hypo_keys = hypo_keys.unsqueeze(-1).expand(hypo.shape)
                    hypo = hypo.gather(dim=1, index=hypo_keys)
                    hypo = torch.cat((hypo, s.unsqueeze(-1)), dim=-1)
                    eos_flags = eos_flags | (logits.argmax(dim=-1) == self.tokenizer.pad_id)
            hypo = hypo[:, :, 1:]  # remove sos
            if reduce_hypo:
                best_ids = probs.argmax(dim=-1)
                hypo = hypo.gather(1, best_ids.contiguous().view(-1, 1, 1).repeat(1, 1, hypo.shape[-1]))
                hypo = hypo.squeeze(1)  # reduce single sample dim

            hypo = hypo.unsqueeze(1)  # match dim for eval code
            logit = logits

        if hasattr(batch, 'sentences'):
            stats = {**stats, 'sentence_len': (batch.sentences != self.tokenizer.pad_id).float().sum(dim=-1).mean().item()}
            del batch.sentences
        targets = batch.targets

        return hypo, logit, targets, reg_loss, stats, batch

    def slice_mean_group_text(self, h, lengths):
        # BLC, BG
        eps = 1e-10
        G = lengths.shape[1]
        h = h[:, 1:]  # drop cls
        lengths = lengths.cumsum(dim=-1)  # lengths to ids
        starts = lengths.roll(1, dims=-1)
        starts[:, 0] = 0
        ids = torch.stack((starts, lengths), dim=1)  # B2G
        h = h.unsqueeze(-2).repeat(1, 1, G, 1)  # BLGC
        mask = torch.zeros(*h.shape[:-1], dtype=torch.long).to(h.device)  # BLG
        mask.scatter_(dim=1, index=ids, value=1)
        mask = mask.cumsum(dim=1)  # 00...111...2...
        mask = mask == 1
        mask = mask.unsqueeze(-1).float() + eps  # avoid NaN
        means = (h * mask).sum(dim=1) / mask.sum(dim=1)
        # BGC

        return means

    def slice_group_context(self, c, shape, keys):
        # B (GL) C
        B, G, Ls = shape
        c = rearrange(c.contiguous(), 'b (g l) c -> b g l c', g=G)
        c = torch.split(c, [*[i + 1 for i in Ls], 1], dim=-2)[:-1]
        c = [t[:, :, :-1] for t in c]  #
        c = {k: v for k, v in zip(keys, c)}
        return c

    def get_reg_loss(self, h, context, group_mask):
        return None, {}

    def run_token(self, hypo, context_infos, lengths, group_mask):
        context_orig, context_shape, context_keys = context_infos
        h_in, past, head_mask = self.prepare_transformer_input(hypo, context_orig, lengths, group_mask)
        h, context = self.run_transformer(h_in, past, head_mask, context_orig)
        if lengths is not None:  # training
            h_mean = self.slice_mean_group_text(h, lengths)
            context_orig_cut = self.slice_group_context(context_orig, context_shape, context_keys)
            reg_loss, stats = self.get_reg_loss(h_mean, context_orig_cut, group_mask)
        else:
            reg_loss = None
            stats = {}
        logits, _, _ = self.get_logits(h, None)
        logits = logits[:, 1:]  # remove context_sep_token

        return logits, reg_loss, stats

    def prepare_transformer_input(self, hypo, context, lengths, group_mask):
        # BGL, B(GL)C, BGK
        B = hypo.shape[0]
        h, res  = transformer_embed(self.net.transformer, hypo)
        past = res['past']
        head_mask = res['head_mask']
        # B (G * L + (G - 1))
        context_sep_embd = self.net.transformer.wte(torch.LongTensor([self.tokenizer.context_sep_id]).to(h.device))
        context_sep_embd = context_sep_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        h = torch.cat((context, context_sep_embd, h), dim=1)

        return h, past, head_mask

    def run_transformer(self, h, past, head_mask, context):
        if self.dropout_before:
            h = self.dropout(h)
        o = transformer_run_cells(self.net.transformer, context, h, past=past, head_mask=head_mask)[0]
        context_embedded = o[:, :context.shape[1]]
        o = o[:, context.shape[1]:]
        if not self.dropout_before:
            o = self.dropout(o)
        return o, context_embedded

    def prepare_context(self, features):
        sample_feature = features[list(features.keys())[0]]
        B, G = sample_feature.shape[:2]
        sep_embd = self.net.transformer.wte(torch.LongTensor([self.tokenizer.sep_id]).to(sample_feature.device))
        seq_sep_embd = self.net.transformer.wte(torch.LongTensor([self.tokenizer.seq_sep_id]).to(sample_feature.device))
        sep_embd = sep_embd.view(1, 1, 1, -1).contiguous().expand(B, G, 1, -1)
        seq_sep_embd = seq_sep_embd.view(1, 1, 1, -1).contiguous().expand(B, G, 1, -1)

        context, context_shape, context_keys = self.merge_context(features, sep_embd, seq_sep_embd)  # B (GL) C

        return context, context_shape, context_keys

    def merge_context(self, features, sep_embd, seq_sep_embd):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        keys = list(features.keys())
        features = list(features.values())
        lengths = [feature.shape[-2] for feature in features]
        features = torch.cat((*list(chain(*[(feature, sep_embd) for feature in features])), seq_sep_embd), dim=-2)
        B, G = features.shape[:2]
        # BGLC
        features = rearrange(features.contiguous(), 'b g l c -> b (g l) c').contiguous()
        # B(GL)C
        return features, (B, G, lengths), keys

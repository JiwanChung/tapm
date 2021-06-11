import copy
import math
import random

from scipy.special import erfinv

import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import make_bert_batch

from tensor_utils import onehot
from .transformer_model import TransformerModel
from .modules import (
    IdentityModule,
    BinaryLayer, saturating_sigmoid,
    LSTMDecoder,
    l_n_norm
)


class DeterministicMasking(TransformerModel):
    transformer_name = 'bert'

    def __init__(self, args, transformer, tokenizer):
        super(DeterministicMasking, self).__init__()

        self.mean = 0  # fix mean
        self.binarize_mask = args.get('binarize_mask', True)
        self.num_keywords = args.get('num_keywords', 1000)
        self.keyword_loss_ema = args.get('keyword_loss_ema', False)

        self.tokenizer = tokenizer

        self.encoder = transformer
        self.encoder.train()

        embedding = self.encoder.bert.embeddings.word_embeddings
        self.decoder = LSTMDecoder(embedding)

        bert_dim = self.encoder.bert.config.hidden_size
        self.mean_encoder = nn.Linear(bert_dim, 1)

        self.straight_through = BinaryLayer()

        self.norm_n = args.get('loss_norm_n', 0)
        self.V = len(tokenizer)
        keyword_vector = torch.cat((torch.ones(self.num_keywords),
                                    torch.zeros(self.V - self.num_keywords)), dim=0).float()
        self.keyword_loss_threshold = l_n_norm(keyword_vector, n=self.norm_n)
        if self.keyword_loss_ema:
            self.ema_alpha = args.get('keyword_ema_alpha', 0.001)
            self.keywords_count = nn.Parameter(keyword_vector.clone())
            self.keywords_count.requires_grad_(False)
            self.keywords_count.fill_(0)

    def onehot(self, x):
        return onehot(x, self.V)

    @staticmethod
    def make_batch(*args, **kwargs):
        return make_bert_batch(*args, **kwargs)

    def get_keyword_loss(self, mask, sentence, batch_per_epoch=1):
        mask = mask.unsqueeze(-1) * self.onehot(sentence)  # BL BLV
        mask = mask.sum(dim=0).sum(dim=0)  # V
        mask = torch.max(mask, torch.zeros(1).to(mask.device))
        if self.keyword_loss_ema:
            keywords_count = mask + self.keywords_count
            loss = l_n_norm(keywords_count, n=self.norm_n)
            loss = loss - self.keyword_loss_threshold
            loss = torch.max(loss, torch.zeros(1).to(loss.device))
            self.keywords_count.data = self.ema_alpha * keywords_count.detach() \
                + (1 - self.ema_alpha) * self.keywords_count
            return loss, l_n_norm(mask, n=self.norm_n).detach()
        else:
            loss = l_n_norm(mask, n=self.norm_n) - self.keyword_loss_threshold / batch_per_epoch
            # loss = l_n_norm(mask, n=2)

            loss = torch.max(loss, torch.zeros(1).to(mask.device))
            loss = loss / (sentence.shape[0] * sentence.shape[1])  # mean for token num
            return loss, None

    def mask_encoder(self, x):
        return self.mean_encoder(x).squeeze(-1)

    def forward(self, batch, batch_per_epoch=1,**kwargs):
        sentence = batch.sentences
        targets = batch.targets
        targets_orig = targets
        # make special token mask
        cls_mask = sentence != self.tokenizer.cls_id
        sep_mask = sentence != self.tokenizer.sep_id
        attention_mask = sentence != self.tokenizer.pad_id
        special_masks = [cls_mask, sep_mask, attention_mask]
        specials_mask = special_masks[0]
        for mask in special_masks:
            specials_mask = mask * specials_mask

        # encoder
        x_embed = self.encoder.bert.embeddings(sentence)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(
            self.encoder.bert.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.encoder.bert.config.num_hidden_layers
        x_feat = self.encoder.bert.encoder(x_embed, extended_attention_mask,
                                           head_mask=head_mask)[0]

        # bottleneck
        z = self.mask_encoder(x_feat)
        m = saturating_sigmoid(z)
        keyword_loss, keywords_count = self.get_keyword_loss(m, sentence, batch_per_epoch)
        if self.binarize_mask:
            m = self.straight_through(m - 0.5)
        keywords_mask = m > 0

        # decoder
        m = m * specials_mask.float()  # remove sep token
        keyword_ratio = keywords_mask.masked_select(specials_mask).float()
        x = m.unsqueeze(-1) * x_embed
        x = x.mean(dim=1)

        shifted_targets = targets.clone()
        shifted_targets[shifted_targets == self.tokenizer.sep_id] = \
            self.tokenizer.pad_id
        shifted_targets = shifted_targets[:, :-1]
        targets = targets[:, 1:]
        logits = self.decoder(x, shifted_targets)

        with torch.no_grad():
            stats = {
                'keyword_ratio': keyword_ratio.mean().item(),
            }

            keywords = []
            scores = []
            all_mask = keywords_mask * specials_mask  # ditch cls, sep, pad
            keywords_unsorted = [targets_orig[i][all_mask[i]] for i in range(sentence.shape[0])]
            scores_unsorted = [m[i][all_mask[i]] for i in range(sentence.shape[0])]
            for keyword, score in zip(keywords_unsorted, scores_unsorted):
                score, keyword_idx = score.sort(dim=-1, descending=True)
                keyword = keyword.gather(dim=0, index=keyword_idx)
                keywords.append(keyword)
                scores.append(score)

            if self.keyword_loss_ema:
                total_keywords_count = l_n_norm(self.keywords_count, n=self.norm_n).item()
                stats = {**stats,
                    'keywords_count': keywords_count.item(),
                    'total_keywords_count': total_keywords_count
                }

        return logits, targets, keyword_loss, \
            stats, keywords

import math

import torch
from torch import nn
import torch.nn.functional as F

from tensor_utils import unsqueeze_expand


class Encoder(nn.Module):
    def __init__(self, args, transformer, tokenizer):
        super(Encoder, self).__init__()

        self.threshold = args.keyword_threshold
        self.gap = args.threshold_gap
        self.aggregate = {
            'mean': lambda x, dim: x.mean(dim=dim),
            'max': lambda x, dim: x.max(dim=dim)[0],
        }[args.transformer_pool.lower()]

        self.net = transformer
        self.embed = self.net.transformer.wte
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad_id

    def forward(self, sentence, lengths):
        # remove first sep
        sentence = sentence[:, 1:]
        lengths = lengths - 1

        B = sentence.shape[0]
        h, _, = self.net.transformer(sentence)

        h = self.aggregate(h, -1)
        scores = torch.sigmoid(h).clone()
        for i in range(B):
            scores[i, lengths[i]:] = 0  # masking pads
        threshold = self.get_threshold(lengths)
        score_regularization_loss = self.get_score_loss(scores, threshold)

        # sort by score and threshold
        sorted_idx = scores.argsort(dim=-1, descending=True)
        # clumsy implementation of length-wise thresholding
        threshold_idx = sorted_idx
        scores = self.gather_with_idx(threshold_idx, scores)
        for i in range(B):
            scores[i, threshold[i] + self.gap:] = 0
        keywords = sentence.clone()
        keywords = self.gather_with_idx(threshold_idx, keywords)
        for i in range(B):
            keywords[i, threshold[i] + self.gap:] = self.pad_id

        # cut 0 indices
        keyword_lengths = (keywords != self.pad_id).sum(-1)
        max_keyword_length = keyword_lengths.max()
        scores = scores[:, :max_keyword_length]
        keywords = keywords[:, :max_keyword_length]

        return keywords, keyword_lengths, scores, score_regularization_loss

    def gather_with_idx(self, idx, t):
        gather_dim = len(idx.shape) - 1
        idx = unsqueeze_expand(idx, t)
        t = torch.gather(t, gather_dim, idx)
        return t * (idx != 0).type_as(t)

    def get_score_loss(self, scores, threshold):
        # get root(l-1/2-norm)
        # scores >= 0
        # the intuition of this loss is to make the score sparse,
        # while getting 0 for num == threshold
        loss = (scores ** (1 / 2)).sum(dim=-1)
        loss = ((loss / threshold.float()) - 1)
        return loss

    def get_threshold(self, lengths):
        if self.threshold >= 1:
            # length threshold num
            return torch.empty_like(lengths).fill_(math.floor(self.threshold)).long()
        else:
            # length threshold ratio
            return (lengths.float() * self.threshold).floor().long()

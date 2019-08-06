import copy
import math
import random

from scipy.special import erfinv

import torch
from torch import nn
import torch.nn.functional as F

from .modules import IdentityModule


class VariationalMasking(nn.Module):
    transformer_name = 'bert'

    def __init__(self, args, transformer, tokenizer):
        super(VariationalMasking, self).__init__()

        self.masking_prob = 1 - args.get('keyword_ratio', 1 / 2)
        self.hard_masking_prob = args.get('hard_masking_prob', 1 / 2)
        self.mean = self.get_mean(self.masking_prob)

        self.pad_id = tokenizer.pad_id

        self.encoder = transformer
        if hasattr(args, 'share_encoder_decoder') and args.share_encoder_decoder:
            self.decoder = self.encoder
        else:
            self.decoder = copy.deepcopy(self.encoder)

        # skip decoder embedding module
        self.decoder.bert.embeddings = IdentityModule()
        self.encoder.train()
        self.decoder.train()

        bert_dim = self.encoder.bert.config.hidden_size
        self.mean_encoder = nn.Linear(bert_dim, 1)
        self.logvar_encoder = nn.Linear(bert_dim, 1)

    @staticmethod
    def get_mean(masking_prob):
        # intuition: a token will be masked with probability p
        # using inverse cdf function of gaussian
        return math.sqrt(2) * erfinv(2 * masking_prob - 1)

    def gaussian_encoder(self, x):
        return self.mean_encoder(x).squeeze(-1), self.logvar_encoder(x).squeeze(-1)

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(std)
        else:
            # return mean for eval
            return mean

    def get_hard_mask(self):
        if not self.training:
            return True
        else:
            return random.random() <= self.hard_masking_prob

    @staticmethod
    def kl_div(mu, logvar):
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, sentence, lengths, targets):
        attention_mask = sentence != self.pad_id
        outputs = self.encoder.bert(sentence, attention_mask=attention_mask)
        # ignore lm head
        a = outputs[0]
        mean, logvar = self.gaussian_encoder(a)
        z = self.reparameterize(mean, logvar)
        z = torch.sigmoid(z)
        m = z - 1 / 2  # mean to 0
        keyword_ratio = (m >= 0).float().mean().item()
        if self.get_hard_mask():
            m = F.relu(m)
        x = m.unsqueeze(-1) * a
        outputs = self.decoder(x, attention_mask=attention_mask)
        logits = outputs[0]

        kl_loss = self.kl_div(mean, logvar)
        stats = {
            'keyword_ratio': keyword_ratio,
            'kl_loss': kl_loss.mean().item()
        }
        masks = (m >= 0)
        keywords = [targets[i][masks[i]] for i in range(targets.shape[0])]

        return logits, targets, kl_loss.sum(), stats, keywords

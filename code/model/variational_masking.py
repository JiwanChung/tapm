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

        self.keyword_ratio = args.get('keyword_ratio', 1 / 2)
        self.hard_masking_prob = args.get('hard_masking_prob', 1 / 2)
        self.mean = self.get_mean(self.keyword_ratio)
        self.std = args.get('latent_std', 1)

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
    def get_mean(keyword_ratio):
        # intuition: a token will be masked with probability p
        # using inverse cdf function of gaussian
        return math.sqrt(2) * erfinv(2 * keyword_ratio - 1)

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

    def kl_div(self, mu, logvar):
        logvar = logvar - 2 * math.log(self.std)
        return -0.5 * (1 + logvar - (mu - self.mean).pow(2) - logvar.exp())

    def forward(self, sentence, lengths, targets):
        attention_mask = sentence != self.pad_id
        outputs = self.encoder.bert(sentence, attention_mask=attention_mask)
        # ignore lm head
        a = outputs[0]
        mean, logvar = self.gaussian_encoder(a)
        z = self.reparameterize(mean, logvar)
        z = torch.sigmoid(z)
        m = z - 1 / 2  # mean to 0
        if self.get_hard_mask():
            m = F.relu(m)
        m[:, 0] = 0  # remove cls token
        m[:, -1] = 0  # remove sep token
        masks = (m > 0)  # true for none-masked tokens
        keyword_ratio = masks.float().mean().item()
        x = m.unsqueeze(-1) * a
        outputs = self.decoder(x, attention_mask=attention_mask)
        logits = outputs[0]

        kl_loss = self.kl_div(mean[:, 1:-1], logvar[:, 1:-1])
        # remove cls, sep token

        stats = {
            'keyword_ratio': keyword_ratio,
            'gt_mu': self.mean,
            'mu': mean.mean().item(),
            'std': logvar.exp().sqrt().mean().item(),
            'kl_loss': kl_loss.mean().item(),
        }
        keywords = [targets[i][masks[i]] for i in range(targets.shape[0])]

        # get loss for masked tokens only
        # remove cls, sep token
        masks_cut = masks[:, 1:-1]
        logits_cut = logits[:, 1:-1]
        targets_cut = targets[:, 1:-1]
        masked_logits = logits_cut.contiguous().masked_select((~masks_cut).contiguous().unsqueeze(-1))
        masked_logits = masked_logits.contiguous().view(-1, logits_cut.shape[-1])
        masked_targets = targets_cut.contiguous().masked_select(~masks_cut).contiguous()

        return masked_logits, masked_targets, kl_loss.mean(), \
            stats, {'keywords': keywords, 'targets': targets}

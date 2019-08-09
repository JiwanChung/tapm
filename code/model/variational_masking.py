import copy
import math
import random

from scipy.special import erfinv

import torch
from torch import nn
import torch.nn.functional as F

from .transformer_model import TransformerModel
from .modules import IdentityModule


class VariationalMasking(TransformerModel):
    transformer_name = 'bert'

    def __init__(self, args, transformer, tokenizer):
        super(VariationalMasking, self).__init__()

        self.keyword_ratio = args.get('keyword_ratio', 1 / 2)
        self.threshold = 1 - self.keyword_ratio
        self.hard_masking_prob = args.get('hard_masking_prob', 1 / 2)
        # self.mean = self.get_mean(self.keyword_ratio)
        self.mean = 0  # fix mean
        self.std = args.get('latent_std', 1)

        self.cls_id = tokenizer.cls_id
        self.sep_id = tokenizer.sep_id
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
        # using inverse cdf function of a normal distribution
        return math.sqrt(2) * erfinv(2 * keyword_ratio - 1)

    def gaussian_encoder(self, x):
        '''
        return self.mean_encoder(x).squeeze(-1).tanh(), \
            (self.logvar_encoder(x).squeeze(-1).sigmoid() \
             * (-4 * math.log(1 / self.std)))
        # tanh, log serves to stabilize training
        '''
        return self.mean_encoder(x).squeeze(-1), \
            self.logvar_encoder(x).squeeze(-1)

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(std)
        else:
            # return mean for eval
            return mean

    def get_hard_mask(self):
        if not self.training:
            return False
        else:
            return random.random() <= self.hard_masking_prob

    def kl_div(self, mu, logvar, all_mask):
        # mu = mu.mean()  # match moment only
        logvar = logvar - 2 * math.log(self.std)
        kl = -0.5 * (1 + logvar - (mu - self.mean).pow(2) - logvar.exp())

        kl = kl.contiguous().masked_select(all_mask)
        return kl.mean()

    def forward(self, sentence, lengths, targets):
        cls_mask = sentence != self.cls_id
        sep_mask = sentence != self.sep_id
        attention_mask = sentence != self.pad_id
        masks = [cls_mask, sep_mask, attention_mask]
        all_mask = masks[0]
        for mask in masks:
            all_mask = mask * all_mask
        outputs = self.encoder.bert(sentence, attention_mask=attention_mask)
        # ignore lm head
        a = outputs[0]
        mean, logvar = self.gaussian_encoder(a)
        z = self.reparameterize(mean, logvar)
        z = torch.sigmoid(z)
        m = 10 * (z - 1 / 2)  # scale to 0 ~ 1
        mean_masks = (10 * (mean.sigmoid() - 1 / 2) > self.threshold)
        if self.get_hard_mask():
            m = F.relu(m - self.threshold) + self.threshold
            masks = (m > 0)  # true for none-masked tokens
        else:
            masks = (m > self.threshold)  # true for none-masked tokens

        m = m * all_mask.float()  # remove sep token
        keyword_ratio = masks.masked_select(all_mask).float().mean().item()
        mean_keyword_ratio = mean_masks.masked_select(all_mask).float().mean().item()
        x = m.unsqueeze(-1) * a
        outputs = self.decoder(x, attention_mask=attention_mask)
        logits = outputs[0]

        kl_loss = self.kl_div(mean, logvar, all_mask)

        stats = {
            'keyword_ratio': keyword_ratio,
            'mean_keyword_ratio': mean_keyword_ratio,
            'mu': mean.mean().item(),
            'std': logvar.exp().sqrt().mean().item(),
            'kl_loss': kl_loss.item(),
        }

        keywords = []
        scores = []
        masks = masks * all_mask  # ditch cls, sep, pad
        keywords_unsorted = [targets[i][masks[i]] for i in range(targets.shape[0])]
        scores_unsorted = [m[i][masks[i]] for i in range(targets.shape[0])]
        for keyword, score in zip(keywords_unsorted, scores_unsorted):
            score, keyword_idx = score.sort(dim=-1, descending=True)
            keyword = keyword.gather(dim=0, index=keyword_idx)
            keywords.append(keyword)
            scores.append(score)

        # get loss for masked tokens only
        # remove cls, sep token
        loss_masks = ~masks
        for mask in [cls_mask, sep_mask, attention_mask]:
            loss_masks = mask * loss_masks

        masked_logits = logits.contiguous().masked_select(loss_masks.contiguous().unsqueeze(-1))
        masked_logits = masked_logits.contiguous().view(-1, logits.shape[-1])
        masked_targets = targets.contiguous().masked_select(loss_masks).contiguous()

        stats = {**stats,
                 'loss_words': masked_targets.nelement()}

        return masked_logits, masked_targets, kl_loss, \
            stats, {'keywords': keywords, 'targets': targets}

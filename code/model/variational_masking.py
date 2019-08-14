import copy
import math
import random

from scipy.special import erfinv

import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import make_bert_batch

from .transformer_model import TransformerModel
from .modules import (
    IdentityModule,
    BinaryLayer, saturating_sigmoid,
    LSTMDecoder
)


class VariationalMasking(TransformerModel):
    transformer_name = 'bert'

    def __init__(self, args, transformer, tokenizer):
        super(VariationalMasking, self).__init__()

        self.threshold = 1 - args.get('keyword_ratio', 1 / 2)  # warning: self.threshold cannot be directly translated into keyword_ratio
        self.mean = 0  # fix mean
        self.std = args.get('latent_std', 1)
        self.binarize_mask = args.get('binarize_mask', False)
        self.delay_kl_loss = args.get('delay_kl_loss', 0)
        self.step = 0

        self.tokenizer = tokenizer

        self.encoder = transformer
        '''
        if hasattr(args, 'share_encoder_decoder') and args.share_encoder_decoder:
            self.decoder = self.encoder
        else:
            self.decoder = copy.deepcopy(self.encoder)
            # tie weight
            self.decoder.cls.predictions.decoder.weight = \
                self.encoder.bert.embeddings.word_embeddings.weight

        # skip decoder embedding module
        self.decoder.bert.embeddings = IdentityModule()
        self.decoder.train()
        '''
        self.encoder.train()

        self.decoder = LSTMDecoder(
            self.encoder.bert.embeddings.word_embeddings)

        bert_dim = self.encoder.bert.config.hidden_size
        self.mean_encoder = nn.Linear(bert_dim, 1)
        self.logvar_encoder = nn.Linear(bert_dim, 1)

        if self.binarize_mask:
            self.straight_through = BinaryLayer()

    @staticmethod
    def make_batch(*args, **kwargs):
        return make_bert_batch(*args, **kwargs)

    '''
    @staticmethod
    def get_mean(keyword_ratio):
        # intuition: a token will be masked with probability p
        # using inverse cdf function of a normal distribution
        return math.sqrt(2) * erfinv(2 * keyword_ratio - 1)
    '''

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

    def kl_div(self, mu, logvar, specials_mask):
        # mu = mu.mean()  # match moment only
        logvar = logvar - 2 * math.log(self.std)
        kl = -0.5 * (1 + logvar - (mu - self.mean).pow(2) - logvar.exp())

        kl = kl.contiguous().masked_select(specials_mask)
        return kl.mean()

    def forward(self, batch, **kwargs):
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
        a = x_feat
        mean, logvar = self.gaussian_encoder(a)
        z = self.reparameterize(mean, logvar)
        m = saturating_sigmoid(z)
        if self.binarize_mask:
            m = self.straight_through(m - self.threshold)
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
        '''
        outputs = self.decoder(x, attention_mask=attention_mask)
        logits = outputs[0]
        '''

        if self.delay_kl_loss > self.step:
            mean = mean.detach()
            logvar = logvar.detach()
        kl_loss = self.kl_div(mean, logvar, specials_mask)

        '''
        # get loss for masked tokens only
        # remove cls, sep token
        loss_mask = (~keywords_mask) * specials_mask

        masked_logits = logits.contiguous().masked_select(loss_mask.contiguous().unsqueeze(-1))
        masked_logits = masked_logits.contiguous().view(-1, logits.shape[-1])
        masked_targets = targets.contiguous().masked_select(loss_mask).contiguous()
        targets = masked_targets
        logits = masked_logits
        '''

        with torch.no_grad():
            stats = {
                'keyword_ratio': keyword_ratio.mean().item(),
                'mu': mean.mean().item(),
                'std': logvar.exp().sqrt().mean().item(),
                'kl_loss': kl_loss.item(),
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

        if self.training:
            self.step += 1

        return logits, targets, kl_loss, \
            stats, keywords

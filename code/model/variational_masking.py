import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


class VariationalMasking(nn.Module):
    transformer_name = 'bert'

    def __init__(self, args, transformer, tokenizer):
        super(VariationalMasking, self).__init__()

        self.masking_prob = args.get('keyword_ratio', 1 / 2)
        self.mean = self.get_mean(self.masking_prob)

        self.encoder = transformer
        if hasattr(args, 'share_encoder_decoder') and args.share_encoder_decoder:
            self.decoder = self.encoder
        else:
            self.decoder = copy.deepcopy(self.encoder)
        self.encoder.train()
        self.decoder.train()

    @staticmethod
    def get_mean(masking_prob):
        p = masking_prob
        return math.log(p / (1 - p))

    def reparameterize(self, mean, std):
        if self.training:
            return mean + std * torch.randn_like(std)
        else:
            # return mean for eval
            return mean

    def forward(self, sentence, lengths):
        attention_mask = sentence != self.pad_id
        outputs = self.encoder(sentence, attention_mask=attention_mask)
        a = outputs[0]
        mean, std = self.gaussian_encoder(a)
        z = self.reparameterize(mean, std)
        z = torch.sigmoid(z)
        m = z
        if hard_mask:
            m = F.relu(m, dim=-1)
        x = m * a
        # TODO: skip embedding
        outputs = self.decoder(x, attention_mask=attention_mask)
        logits = outputs[0]

        return

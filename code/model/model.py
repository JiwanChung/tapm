import torch
from torch import nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder


class Model(nn.Module):
    def __init__(self, args, transformers, tokenizer):
        super(Model, self).__init__()

        self.encoder = Encoder(args, transformers['encoder'], tokenizer)
        self.decoder = Decoder(args, transformers['decoder'], tokenizer)

    def forward(self, sentence, lengths):
        keywords, keyword_lengths, scores, reg_loss = \
            self.encoder(sentence, lengths)
        logits = self.decoder(sentence, keywords,
                              keyword_lengths, scores)
        return logits, reg_loss, scores, keywords

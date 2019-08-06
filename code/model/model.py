import torch
from torch import nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder


class Model(nn.Module):
    transformer_name = 'gpt2'

    def __init__(self, args, transformers, tokenizer):
        super(Model, self).__init__()

        self.use_keyword = args.use_keyword

        if self.use_keyword:
            self.encoder = Encoder(args, transformers['encoder'], tokenizer)
        self.decoder = Decoder(args, transformers['decoder'], tokenizer)

    def forward(self, sentence, lengths, targets):
        if self.use_keyword:
            keywords, keyword_lengths, scores, reg_loss = \
                self.encoder(sentence, lengths)
        else:
            keywords, keyword_lengths, scores, reg_loss = None, None, None, None
        logits = self.decoder(sentence, lengths,
                              keywords, keyword_lengths, scores)
        return logits, targets, reg_loss, scores, keywords

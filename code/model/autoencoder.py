import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import make_autoencoder_batch

from .transformer_model import TransformerModel
from .encoder import Encoder
from .decoder import Decoder


class Autoencoder(TransformerModel):
    transformer_name = 'gpt2'

    def __init__(self, args, transformers, tokenizer):
        super(Autoencoder, self).__init__()

        self.use_keyword = args.use_keyword

        if self.use_keyword:
            self.encoder = Encoder(args, transformers['encoder'], tokenizer)
        self.decoder = Decoder(args, transformers['decoder'], tokenizer)

    def make_batch(self, *args, **kwargs):
        return make_autoencoder_batch(*args, **kwargs)
    def forward(self, sentence, lengths, targets):
        if self.use_keyword:
            keywords, keyword_lengths, scores, reg_loss = \
                self.encoder(sentence, lengths)
        else:
            keywords, keyword_lengths, scores, reg_loss = None, None, None, None
        logits = self.decoder(sentence, lengths,
                              keywords, keyword_lengths, scores)
        return logits, targets, reg_loss, {'prob': scores.mean().item()}, keywords

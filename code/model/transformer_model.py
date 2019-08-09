from torch import nn
from transformers import get_transformer

from data.batcher import make_bert_batch


class TransformerModel(nn.Module):
    def make_batch(self, *args, **kwargs):
        return make_bert_batch(*args, **kwargs)

    @classmethod
    def build(cls, args):
        transformer, tokenizer = get_transformer(cls.transformer_name)
        return cls(args, transformer, tokenizer), tokenizer

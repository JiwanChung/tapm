from torch import nn
from transformers import get_transformer

from data.batcher import make_bert_batch
from data.tokenizer import build_tokenizer


class TransformerModel(nn.Module):
    use_keyword = True

    def make_batch(self, *args, **kwargs):
        return make_bert_batch(*args, **kwargs)

    @classmethod
    def get_args(cls, args):
        return args

    @classmethod
    def build(cls, args, data):
        transformer, tokenizer = get_transformer(cls.transformer_name)
        if tokenizer is None:
            tokenizer = build_tokenizer(args, data)
        return cls(args, transformer, tokenizer), tokenizer

from torch import nn
from get_transformer import get_transformer

from data.batcher import make_bert_batch
from data.tokenizer import build_word_embedding, build_tokenizer


class TransformerModel(nn.Module):
    use_keyword = True

    def make_batch(self, *args, **kwargs):
        return make_bert_batch(*args, **kwargs)

    @classmethod
    def get_args(cls):
        return {}

    @classmethod
    def build(cls, data, transformer_name=None):
        transformer_name = cls.transformer_name if transformer_name is None \
            else transformer_name
        transformer, tokenizer = get_transformer(transformer_name)
        if tokenizer is None:
            tokenizer = build_tokenizer(data)
        tokenizer.embedding = build_word_embedding(tokenizer)
        return cls(transformer, tokenizer), tokenizer

from torch import nn
from transformers import get_transformer


class TransformerModel(nn.Module):
    @classmethod
    def build(cls, args):
        transformer, tokenizer = get_transformer(cls.transformer_name)
        return cls(args, transformer, tokenizer), tokenizer

import torch
from torch import nn

from data.batcher import make_keyword_batch

from .transformer_model import TransformerModel
from .nn_transformer import Transformer


class TransformerKeywordLM(TransformerModel):
    transformer_name = 'bert'
    update_h = False

    @classmethod
    def get_args(cls, args):
        args = super().get_args(args)
        args.eval_generate = True
        return args

    def __init__(self, args, transformer, tokenizer):
        super(TransformerKeywordLM, self).__init__()

        self.V = len(tokenizer)
        self.tokenizer = tokenizer

        self.dim = transformer.bert.config.hidden_size

        self.wte = nn.Embedding(self.V, self.dim)
        self.transformer = Transformer(self.dim)

    @staticmethod
    def make_batch(*args, **kwargs):
        return make_keyword_batch(*args, **kwargs, concat=False)

    def embedding(self, x):
        wte = self.wte(x)

        return wte

    def get_att_mask(self, x):
        return x == self.tokenizer.pad_id

    def out(self, x):
        return torch.matmul(x, self.wte.weight.t())

    def forward(self, batch, **kwargs):
        keywords = batch.keywords
        sentences = batch.sentences
        targets = batch.targets

        h, src_mask = self.encode(keywords)
        logits, _ = self.decode(sentences, h, src_mask)

        if self.training:
            keywords = None  # monkey-patch... refers to train.py and evaluate.py
        return logits, targets, None, None, keywords

    def encode(self, keywords):
        src = self.embedding(keywords)
        src_key_padding_mask = self.get_att_mask(keywords)
        memory = self.transformer.encoder(src.transpose(0, 1),
                                          src_key_padding_mask=src_key_padding_mask).transpose(0, 1)
        return memory, src_key_padding_mask

    def decode(self, tgt, memory, src_key_padding_mask):
        tgt_key_padding_mask = self.get_att_mask(tgt)
        tgt = self.embedding(tgt)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1])
        tgt_mask = tgt_mask.to(tgt.device)
        output = self.transformer.decoder(tgt.transpose(0, 1),
                        memory.transpose(0, 1),
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=src_key_padding_mask,
                        tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        return self.out(output), memory

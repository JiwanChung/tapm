import torch

from .transformer_dis import TransformerDis


class TransformerDis2(TransformerDis):
    def __init__(self, args, transformer, tokenizer):
        super(TransformerDis2, self).__init__(args, transformer, tokenizer)

        del self.reduce_cat

    def add_keyword(self, h, keyword):
        return h

    def get_logits(self, o, keyword):
        # BN * NV
        keyword = torch.matmul(keyword, self.keyword_map)
        o = self.net.lm_head(o)
        return o * keyword.unsqueeze(1).expand(-1, o.shape[1], -1)

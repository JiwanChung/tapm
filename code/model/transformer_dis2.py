import torch

from .transformer_dis import TransformerDis
from .keyword_classifier import KeywordClassifier


class TransformerDis2(TransformerDis):
    def __init__(self, args, transformer, tokenizer):
        super(TransformerDis2, self).__init__(args, transformer, tokenizer)

        del self.reduce_cat

        self.eps = 0.1

    def add_keyword(self, h, keyword):
        return h

    def get_logits(self, o, keyword):
        # BN * NV
        keyword = torch.matmul(keyword, self.keyword_map)
        keyword = self.smooth(keyword)
        o = self.net.lm_head(o)
        return o * keyword.unsqueeze(1).expand(-1, o.shape[1], -1)

    def smooth(self, probs):
        return probs + self.eps


class TransformerDis3(TransformerDis2):
    def __init__(self, args, transformer, tokenizer):
        super(TransformerDis3, self).__init__(args, transformer, tokenizer)

        del self.keyword_classifier

        self.eps = 0
        self.keyword_classifier = KeywordClassifier(
            len(tokenizer), self.dim, self.feature_names,
            self.video_dim, self.image_dim)

    def get_keyword(self, batch, features):
        return self.keyword_classifier(batch.word_subsets, features)

    def get_logits(self, o, keyword):
        keyword = self.smooth(keyword)
        o = self.net.lm_head(o)
        return o * keyword.unsqueeze(1).expand(-1, o.shape[1], -1)

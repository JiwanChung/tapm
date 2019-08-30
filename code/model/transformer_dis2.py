import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .transformer_dis import TransformerDis
from .keyword_classifier import KeywordClassifier
from .modules import MLP


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
        return o * keyword.unsqueeze(1).expand(-1, o.shape[1], -1), {}

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
        return o * keyword.unsqueeze(1).expand(-1, o.shape[1], -1), {}


class TransformerDisMoe(TransformerDis2):
    def __init__(self, args, transformer, tokenizer):
        super(TransformerDisMoe, self).__init__(args, transformer, tokenizer)

        self.moe_dim = self.gpt_dim // 16
        self.moe_in = nn.Linear(self.gpt_dim, self.keyword_num * self.moe_dim)
        self.moe_out = nn.Linear(self.keyword_num * self.moe_dim, self.gpt_dim)
        self.gate_mlp = MLP(self.gpt_dim)

    def moe(self, x, beta):
        # BLC -> BL(CN)
        x = self.moe_in(x)
        x = rearrange(x, 'b l (n c) -> b l n c', n=self.keyword_num)
        x = F.relu(x)
        x = x * beta.unsqueeze(-1)
        x = rearrange(x, 'b l n c -> b l (n c)')
        x = self.moe_out(x)
        return x

    def gate_keyword(self, keyword, o):
        # nc, blc
        keyword = self.gate_mlp(keyword)
        beta = torch.einsum('bnc,blc->bln', keyword, o.detach())
        beta = F.softmax(beta, dim=-1)
        return beta

    def get_logits(self, o, keyword):
        # VC * NV -> NC
        embds = torch.einsum('vc,nv->nc', self.net.transformer.wte.weight, self.keyword_map)
        keyword = keyword.unsqueeze(-1) * embds  # BNC
        beta = self.gate_keyword(keyword, o)  # bln
        o = self.moe(o, beta)  # blnc
        o = self.net.lm_head(o)
        return o, {}


class TransformerDisPtrGen(TransformerDis2):
    def __init__(self, args, transformer, tokenizer):
        super(TransformerDisPtrGen, self).__init__(args, transformer, tokenizer)

        self.k = 20
        self.inf = float('-inf')
        self.gate_mlp = MLP(self.gpt_dim)
        self.small_out = nn.Linear(self.gpt_dim, self.k)

    def gate_keyword(self, embds, o):
        # bkc, blc
        beta = torch.einsum('bkc,blc->blk', embds, o)
        beta = beta.mean(dim=-1)  # bl
        beta = torch.sigmoid(beta)
        return beta

    def get_small_logit(self, keyword_map, o):
        o = self.small_out(o)
        o = torch.einsum('bkv,blk->blv', keyword_map, o)
        return o

    def get_logits(self, o, keyword):
        keyword_top, keyword_top_ids = keyword.topk(k=self.k, dim=-1)
        keyword_map = self.keyword_map.unsqueeze(0).expand(keyword_top_ids.shape[0], -1, -1)
        keyword_map = keyword_map.gather(1, keyword_top_ids.unsqueeze(-1).expand(
                                             -1, -1, keyword_map.shape[-1]))
        # BKV
        embds = torch.einsum('vc,bkv->bkc', self.net.transformer.wte.weight, keyword_map)
        big_logit = self.net.lm_head(o)
        o = self.gate_mlp(o)
        small_logit = self.get_small_logit(keyword_map, o)
        beta = self.gate_keyword(embds, o)  # bl
        stats = {'copy_gate': beta.mean().item()}
        beta = beta.unsqueeze(-1)
        logits = (1 - beta) * big_logit + beta * small_logit
        return logits, stats

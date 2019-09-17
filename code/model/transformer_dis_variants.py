from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from tensor_utils import onehot
from loss import SmoothLoss

from transformers import transformer_embed
from .transformer_dis import TransformerDis
from .modules import MLP, SelfAttention
from .collaborative_experts import CollaborativeExpertsWrapper, calc_ranking_loss


'''
class TransformerDis2(TransformerDis):
    def __init__(self, args, transformer, tokenizer):
        super(TransformerDis2, self).__init__(args, transformer, tokenizer)

        del self.reduce_cat

        self.eps = 0.1

    def add_keyword(self, h, keyword):
        return h

    def get_logits(self, o, keyword, gt=None):
        # BN * NV
        if self.keyword_map is not None:
            keyword = torch.matmul(keyword, self.keyword_map)
        keyword = self.smooth(keyword)
        o = self.net.lm_head(o)
        return o * keyword.unsqueeze(1).expand(-1, o.shape[1], -1), None, {}

    def smooth(self, probs):
        return probs + self.eps
'''


class TransformerDisCE(TransformerDis):
    def __init__(self, args, transformer, tokenizer):
        super(TransformerDisCE, self).__init__(args, transformer, tokenizer)

        self.self_attention = SelfAttention(self.gpt_dim, heads=4)
        self.ce = CollaborativeExpertsWrapper(
            self.feature_names,
            self.video_dim,
            self.image_dim,
            self.flow_dim,
            self.gpt_dim
        )

    def run_transformer_get_loss(self, hypo, features, keyword, group_mask=None, gt=None):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        res = OrderedDict()
        for feature in self.feature_names:
            res[feature] = getattr(self, feature)(features[feature], None)
        o, _ = self.run_transformer(hypo, res, keyword)

        rank_loss, stats = self.get_ranking(hypo, features, group_mask)

        o = self.dropout(o)
        c = o.mean(dim=1)
        c = self.reduce_c(c)
        logits, _, _ = self.get_logits(o, keyword, gt)
        return logits, c, rank_loss, stats

    def get_ranking(self, hypo, features, group_mask):
        h, _, _ = transformer_embed(self.net.transformer, hypo)
        h = self.self_attention(h.detach())
        rank_loss, stats = self.ce(h, features, group_mask)

        return rank_loss, stats


# ranking loss for feature just before GPT2
class TransformerDisRank(TransformerDis):
    def __init__(self, args, transformer, tokenizer):
        super(TransformerDisRank, self).__init__(args, transformer, tokenizer)

        self.margin = args.get('margin', 1)

    def get_rank_loss(self, o, feature, group_mask):
        x1 = F.normalize(o.mean(dim=1))  # BC
        x2 = F.normalize(feature)  # BC

        loss1 = calc_ranking_loss(x1, x2, group_mask, margin=self.margin)
        loss2 = calc_ranking_loss(x2, x1, group_mask, margin=self.margin)

        loss = (loss1 + loss2).mean()
        return loss, {'ranking_loss': loss.item()}

    def run_transformer_get_loss(self, hypo, features, keyword, group_mask=None, gt=None):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        res = OrderedDict()
        for feature in self.feature_names:
            res[feature] = getattr(self, feature)(features[feature], None)
        o, _ = self.run_transformer(hypo, res, keyword)  # BLC, BRC

        feature = sum([x.mean(dim=1) for x in res.values()])

        rank_loss, stats = self.get_rank_loss(o, feature, group_mask)

        o = self.dropout(o)
        c = o.mean(dim=1)
        c = self.reduce_c(c)
        logits, _, _ = self.get_logits(o, keyword, gt)
        return logits, c, rank_loss, stats


'''
class TransformerDis3(TransformerDis2):
    def __init__(self, args, transformer, tokenizer):
        super(TransformerDis3, self).__init__(args, transformer, tokenizer)

        self.eps = 0

    def get_keyword(self, batch, features):
        return self.keyword_classifier(batch.word_subsets, features)

    def get_logits(self, o, keyword, gt=None):
        keyword = self.smooth(keyword)
        o = self.net.lm_head(o)
        return o * keyword.unsqueeze(1).expand(-1, o.shape[1], -1), None, {}
'''


class TransformerDisMoe(TransformerDis):
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

    def get_logits(self, o, keyword, gt=None):
        # VC * NV -> NC
        embds = torch.einsum('vc,nv->nc', self.net.transformer.wte.weight, self.keyword_map)
        keyword = keyword.unsqueeze(-1) * embds  # BNC
        beta = self.gate_keyword(keyword, o)  # bln
        o = self.moe(o, beta)  # blnc
        o = self.net.lm_head(o)
        return o, None, {}


class TransformerDisPtrGen(TransformerDis):
    def __init__(self, args, transformer, tokenizer):
        super(TransformerDisPtrGen, self).__init__(args, transformer, tokenizer)

        self.k = args.get('keyword_top_k', 20)
        self.eval_random = False
        self.small_logit_only = args.get('small_logit_only', False)
        self.inf = float('-inf')
        self.big_gate_prob = args.get('big_gate_prob', 0.5)
        self.expectation_prob = args.get('expectation_prob', 0.5)
        self.gate_mlp = MLP(self.gpt_dim)
        self.small_out = nn.Linear(self.gpt_dim, self.gpt_dim)

        self.loss = SmoothLoss(padding_idx=-100)

    def gate_keyword(self, o, embds, small_logit):
        # bkc, blc, blv
        beta = self.gate_mlp(o)
        o = torch.einsum('bkc,blc->blk', embds, o)
        beta = beta.mean(dim=-1)  # bl
        beta = torch.sigmoid(beta)
        return beta

    def get_small_logit(self, keyword_map, o, embds):
        # BLC, BKC
        o = self.small_out(o)  # BLC
        o = torch.einsum('bkc,blc->blk', embds, o)
        small_logit = torch.einsum('bkv,blk->blv', keyword_map, o)
        return small_logit, o

    def drop_connect(self, beta, x1, x2):
        if self.small_logit_only:
            logits = x2
        if self.training or self.eval_random:
            # use sample for half, use true expectation for half
            # beta_samples = beta.bernoulli().float()
            beta_samples = (torch.rand(beta.shape).to(beta.device) > self.big_gate_prob).float()
            rand_flags = (torch.rand(beta.shape).to(beta.device) > self.expectation_prob).float()
            beta_rand = (1 - rand_flags) * beta + rand_flags * beta_samples
            logits = (1 - beta_rand) * x1 + beta_rand * x2
            if self.big_gate_prob == 0 and self.expectation_prob == 0:
                assert (logits == x2).all(), "Error"
        else:
            logits = (1 - beta) * x1 + beta * x2
        return logits

    def add_zero_keyword_bias(self, k):
        # any value > 0 should be selected first
        t = torch.zeros_like(k).float()
        t.masked_scatter_(k != 0, torch.ones(1).float().to(k.device).view(1, 1).contiguous().expand(*k.shape))
        k = k + t
        t.fill_(0)
        t.masked_scatter_(k == 0, self.keyword_freq.unsqueeze(0).expand(k.shape[0], -1))
        k = k + t
        return k

    def get_logits(self, o, keyword, gt=None):
        keyword_with_bias = self.add_zero_keyword_bias(keyword)
        keyword_top, keyword_top_ids = keyword_with_bias.topk(k=self.k, dim=-1)
        if self.keyword_map is not None:
            keyword_map = self.keyword_map.unsqueeze(0).expand(keyword_top_ids.shape[0], -1, -1)
            keyword_map = keyword_map.gather(1, keyword_top_ids.unsqueeze(-1).expand(
                                                -1, -1, keyword_map.shape[-1]))
            # BKV
            embds = torch.einsum('vc,bkv->bkc', self.net.transformer.wte.weight, keyword_map)
        else:
            keyword_map = onehot(keyword_top_ids, total=len(self.tokenizer)).float()
            embds = self.net.transformer.wte(keyword_top_ids)
        big_logit = self.net.lm_head(o)
        small_logit, small_vocab_logit = self.get_small_logit(keyword_map, o, embds)
        beta = self.gate_keyword(o, embds, small_logit)  # bl
        stats = {'copy_gate': beta.mean().item()}
        loss = None
        if gt is not None:
            small_tgt_val, small_tgt = (keyword_map.argmax(dim=-1).unsqueeze(1) != gt.unsqueeze(-1)).max(dim=-1)
            tgt_mask = small_tgt_val == 0
            small_tgt.masked_scatter_(tgt_mask, torch.Tensor([-100]).long().to(small_tgt.device))
            loss, _ = self.loss(small_vocab_logit, small_tgt)
            stats = {**stats, 'small_ce_loss': loss.mean().item()}
        beta = beta.unsqueeze(-1)
        logits = self.drop_connect(beta, big_logit, small_logit)
        return logits, loss, stats


'''
class TransformerDisPtrGen2(TransformerDisPtrGen):
    def __init__(self, args, transformer, tokenizer):
        super(TransformerDisPtrGen2, self).__init__(args, transformer, tokenizer)

        del self.gate_mlp

        keyword_bias = torch.ones(self.keyword_num).float()
        keyword_bias = keyword_bias * 0.5 / self.keyword_num
        keyword_bias.requires_grad_(True)
        self.keyword_bias = nn.Parameter(keyword_bias)

    def gate_keyword(self, o, embds, small_logit):
        # bkc, blc, blv
        beta = torch.einsum('blv,v->bl', small_logit, self.keyword_bias)
        beta = torch.sigmoid(beta)
        return beta
'''


class TransformerDisSmallVocab(TransformerDisPtrGen):
    def __init__(self, args, transformer, tokenizer):
        args.use_word_subset = True
        super(TransformerDisSmallVocab, self).__init__(args, transformer, tokenizer)

        self.k = args.get('keyword_top_k', 20)
        self.eval_random = True
        self.small_logit_only = False

    def get_keyword(self, batch, features):
        return self.keyword_classifier(batch.word_subsets, features)

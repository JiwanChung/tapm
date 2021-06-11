import copy
from itertools import product

import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import make_feature_lm_batch_with_keywords

from loss.base import FocalLoss, BinaryCELoss, BinaryCERecallLoss, SortedLoss
from .modules import ResBlock, SelfAttention, MultiHeadAttention, Pffn, Residual
from .transformer_model import TransformerModel


def get_keyword_classifier(name):
    dt = {
        'base': KeywordClassifier,
        'attention': KeywordClassifierAttention
    }
    if name in dt:
        return dt[name]
    else:
        return dt['base']


class KeywordClassifier(nn.Module):
    def __init__(self, wte, keyword_num, dim, feature_names,
                 feature_dims, gamma=2, dropout=0,
                 recall_k=20, loss_type='bce'):
        super(KeywordClassifier, self).__init__()

        self.eps = 1e-8
        self.recall_k = recall_k

        self.keyword_num = keyword_num
        self.feature_names = feature_names
        self.dim = dim
        self.feature_dims = feature_dims
        self.recall_weight = 100

        # self.wte = copy.deepcopy(wte)
        # self.wte.weight.requires_grad_(False)
        # self.wte.train()

        for feature in self.feature_names:
            setattr(self, feature, nn.Linear(self.feature_dims[feature], self.dim))
        self.res_block = ResBlock(self.dim, dropout)
        # self.out_linear = nn.Linear(self.dim, self.keyword_num)
        # self.out = self.out_shared if self.keyword_num == self.wte.weight.shape[0] else self.out_linear
        self.out = nn.Linear(self.dim, self.keyword_num)

        self.loss = {
            'bce': BinaryCELoss(),
            'focal': FocalLoss(gamma),
            'recall': BinaryCERecallLoss(recall_weight=self.recall_weight),
            'sorted': SortedLoss(top_k=recall_k, normalized=True),
            'sorted_unnormalized': SortedLoss(top_k=recall_k, normalized=False),
        }[loss_type.lower()]

    '''
    def out_shared(self, x):
        return torch.matmul(x, self.wte.weight.t())
    '''

    def process_features(self, features):
        # BVK, BVNC
        hypo = {}
        for feature in self.feature_names:
            hypo[feature] = getattr(self, feature)(features[feature])
            if hypo[feature].dim() > 3:
                hypo[feature] = hypo[feature].mean(dim=-2)
        # BVK
        hypo = torch.stack(list(hypo.values()), dim=0).mean(dim=0)
        hypo = self.res_block(hypo)
        return hypo

    def forward(self, keywords, features):
        hypo = self.process_features(features)
        hypo = self.out(hypo)

        loss = None
        stats = {}
        if keywords is not None:
            loss, _ = self.loss(hypo, keywords)
            hypo = torch.sigmoid(hypo)
            with torch.no_grad():
                keywords = keywords.bool().long()
                keyword_num = keywords.sum(dim=-1)
                pred_keyword_num = (hypo >= 0.5).float().sum(dim=-1)
                no_keyword_mask = keyword_num != 0
                stats['keyword_loss'] = loss.mean().item()
                stats['keyword_num'] = keyword_num.float().mean().cpu().item()
                stats['pred_keyword_num'] = pred_keyword_num.float().mean().cpu().item()
                hypo_mask = hypo >= 0.5
                keywords = keywords.bool()
                intersection = (hypo_mask & keywords).float().sum(dim=-1)
                recall = intersection / (keywords.float().sum(dim=-1) + self.eps)
                acc = intersection / (hypo_mask.float().sum(dim=-1) + self.eps)
                stats['keyword_recall'] = recall.masked_select(no_keyword_mask).mean().cpu().item()
                stats['keyword_acc'] = acc.masked_select(no_keyword_mask).mean().cpu().item()
                topk = hypo.topk(self.recall_k, dim=-1)[1]
                topk = keywords.gather(dim=-1, index=topk).float().sum(dim=-1)
                topk_recall = topk / (keywords.float().sum(dim=-1) + self.eps)
                stats[f'keyword_top{self.recall_k}_recall'] = topk_recall.masked_select(no_keyword_mask).mean().cpu().item()
        else:
            hypo = torch.sigmoid(hypo)

        return hypo, loss, stats


class KeywordClassifierAttention(KeywordClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.large_dim = self.dim * 4

        for feature in self.feature_names:
            setattr(self, f'{feature}_self_attention', nn.Sequential(
                Residual(SelfAttention(self.dim, heads=4)),
                Residual(Pffn(self.dim, self.large_dim)),
                Residual(SelfAttention(self.dim, heads=4)),
                Residual(Pffn(self.dim, self.large_dim))
            ))
            setattr(self, f'{feature}_res', ResBlock(self.dim))
        for f1, f2 in product(self.feature_names, self.feature_names):
            if f1 != f2:
                setattr(self, f'{f1}_{f2}_attention', MultiHeadAttention(self.dim, heads=4))
        self.final_res = ResBlock(self.dim)

    def process_features(self, features):
        hypo = {}
        for feature in self.feature_names:
            B, G = features[feature].shape[:2]
            hypo[feature] = getattr(self, feature)(features[feature])
            hypo[feature] = hypo[feature].view(-1, *hypo[feature].shape[2:]).contiguous()
            hypo[feature] = getattr(self, f'{feature}_self_attention')(hypo[feature])
        for f1, f2 in product(self.feature_names, self.feature_names):
            if f1 != f2:
                hypo[f1] = getattr(self, f'{f1}_{f2}_attention')(hypo[f1], hypo[f2])
        hypo = {k: h.mean(dim=1) for k, h in hypo.items()}
        hypo = {k: getattr(self, f'{k}_res')(h) for k, h in hypo.items()}
        # hypo = OrderedDict(sorted(hypo.items()))  # canonical ordering
        hypo = sum(hypo.values())
        hypo = self.final_res(hypo)
        hypo = hypo.view(B, G, -1).contiguous()

        return hypo


class KeywordClassifierWrapper(TransformerModel):
    transformer_name = 'gpt2'  # assign transformer_name = 'bert' to use BPE
    model_type = 'caption'
    use_keyword = False

    def __init__(self, args, transformer, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer

        self.dim = args.get('dim', 512)
        self.keyword_num = args.get('keyword_num', 1000)
        self.dropout_ratio = args.get('dropout', 0.5)
        self.feature_names = args.get('feature_names',
                                      ['video', 'image'])
        self.feature_dims = {k: v for k, v in args.feature_dims.items() if k in self.feature_names}
        self.keyword_loss_type = args.get('keyword_classification_loss', 'bce')

        self.gpt_dim = transformer.transformer.config.n_embd
        self.net = get_keyword_classifier(args.get('keyword_classifier', 'base'))(
            transformer.transformer.wte,
            self.keyword_num, self.gpt_dim, self.feature_names,
            self.feature_dims,
            self.dropout_ratio,
            loss_type=self.keyword_loss_type
        )

    def make_batch(self, *args, **kwargs):
        return make_feature_lm_batch_with_keywords(*args, **kwargs)

    def keyword_to_word(self, t, keyword_map):
        t = t.bool()
        t = t.view(-1, t.shape[-1]).contiguous()
        res = []
        for i in range(t.shape[0]):
            x = keyword_map.masked_select(t[i])
            res.append(x)
        return res

    def forward(self, batch, **kwargs):
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.feature_names}.items()}
        hypo, loss, stats = self.net(batch.keyword_masks, features)

        with torch.no_grad():
            stats = {**stats, 'sentence_len': (batch.sentences != self.tokenizer.pad_id).float().sum(dim=-1).mean().item()}
            hypo = hypo >= 0.5
            hypo = self.keyword_to_word(hypo, batch.keyword_map)
            targets = self.keyword_to_word(batch.keyword_masks, batch.keyword_map)
        return None, targets, loss, stats, hypo


class WordSubsetClassifier(TransformerModel):
    transformer_name = 'gpt2'  # assign transformer_name = 'bert' to use BPE
    model_type = 'caption'
    use_keyword = False

    def __init__(self, args, transformer, tokenizer):
        super().__init__()

        self.dim = args.get('dim', 512)
        self.keyword_num = args.get('keyword_num', 1000)
        self.dropout_ratio = args.get('dropout', 0.5)
        self.feature_names = args.get('feature_names',
                                      ['video', 'image', 'flow', 'box'])
        self.feature_dims = {k: v for k, v in args.feature_dims.items() if k in self.feature_names}
        self.k = args.get('keyword_top_k', 20)
        self.keyword_loss_type = args.get('keyword_classification_loss', 'bce')

        self.gpt_dim = transformer.transformer.config.n_embd
        self.net = get_keyword_classifier(args.get('keyword_classifier', 'base'))(
            transformer.transformer.wte,
            len(tokenizer), self.gpt_dim, self.feature_names,
            self.feature_dims,
            self.dropout_ratio,
            recall_k=self.k,
            loss_type=self.keyword_loss_type
        )

    def make_batch(self, *args, **kwargs):
        return make_feature_lm_batch_with_keywords(*args, **kwargs)

    def forward(self, batch, **kwargs):
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.feature_names}.items()}
        hypo, loss, stats = self.net(batch.word_subsets, features)
        return None, None, loss, stats, batch

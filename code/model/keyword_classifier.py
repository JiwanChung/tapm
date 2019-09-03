import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import make_feature_lm_batch_with_keywords

from loss import FocalLoss, BinaryCELoss
from .modules import ResBlock
from .transformer_model import TransformerModel


class KeywordClassifier(nn.Module):
    def __init__(self, keyword_num, dim, feature_names,
                 video_dim, image_dim, gamma=2, dropout=0,
                 recall_k=20, loss_type='bce'):
        super(KeywordClassifier, self).__init__()

        self.eps = 1e-8
        self.recall_k = recall_k

        self.keyword_num = keyword_num
        self.feature_names = feature_names
        self.dim = dim
        self.video_dim = video_dim
        self.image_dim = image_dim

        for feature in self.feature_names:
            setattr(self, feature, nn.Linear(getattr(self, f"{feature}_dim"), self.dim))
        self.res_block = ResBlock(self.dim, dropout)
        self.out = nn.Linear(self.dim, self.keyword_num)

        self.loss = {
            'bce': BinaryCELoss(),
            'focal': FocalLoss(gamma)
        }[loss_type.lower()]

    def forward(self, keywords, features):
        # BVK, BVNC
        hypo = {}
        for feature in self.feature_names:
            hypo[feature] = getattr(self, feature)(features[feature])
            if hypo[feature].dim() > 3:
                hypo[feature] = hypo[feature].mean(dim=-2)
        # BVK
        hypo = torch.stack(list(hypo.values()), dim=0).mean(dim=0)
        hypo = self.res_block(hypo)
        hypo = self.out(hypo)

        loss = None
        stats = {}
        if keywords is not None:
            loss, _ = self.loss(hypo, keywords)
            hypo = torch.sigmoid(hypo)
            with torch.no_grad():
                keywords = keywords.byte().long()
                keyword_num = keywords.sum(dim=-1)
                no_keyword_mask = keyword_num != 0
                stats['keyword_num'] = keyword_num.float().mean().cpu().item()
                hypo_mask = hypo >= 0.5
                keywords = keywords.byte()
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


class KeywordClassifierWrapper(TransformerModel):
    transformer_name = 'none'  # assign transformer_name = 'bert' to use BPE
    model_type = 'caption'
    use_keyword = False

    def __init__(self, args, transformer, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer

        self.dim = args.get('dim', 512)
        self.video_dim = args.get('video_dim', 1024)
        self.image_dim = args.get('image_dim', 2048)
        self.keyword_num = args.get('keyword_num', 1000)
        self.dropout_ratio = args.get('dropout', 0.5)
        self.feature_names = args.get('feature_names',
                                      ['video', 'image'])
        self.keyword_loss_type = args.get('keyword_classification_loss', 'bce')

        self.net = KeywordClassifier(
            self.keyword_num, self.dim, self.feature_names,
            self.video_dim, self.image_dim, self.dropout_ratio,
            loss_type=self.keyword_loss_type
        )

    def make_batch(self, *args, **kwargs):
        return make_feature_lm_batch_with_keywords(*args, **kwargs)

    def forward(self, batch, **kwargs):
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.feature_names}.items()}
        hypo, loss, stats = self.net(batch.keyword_masks, features)
        stats = {**stats, 'sentence_len': (batch.sentences != self.tokenizer.pad_id).float().sum(dim=-1).mean().item()}
        return None, None, loss, stats, batch


class WordSubsetClassifier(TransformerModel):
    transformer_name = 'none'  # assign transformer_name = 'bert' to use BPE
    model_type = 'caption'
    use_keyword = False

    def __init__(self, args, transformer, tokenizer):
        super().__init__()

        self.dim = args.get('dim', 512)
        self.video_dim = args.get('video_dim', 1024)
        self.image_dim = args.get('image_dim', 2048)
        self.keyword_num = args.get('keyword_num', 1000)
        self.dropout_ratio = args.get('dropout', 0.5)
        self.feature_names = args.get('feature_names',
                                      ['video', 'image'])
        self.k = args.get('keyword_top_k', 20)
        self.keyword_loss_type = args.get('keyword_classification_loss', 'bce')

        self.net = KeywordClassifier(
            len(tokenizer), self.dim, self.feature_names,
            self.video_dim, self.image_dim, self.dropout_ratio,
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

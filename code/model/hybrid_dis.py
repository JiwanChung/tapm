from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from utils import jsonl_to_json, mean
from data.batcher import make_feature_lm_batch_with_keywords

from .modules import Attention, GRU
from .scn_rnn import SCNLSTM
from .transformer_model import TransformerModel
from .keyword_classifier import KeywordClassifier


'''
currently, this implementation deviates from the original repo
in the following regards:
    1. GRU instead of LSTM
    2. An (deactivated) option to share in_out embeddings
Aside from the above, I tried to closely follow the given details.
'''
class HybridDis(TransformerModel):
    transformer_name = 'none'  # assign transformer_name = 'bert' to use BPE
    model_type = 'caption'
    use_keyword = False

    def __init__(self, args, transformer, tokenizer):
        super(HybridDis, self).__init__()

        self.dim = args.get('dim', 512)
        self.video_dim = args.get('video_dim', 1024)
        self.image_dim = args.get('image_dim', 2048)
        self.keyword_num = args.get('keyword_num', 1000)
        self.dropout_ratio = args.get('dropout', 0.5)
        self.feature_names = args.get('feature_names',
                                      ['video', 'image'])
                                      # ['video', 'image', 'box'])
        self.share_in_out = args.get('share_in_out',
                                     False)
        self.use_gt_keywords = args.get('use_gt_keywords', False)
        self.use_word_subset = args.get('use_word_subset', False)
        self.max_target_len = args.max_target_len
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.k = args.get('keyword_top_k', 20)
        self.keyword_loss_type = args.get('keyword_classification_loss', 'bce')

        for feature in self.feature_names:
            setattr(self, feature, FeatureEncoder(getattr(self, f"{feature}_dim"), self.dim))
        self.encoder = nn.Linear(len(self.feature_names) * self.dim, self.dim)
        self.pretrained_embedding = args.get('pretrained_embedding', False)
        self.wte_dim = 300 if self.pretrained_embedding else self.dim
        self.wte = nn.Embedding(self.vocab_size, self.wte_dim)

        self.keyword_num = len(tokenizer) if self.use_word_subset else self.keyword_num
        self.keyword_classifier = KeywordClassifier(
            self.wte,
            self.keyword_num, self.dim, self.feature_names,
            self.video_dim, self.image_dim, self.dropout_ratio,
            recall_k=self.k,
            loss_type=self.keyword_loss_type
        )

        self.context_dim = self.dim // 4
        num_layers = 1
        self.rnn = {
            'rnn': GRU(num_layers, self.wte_dim + self.dim + self.context_dim, self.dim, dropout=self.dropout_ratio),
            'scn': SCNLSTM(self.wte_dim + self.dim + self.context_dim, self.keyword_num, self.dim,
                           num_layers, batch_first=True, dropout=self.dropout_ratio)
        }[args.get('decoder_type', 'scn')]
        self.context_encoder = PrevEncoder(self.dim, self.context_dim)
        self.dropout = nn.Dropout(self.dropout_ratio)

        if self.share_in_out:
            self.out = self.out_shared
        else:
            self.out = nn.Linear(self.dim, self.vocab_size)
        self.init_weights()
        self.use_context = False

    def make_batch(self, *args, **kwargs):
        return make_feature_lm_batch_with_keywords(*args, **kwargs)

    def epoch_update(self, epoch):
        if epoch > 10:
            self.context = True

    def get_keyword_map(self, ids):
        # get NV
        if not self.use_word_subset:
            storage = torch.zeros(ids.shape[0], len(self.tokenizer)).float().to(ids.device)
            storage.scatter_(-1, ids.unsqueeze(-1), 1)
        else:
            # storage = torch.eye(len(self.tokenizer)).float().to(ids.device)
            storage = None
        return storage

    def init_weights(self):
        init_range = 0.1
        for feature in self.feature_names:
            getattr(self, feature).linear.weight.data.uniform_(-init_range, init_range)
        if not self.share_in_out:
            self.out.bias.data.fill_(0)
            self.out.weight.data.uniform_(-init_range, init_range)
        if self.pretrained_embedding is not None:
            self.wte.load_state_dict({'weight': self.tokenizer.embedding})

    def out_shared(self, x):
        return torch.matmul(x, self.wte.weight.t())

    def generate_token(self, hypo, features, c, h, keyword):
        s = hypo[:, -1]  # get last token
        s = self.wte(s).unsqueeze(1)  # B1C
        s = torch.cat((features, c, s), dim=-1)
        o, h = self.rnn(s, h, keyword=keyword)
        o = self.dropout(o)
        logits = self.out(o)  # BV
        return logits, h

    def run_token(self, features, hypo, h, c, keyword):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        for feature in self.feature_names:
            features[feature] = getattr(self, feature)(features[feature], h)
        features = self.encoder(torch.cat(list(features.values()), dim=-1))
        logits, h = self.generate_token(hypo, features, c, h, keyword)
        return h, c, logits

    def run_video(self, features, c, v, L, sentences=None, sampler=None,
                  keyword=None, reduce_hypo=True):
        video = features['video']
        B = video.shape[0]
        empty = torch.full((B, self.vocab_size), float('-inf')).to(video.device)
        sent = []
        eos_flags = torch.LongTensor([0] * B).byte().to(video.device)
        h = self.rnn.init_h(B, device=video.device) if hasattr(self, 'rnn') else None
        c = self.rnn.init_c(B, self.context_dim, device=video.device) if hasattr(self, 'rnn') else None
        s0 = sentences[:, v, 0] if sentences is not None \
            else torch.Tensor([self.tokenizer.cls_id]).long().to(video.device).expand(B)
        s = s0
        hypo = s0.unsqueeze(-1)

        for w in range(L):
            if eos_flags.all():
                logits = empty.clone()
            else:
                h, c, logits = self.run_token(features, hypo, h, c, keyword=keyword)
                if sentences is not None:  # training
                    s = sentences[:, v, min(L - 1, w + 1)].clone()
                    eos_flags = eos_flags | (sentences[:, v, min(L - 1, w + 1)] == self.tokenizer.sep_id)
                else:
                    s, probs = sampler(logits)
                    eos_flags = eos_flags | (logits.argmax(dim=-1) == self.tokenizer.pad_id)
            hypo = torch.cat((hypo, s.unsqueeze(-1)), dim=1)
            sent.append(logits)

        hypo = hypo[:, 1:]
        if sentences is None and reduce_hypo:
            hypo = hypo[probs.argmax(dim=-1)]
        else:
            sent = torch.stack(sent, 1).contiguous()
        c = self.context_encoder(h)
        if not self.use_context:
            c = torch.full_like(c.detach(), 0)
            c.requires_grad_(False)
        return c, sent, hypo, None, {}

    def get_keyword(self, batch, features):
        keyword = batch.word_subsets if self.use_word_subset else batch.keyword_masks
        return self.keyword_classifier(keyword, features)

    def forward(self, batch, **kwargs):
        # BVLC, BVL
        video = batch.video
        B, V = video.shape[:2]
        L = batch.sentences.shape[2] if hasattr(batch, 'sentences') else self.max_target_len
        sent_gt = batch.sentences if hasattr(batch, 'sentences') else None

        if (not hasattr(self, 'keyword_map')) and hasattr(batch, 'keyword_map'):
            self.keyword_map = self.get_keyword_map(batch.keyword_map)

        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.feature_names}.items()}
        keywords, reg_loss, stats = self.get_keyword(batch, features)
        keywords = keywords.detach()
        if self.use_gt_keywords:
            if not self.use_word_subset:
                keywords = batch.keyword_masks.float()
            else:
                keywords = batch.word_subsets.float()
        stats = {**stats, 'sentence_len': (batch.sentences != self.tokenizer.pad_id).float().sum(dim=-1).mean().item()}

        res = []
        vid_stats = []
        losses = []
        for v in range(V):
            feature = {k: val[:, v] for k, val in features.items()}
            c = self.rnn.init_c(B, self.context_dim, device=video.device) if hasattr(self, 'rnn') else None
            keyword = keywords[:, v] if keywords is not None else None
            c, sent, _, small_loss, vid_stat = self.run_video(feature, c, v, L, sentences=sent_gt, keyword=keyword)
            losses.append(small_loss)
            vid_stats.append(vid_stat)
            res.append(sent)  # BLV
        vid_stats = {k: mean(v) for k, v in jsonl_to_json(vid_stats).items()}
        stats = {**stats, **vid_stats}
        del batch.sentences  # for generation
        small_loss = None if losses[0] is None else mean(losses)
        reg_loss = reg_loss + small_loss
        return torch.stack(res, 1).contiguous(), batch.targets, reg_loss, stats, batch


class FeatureEncoder(nn.Module):
    def __init__(self, video_dim, dim):
        super(FeatureEncoder, self).__init__()

        self.linear = nn.Linear(video_dim, dim)
        self.attention = Attention(dim)

    def forward(self, feature, h):
        # BLC
        if isinstance(h, tuple):  # check LSTM/GRU
            h = h[0]
        feature = self.linear(feature)
        h = h.mean(dim=1)
        return self.attention(h, feature).unsqueeze(1)


class PrevEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PrevEncoder, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h):
        # BLC
        if isinstance(h, tuple):  # check LSTM/GRU
            h = h[0]
        return self.linear(h)

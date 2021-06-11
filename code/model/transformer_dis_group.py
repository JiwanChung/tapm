from collections import OrderedDict, defaultdict
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from exp import ex

from data.batcher import pad_tensor
from sampler import get_sampler
from run_transformer import transformer_lm_head
from utils import flatten_list, chunks

from .collaborative_experts import CollaborativeExperts
from .ss_loss import calc_loss
from .transformer_dis import TransformerDis, FeatureEncoder
from .modules import Residual, MLP
from .NetVLAD import NetVLADWrapper
from .encoders import DeepEncoder


class TransformerDisGroup(TransformerDis):
    model_type = 'caption_single'
    reinforce_available = True
    transformer_name = 'gpt2'

    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, num_samples):
        super().__init__(transformer, tokenizer, dropout_before)

        self.reg_loss_after = False

        self.frame_start_linear = nn.Linear(self.gpt_dim, 1)
        self.frame_end_linear = nn.Linear(self.gpt_dim, 1)

        self.ce_loss = nn.CrossEntropyLoss()

        self.num_samples = num_samples
        self.negative_large = -1e+10

        self.eps = 1e-10

    def baseline_estimator(self, x):
        if not hasattr(self, 'baseline'):
            self.baseline = nn.Linear(self.gpt_dim, 1)
            self.baseline.weight.data = self.baseline.weight.data.to(x.device)
            if hasattr(self.baseline, 'bias'):
                self.baseline.bias.data = self.baseline.bias.data.to(x.device)

        return self.baseline(x).squeeze(-1)

    def process_features(self, features):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        res = OrderedDict()
        for feature in self.feature_names:
            res[feature] = getattr(self, feature)(features[feature])
        return res

    def get_reg_loss(self, o, f, mask):
        return None, {}

    def get_frame_loss(self, c, features, frame):
        if frame is not None:
            features = OrderedDict(sorted(features.items()))  # canonical ordering
            shape = features[list(features.keys())[0]].shape
            c = c[:, :, 1: shape[2] + 1]  # remove cls, use the first feature shape
            loss, stats = self._get_frame_loss(c, frame)
            return loss, {**stats, 'frame_loss': loss.item()}
        else:
            return None, {}

    def get_ce_loss(self, hypo, tgt):
        hypo = hypo.contiguous().view(-1, hypo.shape[-1])
        tgt = tgt.contiguous().view(-1)
        return self.ce_loss(hypo, tgt)

    def _get_frame_loss(self, c, frame):
        # BGLC, BG2
        loss_start = self.get_ce_loss(self.frame_start_linear(c).squeeze(-1), frame[:, :, 0])
        loss_end = self.get_ce_loss(self.frame_end_linear(c).squeeze(-1), frame[:, :, 1])

        return loss_start + loss_end, {}

    def process_reg_loss(self, o, context, features, group_mask, G, frame=None):
        o = rearrange(o, '(b g) l c -> b g l c', g=G)
        c = rearrange(context, '(b g) l c -> b g l c', g=G)

        frame_supervision_loss, frame_stats = self.get_frame_loss(c, features, frame)
        reg_loss, stats = self._process_reg_loss(o, c, features, group_mask, G)

        if frame_supervision_loss is not None:
            if reg_loss is not None:
                reg_loss = frame_supervision_loss + reg_loss
            else:
                reg_loss = frame_supervision_loss
        stats = {**frame_stats, **stats}

        return reg_loss, stats

    def mean_pool_text(self, o):
        return o.mean(dim=-2)

    def mean_pool_features(self, features):
        return sum([x.mean(dim=-2) for x in features])

    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss, stats = self.get_reg_loss(o.detach(), feature, group_mask)
        return reg_loss, stats

    def prepare_group(self, batch):
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.feature_names}.items()}
        keywords, stats, reg_loss = self.process_keyword(batch, features)

        G = features[list(features.keys())[0]].shape[1]

        features = self.process_features(features)
        keywords = rearrange(keywords.contiguous(), 'b g k -> (b g) k')
        features_merged = {k: rearrange(v.contiguous(), 'b g l c -> (b g) l c') for k, v in features.items()}

        return features, features_merged, keywords, G

    def run_token(self, B, hypo, features, keywords, infer=False):
        o, context = self.run_transformer(B, hypo, features, keywords,
                                          infer=infer)
        # (BG)LC
        logits = transformer_lm_head(self.net, o)

        return logits, o, context

    def forward(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    @ex.capture
    def _forward(self, batch, sampler=None, epoch=0,
                 reduce_hypo=True, reinforce=False,
                 length_normalize_beam=False, sample_ema_coeff=0,
                 sample_eval_at_last=False, auxloss_generate=False,
                 postprocess_duplicates=1,
                 **kwargs):
        features, features_merged, keywords, G = self.prepare_group(batch)
        if getattr(self, 'auxloss_generate', False):
            auxloss_generate = self.auxloss_generate
        '''
        sampler = kwargs.get('sampler', None)
        reduce_hypo = kwargs.get('reduce_hypo', True)
        reinforce = kwargs.get('reinforce', False)
        length_normalize_beam = kwargs.get('length_normalize_beam', False)
        sample_ema_coeff = kwargs.get('sample_ema_coeff', 0)
        sample_eval_at_last = kwargs.get('sample_eval_at_last', False)
        auxloss_generate = kwargs.get('auxloss_generate', False)
        '''
        if sampler is None:  # training
            f = self.run_train
        else:
            f = self.run_generation

        hypo, logits, reg_loss, stats = f(batch, features, features_merged, keywords,
                               sampler=sampler, reduce_hypo=reduce_hypo,
                               reinforce=reinforce,
                               length_normalize_beam=length_normalize_beam,
                               sample_ema_coeff=sample_ema_coeff,
                               sample_eval_at_last=sample_eval_at_last,
                               train_auxloss_generate=auxloss_generate,
                               postprocess_duplicates=postprocess_duplicates
                                          )

        targets = None
        if hasattr(batch, 'sentences'):
            stats = {**stats, 'sentence_len': (batch.sentences != self.tokenizer.pad_id).float().sum(dim=-1).mean().item()}
            targets = batch.targets

        if self.normalizer_alpha is not None:
            stats['alpha'] = self.normalizer_alpha.item()

        return hypo, logits, targets, reg_loss, stats, batch

    def run_train(self, batch, features, features_merged, keywords, **kwargs):
        hypo = batch.sentences
        B, G = hypo.shape[:2]
        hypo = rearrange(hypo.contiguous(), 'b g l -> (b g) l')
        if kwargs.get('train_auxloss_generate', False):
            greedy_sampler = get_sampler(self, 'greedy')
            _, logits, _, _ = self.run_generation(batch, features, features_merged,
                                                  keywords,
                                                  sampler=greedy_sampler.sample_token_faster_greedy,
                                                  auxloss_generate=True,
                                                  sample_eval_at_last=True)
            logits, o, context = logits
        else:
            logits, o, context = self.run_token(B, hypo,
                                                features_merged, keywords,
                                                infer=False)
            if torch.isnan(logits).any():
                import ipdb; ipdb.set_trace()  # XXX DEBUG
        frame = batch.frame if hasattr(batch, 'frame') else None
        reg_loss, stats = self.process_reg_loss(o, context, features,
                                                batch.group_mask, G,
                                                frame)

        return o, logits, reg_loss, stats

    def get_hypo_init(self, B, G):
        s0 = torch.Tensor([self.tokenizer.cls_id]).long().expand(B)
        s0 = s0.unsqueeze(-1)
        hypo = s0.unsqueeze(-1)

        return hypo

    def run_generation(self, batch, features, features_merged, keywords,
                       sampler=None, reduce_hypo=True, reinforce=False,
                       length_normalize_beam=False, postprocess_duplicates=1,
                       sample_ema_coeff=0, sample_eval_at_last=False,
                       auxloss_generate=False, **kwargs):
        sample_feature = features[list(features.keys())[0]]
        G = sample_feature.shape[1]
        B = features_merged[list(features.keys())[0]].shape[0]  # (BG)
        device = features_merged[list(features.keys())[0]].device
        empty = torch.full((B, self.num_samples, self.vocab_size), float('-inf'), dtype=sample_feature.dtype).to(device)
        hypo_init = self.get_hypo_init(B, G).to(device)
        s = hypo_init.clone()
        hypo = hypo_init.clone()
        eos_flags_empty = torch.LongTensor([0] * B).bool().to(device).unsqueeze(1).expand(B, self.num_samples)
        eos_flags = eos_flags_empty.clone()

        reg_loss = None,
        stats = {}
        training = self.training
        self.eval()
        hypos_fin = defaultdict(list)
        term_probs_fin = defaultdict(list)
        curr_probs = None
        if sample_eval_at_last:
            sample_ema_coeff = 1
        with torch.no_grad():
            for i, w in enumerate(range(self.max_target_len)):
                eos_batch = eos_flags.prod(dim=-1).bool()
                if eos_flags.all():
                    logits = empty.clone()
                else:
                    logits = []
                    if (hypo > len(self.tokenizer)).any():
                        import ipdb; ipdb.set_trace()  # XXX DEBUG
                    for n in range(hypo.shape[1]):
                        logit, _, _ = self.run_token(
                            B,
                            hypo[:, n].to(device),
                            features_merged, keywords,
                            infer=True
                        )
                        logits.append(logit[:, -1])  # get last token
                    logits = torch.stack(logits, dim=1)  # BNV
                    if torch.isnan(logits).any():
                        import ipdb; ipdb.set_trace()  # XXX DEBUG
                    if getattr(logit, 'if_log_softmax', False):
                        logits.if_log_softmax = logit.if_log_softmax
                    if not getattr(logits, 'if_log_softmax', False):
                        cumul_prob = sampler.normalizer(logits, dim=-1,
                                                      alpha=self.normalizer_alpha)
                    else:
                        cumul_log_prob = logits
                        cumul_prob = cumul_log_prob.exp()
                    if curr_probs is not None:
                        # cumul_prob = sample_ema_coeff * cumul_prob + (1 - sample_ema_coeff) * curr_probs.unsqueeze(-1)
                        cumul_prob = cumul_prob ** sample_ema_coeff * curr_probs.unsqueeze(-1) ** (1 - sample_ema_coeff)
                    s, token_probs, hypo_keys = sampler(logits, cumul_prob=cumul_prob,
                                                        alpha=self.normalizer_alpha)  # BN, BN, BN
                    token_log_probs = token_probs.log()
                    if curr_probs is None:
                        curr_probs = token_probs
                    else:
                        # curr_probs = sample_ema_coeff * token_probs + (1 - sample_ema_coeff) * curr_probs
                        curr_prob = token_probs ** sample_ema_coeff * curr_probs ** (1 - sample_ema_coeff)

                    if i == 0:  # first iter
                        hypo = hypo.repeat(1, s.shape[1], 1)
                    hypo_keys = hypo_keys.unsqueeze(-1).expand(hypo.shape)
                    if hypo_keys.max() >= hypo.shape[1]:
                        import ipdb; ipdb.set_trace()  # XXX DEBUG
                    hypo = hypo.gather(dim=1, index=hypo_keys)
                    hypo = torch.cat((hypo, s.unsqueeze(-1)), dim=-1)
                    eos_flags = eos_flags | (s == self.tokenizer.sep_id).bool()  # BN

                    # pop and store finished sentences
                    if eos_flags.any() and not sample_eval_at_last:
                        # hypos: BNL, curr_probs: BN
                        # store
                        for b in range(eos_flags.shape[0]):
                            if eos_flags[b].any():  # N
                                L = hypo[b].shape[1]
                                idx = eos_flags[b].nonzero().squeeze(-1)  # I
                                if idx.max() >= hypo[b].shape[0]:
                                    import ipdb; ipdb.set_trace()  # XXX DEBUG
                                hypo_fin = hypo[b].gather(dim=0,
                                                          index=idx.unsqueeze(-1).repeat(1, L))  # IL
                                if idx.max() >= curr_probs[b].shape[0]:
                                    import ipdb; ipdb.set_trace()  # XXX DEBUG
                                term_prob_fin = curr_probs[b].gather(dim=0, index=idx)
                                if not eos_batch[b].all():
                                    hypos_fin[b].append(hypo_fin)
                                    term_probs_fin[b].append(term_prob_fin)

                        # fill
                        curr_probs = (~eos_flags).float() * curr_probs + \
                            eos_flags.float() * self.negative_large # inf out finished sentence
                        best_idx = curr_probs.argmax(dim=-1).unsqueeze(1)  # B1
                        # B1L
                        best_hypo = hypo.gather(dim=1,
                                                index=best_idx.unsqueeze(-1).repeat(1, 1, hypo.shape[-1]))
                        # B1
                        best_prob = curr_probs.gather(dim=1,
                                                index=best_idx)
                        mask = eos_flags
                        hypo_mask = mask.unsqueeze(-1).repeat(1, 1, hypo.shape[-1])
                        best_hypo = best_hypo.repeat(1, hypo.shape[1], 1)
                        best_prob = best_prob.repeat(1, curr_probs.shape[1])
                        hypo_mask = hypo_mask.bool()
                        hypo = hypo * (~hypo_mask).long() + best_hypo * hypo_mask.long()
                        curr_probs = curr_probs * (~mask).float() + best_prob * mask.float()
                        if not eos_flags.all():
                            eos_flags = eos_flags_empty

            for b in range(hypo.shape[0]):
                if len(hypos_fin[b]) == 0:
                    hypos_fin[b].append(hypo[b])
                    term_probs_fin[b].append(curr_probs[b])
                    # log_probs_fin[b].append(log_probs[b])

            if reduce_hypo:
                hypos_fin = list(sorted(hypos_fin.items()))
                hypos_fin = [list(chain(*[t.split(1, dim=0) for t in v])) for b, v in hypos_fin]
                hypo_lengths = [torch.LongTensor([t.shape[1] for t in v]).to(device) for v in hypos_fin]
                hypos_fin = [pad_tensor(v, val=self.tokenizer.pad_id).long() for v in hypos_fin]  # BIL
                hypos_fin = [v.squeeze(1) for v in hypos_fin]
                term_probs_fin = [torch.cat(v, dim=0) for b, v in sorted(term_probs_fin.items())]  # BI

                '''
                ema_coeff_fin = torch.full((self.max_target_len,), (1 - sample_ema_coeff)).float().to(device)
                ema_coeff_fin = ema_coeff_fin.cumprod(dim=-1) / ema_coeff_fin  # (1-a)^0, (1-a)^1, ...
                ema_coeff_fin = ema_coeff_fin.flip(dims=[-1]) * sample_ema_coeff  # a (1-a)^(t-1), a (1-a)^(t), ...
                ema_coeff_fin[0] = ema_coeff_fin[0] / sample_ema_coeff  # (1-a)^(t-1), a (1-a)^(t), ...
                ema_coeff_fin = ema_coeff_fin.unsqueeze(0)
                '''

                if length_normalize_beam:
                    term_probs_fin = [v / l.float() for v, l in zip(term_probs_fin, hypo_lengths)]  # BI

                if postprocess_duplicates != 1:
                    duplicate_threshold = 0.5
                    # penalty duplicate
                    best_probs = [p.topk(min(postprocess_duplicates, p.shape[0])) for p in term_probs_fin]
                    hypo = [(pad_tensor([v[i] for i in probs[1]], val=self.tokenizer.pad_id).long(), probs[0]) for probs, v in zip(best_probs, hypos_fin)]
                    hypo = list(chunks(hypo, G))
                    for g, h in enumerate(hypo):
                        li = []
                        for i, t in enumerate(h):
                            t, probs = t
                            if i == 0:  # base case
                                li.append(t[0])
                            else:
                                li_t = pad_tensor(li, val=self.tokenizer.pad_id).long().to(device)
                                li_t = li_t[:, :t.shape[1]].unsqueeze(1)
                                pad_mask = (li_t != self.tokenizer.pad_id) & (li_t != self.tokenizer.sep_id)
                                t_short = t[:, :li_t.shape[2]].unsqueeze(0).to(device)
                                masks = (t_short == li_t) * pad_mask  # ikL
                                masks = masks.float().sum(dim=-1)  # ik
                                masks = masks / pad_mask.float().sum(dim=-1)  # normalize
                                masks = masks.max(dim=0)[0]  # k
                                masks = (masks <= duplicate_threshold)
                                if masks.max() == 1:  # at least one candidate
                                    probs *= masks.float()  # mask out duplicates
                                idx = probs.argmax()
                                li.append(t[idx])
                        hypo[g] = li
                    hypo = flatten_list(hypo)
                else:
                    best_ids = [p.argmax() for p in term_probs_fin]
                    hypo = [v[i] for i, v in zip(best_ids, hypos_fin)]
                hypo = pad_tensor(hypo, val=self.tokenizer.pad_id).long()
                # ht = [self.tokenizer.decode(h[1:].cpu().numpy()) for h in hypo]
                '''
                hypo = hypo.gather(1, best_ids.contiguous().view(-1, 1, 1).repeat(1, 1, hypo.shape[-1]))
                hypo = hypo.squeeze(1)
                '''

        if training:
            self.train()

        if reinforce or auxloss_generate:
            logits, o, context = self.run_token(B, hypo[:, :-1].to(device), features_merged, keywords,
                                                infer=False)
            logits = rearrange(logits.contiguous(), '(b g) l v -> b g l v', g=G)
            if reinforce:
                baseline = self.baseline_estimator(o.detach())
                baseline = rearrange(baseline.contiguous(), '(b g) c -> b g c', g=G)
                logits = (logits, baseline)
            elif auxloss_generate:
                logits = (logits, o, context)
        else:
            logits = None

        hypo = hypo[:, 1:]  # remove sos
        hypo = rearrange(hypo.contiguous(), '(b g) l -> b g l', g=G)
        init_length = hypo_init.shape[-1]
        hypo = hypo[:, :, init_length - 1:]

        return hypo, logits, reg_loss, stats


class TransformerDisGroupReg(TransformerDisGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.margin = 1

        self.roll_linear = nn.ModuleDict()
        self.roll_linear['left'] = nn.Linear(self.gpt_dim, self.gpt_dim)
        self.roll_linear['right'] = nn.Linear(self.gpt_dim, self.gpt_dim)
        self.direction = {'left': -1, 'right': 1}

        self.calc_loss = calc_loss

    def get_reg_loss(self, h, c, group_mask):
        roll_loss, roll_stats = self.get_roll_losses(h, c, group_mask)
        rank_loss, rank_stats = self.get_rank_loss(h, c, group_mask)

        loss = (roll_loss + rank_loss).mean()

        stats = {**roll_stats,
                 **rank_stats,
                 'roll_loss': roll_loss.item(),
                 'rank_loss': rank_loss.item()}

        return loss, stats

    def get_roll_loss(self, h, c, group_mask, direction):
        h_rolled = torch.roll(h, self.direction[direction], 1)
        m_rolled = torch.roll(group_mask.byte(), self.direction[direction], 1).bool() & group_mask

        loss, stats = self.get_rank_loss(self.roll_linear[direction](c),
                                         h_rolled, m_rolled,
                                         skip_idx=self.direction[direction])
        return loss, {f"{direction}_{k}": v for k, v in stats.items()}

    def get_roll_losses(self, h, c, group_mask):
        left_loss, left_stats = self.get_roll_loss(h, c, group_mask, 'left')
        right_loss, right_stats = self.get_roll_loss(h, c, group_mask, 'right')
        stats = {
            **left_stats, **right_stats,
            'roll_accuracy': (left_stats['left_rank_accuracy'] + right_stats['right_rank_accuracy']) / 2
        }

        if left_loss is not None and right_loss is not None:
            loss = (left_loss + right_loss).mean()
        elif left_loss is not None:
            loss = left_loss.mean()
        elif right_loss is not None:
            loss = right_loss.mean()
        else:
            loss = None

        return loss, stats

    def get_rank_loss(self, h, c, group_mask, skip_idx=0):
        x1 = F.normalize(h)
        x2 = F.normalize(c)

        loss1, acc1, loss2, acc2 = self.run_rank_loss(x1, x2, group_mask, skip_idx)

        stats = {'rank_accuracy': (acc1 + acc2) / 2}
        if loss1 is None or loss2 is None:
            loss = None
        else:
            loss = (loss1 + loss2).mean()
        return loss, stats

    def run_rank_loss(self, x1, x2, group_mask, skip_idx=0):
        x1 = x1.view(-1, x1.shape[-1])
        x2 = x2.view(-1, x2.shape[-1])
        group_mask = group_mask.view(-1)

        loss1, acc1 = self.calc_loss(x1, x2, group_mask,
                                        margin=self.margin, pool='mean',
                                        skip_idx=skip_idx)
        loss2, acc2 = self.calc_loss(x2, x1, group_mask,
                                        margin=self.margin, pool='mean',
                                        skip_idx=-skip_idx)

        return loss1, acc1, loss2, acc2


class TransformerDisGroupRegFeat(TransformerDisGroupReg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reg_loss_after = True

        self.ce = CollaborativeExperts(
            self.feature_names,
            self.feature_dims,
            self.gpt_dim,
            feature_encoder = DeepEncoder
        )

        self.special_features = ['human_i3d']
        self.special_features = [f for f in self.special_features
                                 if f in self.feature_names]
        for feature in self.special_features:
            dim = self.feature_dims[feature]
            setattr(self, f"enc_{feature}",
                nn.Sequential(*[NetVLADWrapper(feature_size=dim, cluster_size=48),
                                FeatureEncoder(dim * 48, dim)]))

    def process_features(self, features):
        for feature in self.special_features:
            features[feature] = getattr(self, f"enc_{feature}")(features[feature])

        features = OrderedDict(sorted(features.items()))  # canonical ordering
        return self.ce._forward(features)

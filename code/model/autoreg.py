from collections import OrderedDict, defaultdict
from itertools import chain

import torch
# from torch import nn
import torch.nn.functional as F
from einops import rearrange

from exp import ex
from run_transformer import transformer_embed, transformer_run_cells
from data.batcher import pad_tensor
from utils import flatten_list, chunks

from .no_gt import NoGtSos


class AutoReg(NoGtSos):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

    def run_transformer(self, B, hypo, features, keyword, infer=False):
        text_reps = None
        if isinstance(hypo, dict):
            text_reps = hypo['hypo_context']
            hypo = hypo['hypo']
        h, inputs = transformer_embed(self.net.transformer, hypo,
                                      skip_ids=[self.tokenizer.pad_id, self.tokenizer.sep_id],
                                      infer=infer)

        if text_reps is None:
            pad_mask = hypo != self.tokenizer.pad_id  # (BG)L
            pad_mask = pad_mask.unsqueeze(-1)  # (BG)L1
            text_reps = h.clone()
            text_reps.masked_fill_(~pad_mask, 0)

            pad_mask_sum = pad_mask.float().sum(dim=1)
            text_reps = text_reps.sum(dim=1) / (pad_mask_sum + self.eps)  # BC
            pad_mask_sum = (pad_mask_sum > 0)  # B
            text_reps.masked_fill_(~pad_mask_sum, 0)
            text_reps = rearrange(text_reps, '(b g) c -> b g c', b=B)
            # h = self.add_keyword(h, keyword)
            # text_reps = h.mean(dim=2)  # b g c
            B, G, C = text_reps.shape
            # text_reps = rearrange(text_reps.contiguous(), 'b g c -> b g k c', k=g - 1)
            text_reps = text_reps.unsqueeze(1).expand(B, G, G, C)[:, :, :-1]

            k = text_reps.shape[-2]
            text_reps = text_reps.clone()
            for i in range(G):
                text_reps[:, i, i:] = 0
            text_reps = text_reps.detach()
            text_reps = rearrange(text_reps.contiguous(), 'b g x c -> (b g) x c')

        k = text_reps.shape[1]
        h = torch.cat((text_reps, h), dim=-2)  # (B G) g-1+l c
        cls_embd = self.net.transformer.word_embedding(torch.LongTensor([self.tokenizer.cls_id]).to(h.device))
        sep_embd = self.net.transformer.word_embedding(torch.LongTensor([self.tokenizer.sep_id]).to(h.device))
        B, L, C = h.shape
        cls_embd = cls_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        sep_embd = sep_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        context = self.merge_context(features, cls_embd, sep_embd)

        o, context_embedded = transformer_run_cells(self.net.transformer, context, h, **inputs)
        o = self.dropout(o)
        context_embedded = self.visual_dropout(context_embedded)

        o = o[:, k:]

        return o, context_embedded

    def get_hypo_context(self, hypo_li, B, G, device):
        C = self.gpt_dim
        c = torch.zeros(B, G - 1, C).to(device).float()
        for i, hypo in enumerate(hypo_li):
            hypo = hypo.to(device)
            pad_mask = hypo != self.tokenizer.pad_id  # BL
            pad_mask = pad_mask.unsqueeze(-1)  # BL1
            h = self.net.transformer.word_embedding(hypo)  # BLC
            h.masked_fill_(~pad_mask, 0)
            pad_mask_sum = pad_mask.float().sum(dim=1)
            h = h.sum(dim=1) / (pad_mask_sum + self.eps)  # BC
            pad_mask_sum = (pad_mask_sum > 0)  # B
            h.masked_fill_(~pad_mask_sum, 0)
            c[:, i] = h
        return c.detach()

    def run_generation(self, batch, features, features_merged, keywords,
                       reinforce=False, auxloss_generate=False, **kwargs):
        sample_feature = features[list(features.keys())[0]]
        G = sample_feature.shape[1]
        B = features[list(features.keys())[0]].shape[0]  # B
        device = features[list(features.keys())[0]].device

        reg_loss = None,
        stats = {}
        training = self.training
        self.eval()

        hypo = []
        with torch.no_grad():
            for g in range(G):
                group_features = {k: v[:, g] for k, v in features.items()}
                hypo_context = self.get_hypo_context(hypo, B, G, device)
                group_hypo = self.run_group_generation(B, device,
                                                       group_features, hypo_context,
                                                       **kwargs)
                hypo.append(group_hypo)
        hypo = pad_tensor(hypo, val=self.tokenizer.pad_id).long()
        hypo = rearrange(hypo, 'g b l -> b g l')

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

        hypo = hypo[:, :, 1:]  # remove sos
        # hypo = rearrange(hypo.contiguous(), '(b g) l -> b g l', g=G)

        return hypo, logits, reg_loss, stats

    def run_group_generation(self, B, device, features, hypo_context,
                                sampler=None, reduce_hypo=True,
                                length_normalize_beam=False, postprocess_duplicates=1,
                                sample_ema_coeff=0, sample_eval_at_last=False,
                                **kwargs):
        hypos_fin = defaultdict(list)
        # log_probs_fin = defaultdict(list)
        term_log_probs_fin = defaultdict(list)
        curr_log_probs = None
        G = 1

        if sample_eval_at_last:
            sample_ema_coeff = 1

        empty = torch.full((B, self.num_samples, self.vocab_size), float('-inf')).to(device)
        s0 = torch.Tensor([self.tokenizer.cls_id]).long().to(device).expand(B)
        s0 = s0.unsqueeze(-1)
        s = s0
        hypo = s0.unsqueeze(-1)
        eos_flags_empty = torch.LongTensor([0] * B).bool().to(device).unsqueeze(1).expand(B, self.num_samples)
        eos_flags = eos_flags_empty.clone()
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
                        {'hypo': hypo[:, n].to(device).unsqueeze(1),  # add G dim
                         'hypo_context': hypo_context},  # add G dim
                        features, None,
                        infer=True
                    )
                    logits.append(logit[:, -1])  # get last token
                logits = torch.stack(logits, dim=1)  # BNV
                if torch.isnan(logits).any():
                    import ipdb; ipdb.set_trace()  # XXX DEBUG
                if getattr(logit, 'if_log_softmax', False):
                    logits.if_log_softmax = logit.if_log_softmax
                if not getattr(logits, 'if_log_softmax', False):
                    log_prob = F.log_softmax(logits, dim=-1)
                else:
                    log_prob = logits
                cumul_log_prob = log_prob
                if curr_log_probs is not None:
                    cumul_log_prob = sample_ema_coeff * cumul_log_prob + (1 - sample_ema_coeff) * curr_log_probs.unsqueeze(-1)
                s, token_probs, hypo_keys = sampler(log_prob, cumul_log_prob=cumul_log_prob)  # BN, BN, BN
                token_log_probs = token_probs.log()
                if curr_log_probs is None:
                    curr_log_probs = token_log_probs
                    # log_probs = token_log_probs.unsqueeze(-1)  # BNL
                else:
                    curr_log_probs = sample_ema_coeff * token_log_probs + (1 - sample_ema_coeff) * curr_log_probs
                    # curr_log_probs = curr_log_probs + token_log_probs
                    '''
                    log_probs = log_probs.gather(dim=1, index=hypo_keys.unsqueeze(-1).repeat(1, 1, log_probs.shape[-1]))
                    log_probs = torch.cat((log_probs, token_log_probs.unsqueeze(-1)), dim=-1)
                    '''
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
                    # hypos: BNL, curr_log_probs: BN
                    # store
                    for b in range(eos_flags.shape[0]):
                        if eos_flags[b].any():  # N
                            L = hypo[b].shape[1]
                            idx = eos_flags[b].nonzero().squeeze(-1)  # I
                            if idx.max() >= hypo[b].shape[0]:
                                import ipdb; ipdb.set_trace()  # XXX DEBUG
                            hypo_fin = hypo[b].gather(dim=0,
                                                    index=idx.unsqueeze(-1).repeat(1, L))  # IL
                            if idx.max() >= curr_log_probs[b].shape[0]:
                                import ipdb; ipdb.set_trace()  # XXX DEBUG
                            term_log_prob_fin = curr_log_probs[b].gather(dim=0, index=idx)
                            if not eos_batch[b].all():
                                hypos_fin[b].append(hypo_fin)
                                term_log_probs_fin[b].append(term_log_prob_fin)

                    # fill
                    curr_log_probs = (~eos_flags).float() * curr_log_probs + \
                        eos_flags.float() * self.negative_large # inf out finished sentence
                    best_idx = curr_log_probs.argmax(dim=-1).unsqueeze(1)  # B1
                    # B1L
                    best_hypo = hypo.gather(dim=1,
                                            index=best_idx.unsqueeze(-1).repeat(1, 1, hypo.shape[-1]))
                    # B1
                    best_prob = curr_log_probs.gather(dim=1,
                                            index=best_idx)
                    mask = eos_flags
                    hypo_mask = mask.unsqueeze(-1).repeat(1, 1, hypo.shape[-1])
                    best_hypo = best_hypo.repeat(1, hypo.shape[1], 1)
                    best_prob = best_prob.repeat(1, curr_log_probs.shape[1])
                    hypo_mask = hypo_mask.bool()
                    hypo = hypo * (~hypo_mask).long() + best_hypo * hypo_mask.long()
                    curr_log_probs = curr_log_probs * (~mask).float() + best_prob * mask.float()
                    if not eos_flags.all():
                        eos_flags = eos_flags_empty

        for b in range(hypo.shape[0]):
            if len(hypos_fin[b]) == 0:
                hypos_fin[b].append(hypo[b])
                term_log_probs_fin[b].append(curr_log_probs[b])

        if reduce_hypo:
            hypos_fin = list(sorted(hypos_fin.items()))
            hypos_fin = [list(chain(*[t.split(1, dim=0) for t in v])) for b, v in hypos_fin]
            hypo_lengths = [torch.LongTensor([t.shape[1] for t in v]).to(device) for v in hypos_fin]
            hypos_fin = [pad_tensor(v, val=self.tokenizer.pad_id).long() for v in hypos_fin]  # BIL
            hypos_fin = [v.squeeze(1) for v in hypos_fin]
            term_log_probs_fin = [torch.cat(v, dim=0) for b, v in sorted(term_log_probs_fin.items())]  # BI

            if length_normalize_beam:
                term_log_probs_fin = [v / l.float() for v, l in zip(term_log_probs_fin, hypo_lengths)]  # BI

            if postprocess_duplicates != 1:
                duplicate_threshold = 0.5
                # penalty duplicate
                best_probs = [p.topk(min(postprocess_duplicates, p.shape[0])) for p in term_log_probs_fin]
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
                best_ids = [p.argmax() for p in term_log_probs_fin]
                hypo = [v[i] for i, v in zip(best_ids, hypos_fin)]
            hypo = pad_tensor(hypo, val=self.tokenizer.pad_id).long()

        return hypo

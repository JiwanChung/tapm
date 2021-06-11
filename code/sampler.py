from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from entmax import sparsemax, entmax15, entmax_bisect

from exp import ex


def get_normalizer(normalizer_sparsity):
    sparsity = normalizer_sparsity
    sparsity = '1' if sparsity is None else sparsity
    sparsity = str(sparsity)
    eps = 1e-4
    probers = {
        '1': partial(entmax_bisect, alpha=1 + eps),
        # '1': F.softmax,
        '2': sparsemax,
        '1.5': entmax15,
        'adaptive': entmax_bisect,
    }
    prober = probers[sparsity]
    if sparsity == 'adaptive':
        f = prober
    else:
        def f(hypo, dim, alpha=None):
            return prober(hypo, dim=dim)

    return f, sparsity


@ex.capture
def get_sampler(model, sampling_method, num_samples):
    if sampling_method == 'beam':
        sampling_method = 'greedy'
        if num_samples == 1:
            num_samples = 3
        # print(f"beam search with beam size : {num_samples}")
    sampler = {
        'greedy': GreedySampler,
        'topk': TopKSampler,
        'nucleus': NucleusSampler,
        'max_nucleus': MaxNucleusSampler,
    }[sampling_method]()

    sample_class_dt = {
        'caption': CaptionSampler,
        'caption_single': CaptionSingleSampler,
    }

    sample_class = sampler
    if hasattr(model, 'model_type') and \
            model.model_type in sample_class_dt:
        sample_class = sample_class_dt[model.model_type]
    return sample_class(model, sampler,
                        num_samples=num_samples, sampling_method=sampling_method)


'''
class Sampler(nn.Module):
    @ex.capture
    def __init__(self, model, sampler,
                 max_target_len, num_samples):
        super(Sampler, self).__init__()

        self.model = model
        self.max_target_len = max_target_len
        self.num_samples = num_samples
        self.truncate = sampler

    def forward(self, keywords):
        if torch.is_tensor(keywords):
            keywords = keywords.detach()
        with torch.no_grad():
            B = keywords.shape[0]
            res = []
            inputs = self.model.encode(keywords)
            for i in range(B):
                single_input = [t[i].unsqueeze(0) if torch.is_tensor(t) else t for t in inputs]
                res.append(self.sample_single_source(*single_input))
        return res

    def sample_single_source(self, h_0, *args):
        hypo = torch.Tensor([self.model.tokenizer.cls_id]).long().to(h_0.device).unsqueeze(0)  # 11
        h = h_0
        for i in range(1, self.max_target_len):
            logits, h = self.model.decode(hypo, h.contiguous(), *args)  # KL
            logits = logits[:, -1]  # KV (get last token)
            probs = get_normalizer(logits, dim=-1)  # KV
            idx = self.truncate(probs)  # KV -> K'2 (K, V)
            probs = probs[idx[:, 0], idx[:, 1]]  # K'
            probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize
            sample = torch.multinomial(probs, self.num_samples)  # N
            idx = idx[sample]  # N2
            hypo = hypo[idx[:, 0]]
            tokens = idx[:, 1]
            if self.model.update_h:
                h = h[:, idx[:, 0]]  # (layers)NC
            hypo = torch.cat((hypo, tokens.unsqueeze(-1)), dim=1)

        return hypo[probs.argmax(dim=-1)]
'''


class CaptionSampler(nn.Module):
    @ex.capture
    def __init__(self, model, sampler,
                 max_target_len, num_samples, sampling_method,
                 length_normalize_beam=False,
                 sample_ema_coeff=0,
                 sample_eval_at_last=False):
        super(CaptionSampler, self).__init__()
        self.model = model
        self.max_target_len = max_target_len
        self.num_samples = num_samples
        self.sampling_method = sampling_method
        self.length_normalize_beam = length_normalize_beam
        self.sample_ema_coeff = sample_ema_coeff
        self.sample_eval_at_last = sample_eval_at_last

        # token_sampler = TokenSampler if self.sampling_method == 'greedy' else FasterGreedyTokenSampler
        token_sampler = FasterGreedyTokenSampler if self.sampling_method == 'greedy' else TokenSampler
        self.sampler = token_sampler(False, sampler)

    def forward(self, batch, **kwargs):
        if self.sampling_method == 'greedy':
            return self.forward_faster_greedy(batch, **kwargs)
        else:
            return self.forward_base(batch, **kwargs)

    def forward_base(self, batch):  # we only use features
        video = batch.video
        B, V = video.shape[:2]
        res = []
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.model.feature_names}.items()}
        keywords, reg_loss, stats = self.model.get_keyword(batch, features)
        keywords = keywords.detach()
        if self.model.use_gt_keywords:
            if not self.model.use_word_subset:
                keywords = batch.keyword_masks.float()
            else:
                keywords = batch.word_subsets.float()
        with torch.no_grad():
            for i in range(B):
                vid = []
                for v in range(V):
                    feature = {k: val[i, v].unsqueeze(0) for k, val in features.items()}
                    c = self.model.rnn.init_c(B, self.model.context_dim, device=video.device)
                    keyword = keywords[i, v].unsqueeze(0) if keywords is not None else None
                    group_mask = batch.group_mask[i, v].unsqueeze(0)
                    c, _, hypo, _, _ = self.model.run_video(feature, c, v,
                                                      self.max_target_len,
                                                      sentences=None,
                                                      sampler=self.sampler,
                                                      keyword=keyword,
                                                      group_mask=group_mask)
                    vid.append(hypo)
                res.append(vid)
        return res, stats

    def forward_faster_greedy(self, batch):
        video = batch.video
        B, V = video.shape[:2]
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.model.feature_names}.items()}
        keywords, reg_loss, stats = self.model.get_keyword(batch, features)
        keywords = keywords.detach()
        if self.model.use_gt_keywords:
            if not self.model.use_word_subset:
                keywords = batch.keyword_masks.float()
            else:
                keywords = batch.word_subsets.float()
        with torch.no_grad():
            vid = []
            for v in range(V):
                feature = {k: val[:, v] for k, val in features.items()}
                c = self.model.rnn.init_c(B, self.model.context_dim, device=video.device)
                keyword = keywords[:, v] if keywords is not None else None
                group_mask = batch.group_mask[:, v]
                c, _, hypo, _, _ = self.model.run_video(feature, c, v,
                                                    self.max_target_len,
                                                    sentences=None,
                                                    sampler=self.sampler,
                                                    keyword=keyword,
                                                    reduce_hypo=False,
                                                    group_mask=group_mask)
                vid.append(hypo)
        res = []
        for i in range(B):
            vid_res = []
            for v in range(V):
                vid_res.append(vid[v][i])
            res.append(vid_res)

        return res, stats


class CaptionSingleSampler(CaptionSampler):
    @ex.capture
    def __init__(self, model, sampler, *args, **kwargs):
        super(CaptionSingleSampler, self).__init__(model, sampler, *args, **kwargs)

        token_sampler = FasterGreedyTokenSampler if self.sampling_method == 'greedy' else TokenSampler
        self.sampler = token_sampler(True, sampler)

    def forward_base(self, batch, **kwargs):
        hypo, logits, _, _, stats, _ = self.model(batch, sampler=self.sampler,
                                                  length_normalize_beam=self.length_normalize_beam,
                                                  sample_ema_coeff=self.sample_ema_coeff,
                                                  sample_eval_at_last=self.sample_eval_at_last,
                                                    **kwargs)
        return hypo, logits, stats

    def forward_faster_greedy(self, batch, **kwargs):
        hypo, logits, _, _, stats, _ = self.model(batch,
                                            sampler=self.sampler,
                                            reduce_hypo=True,
                                                  length_normalize_beam=self.length_normalize_beam,
                                                  sample_ema_coeff=self.sample_ema_coeff,
                                                  sample_eval_at_last=self.sample_eval_at_last,
                                                    **kwargs)
        return hypo, logits, stats


class TokenSampler:
    @ex.capture
    def __init__(self, single, sampler,
                 num_samples,
                 normalizer_sparsity=None):
        self.forward = self.sample_token_single if single else self.sample_token

        self.normalizer, sparsity = get_normalizer(normalizer_sparsity)
        self.normalizer.sparsity = sparsity
        self.truncate = sampler
        self.num_samples = num_samples

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def sample_token(self, logits, alpha=None):
        logits = logits[:, -1]  # KV (get last token)
        probs = self.normalizer(logits, dim=-1, alpha=alpha)  # KV
        idx = self.truncate(probs)  # KV -> K'2 (K, V)
        probs = probs[idx[:, 0], idx[:, 1]]  # K'
        probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize
        sample = torch.multinomial(probs, self.num_samples)  # N
        idx = idx[sample]  # N2
        tokens = idx[:, 1]
        return tokens, probs

    def sample_token_single(self, logits, cumul_log_prob=None, cumul_prob=None, alpha=None):
        B = logits.shape[0]
        V = logits.shape[-1]
        if logits.dim() == 4:
            logits = logits[:, :, -1]  # BNV (get last token)
        if cumul_log_prob is None:
            if cumul_prob is None:
                cumul_prob = self.normalizer(logits, dim=-1, alpha=alpha)  # BNV
            probs = cumul_prob.clone()
        else:
            cumul_prob = cumul_log_prob.exp()
            probs = self.normalizer(logits, dim=-1, alpha=alpha)
        idx = self.truncate(cumul_prob)  # BNV -> BK (n * V + v)
        cumul_prob = cumul_prob.view(B, -1).contiguous()  # B(NV)
        cumul_prob = cumul_prob.gather(dim=-1, index=idx)  # BK
        cumul_prob = cumul_prob / cumul_prob.sum(dim=-1, keepdim=True)  # renormalize
        sample = torch.multinomial(cumul_prob, self.num_samples)  # BN
        idx = idx.gather(dim=-1, index=sample)
        probs = rearrange(probs, 'b n v -> b (n v)').gather(dim=-1, index=idx)
        idx = torch.stack((idx // V, idx % V), dim=-1)  # BN2
        # idx = idx[sample]  # BN2
        tokens = idx[:, :, 1]
        hypo_keys = idx[:, :, 0]
        return tokens, probs, hypo_keys


class FasterGreedyTokenSampler(TokenSampler):
    def sample_token(self, logits, alpha=None):
        logits = logits[:, -1]  # BV (get last token)
        probs = self.normalizer(logits, dim=-1, alpha=alpha)  # BV
        tokens = probs.argmax(dim=-1)  # B
        return tokens, probs

    def sample_token_single(self, logits, cumul_log_prob=None, cumul_prob=None, alpha=None):
        # greedy with num_samples > 2 is a BEAM search
        if logits.dim() == 4:
            logits = logits[:, :, -1]  # BNV (get last token)
        if cumul_log_prob is None:
            if cumul_prob is None:
                cumul_prob = self.normalizer(logits, dim=-1, alpha=alpha)  # BNV
            probs = cumul_prob.clone()
        else:
            cumul_prob = cumul_log_prob.exp()
            probs = self.normalizer(logits, dim=-1, alpha=alpha)
        V = cumul_prob.shape[-1]
        cumul_prob = rearrange(cumul_prob, 'b n v -> b (n v)')  # B(NV)
        _, idx = cumul_prob.topk(self.num_samples, dim=-1, largest=True)
        probs = rearrange(probs, 'b n v -> b (n v)').gather(dim=-1, index=idx)
        hypo_keys = idx // V
        tokens = idx % V
        return tokens, probs, hypo_keys


class TopKSampler:
    @ex.capture
    def __init__(self, sampling_k):
        self.k = sampling_k

    def __call__(self, probs):
        # BKV
        B = probs.shape[0]
        V = probs.shape[1]
        idx = probs.contiguous().view(B, -1).topk(self.k, dim=-1)[1]
        # idx = torch.stack((idx // V, idx % V), dim=-1)
        return idx


class GreedySampler(TopKSampler):
    def __init__(self):
        self.k = 1


class NucleusSampler:
    @ex.capture
    def __init__(self, sampling_p):
        self.p = sampling_p

    def __call__(self, probs):
        # get min card set with cumul prob > self.p
        B = probs.shape[0]
        V = probs.shape[-1]
        probs, ids = probs.contiguous().view(B, -1).sort(dim=-1, descending=True)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # pick first index with cumul prob > self.p
        probs = probs.cumsum(dim=-1)
        mask = probs >= self.p
        res = []
        for i in range(B):
            sentinel = mask[i].nonzero()[0].item()

            # cut idx
            idx = ids[i, :sentinel + 1]
            idx = torch.stack((idx // V, idx % V), dim=-1)
            res.append(idx)
        return res


class MaxNucleusSampler(NucleusSampler):
    def __call__(self, probs):
        # get min card set with cumul prob > self.p
        B = probs.shape[0]
        V = probs.shape[-1]
        probs, ids = probs.contiguous().view(B, -1).sort(dim=-1, descending=True)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # pick first index with cumul prob > self.p
        probs = probs.cumsum(dim=-1)
        mask = probs >= self.p
        # get max length
        mask, _ = mask.min(dim=0)  # V
        sentinel = mask.nonzero()[0].item()

        # cut idx
        idx = ids[:, :sentinel + 1]

        # idx = torch.stack((idx // V, idx % V), dim=-1)
        return idx

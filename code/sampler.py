from munch import Munch

import torch
from torch import nn
import torch.nn.functional as F


def get_sampler(args, model):
    sampler =  {
        'greedy': GreedySampler,
        'topk': TopKSampler,
        'nucleus': NucleusSampler
    }[args.sampling_method](args)

    sample_class = CaptionSampler if hasattr(model, 'model_type') and \
        model.model_type == 'caption' else Sampler
    return sample_class(args, model, sampler)


class Sampler(nn.Module):
    def __init__(self, args, model, sampler):
        super(Sampler, self).__init__()

        self.model = model
        self.max_target_len = args.max_target_len
        self.num_samples = args.get('num_samples', 1)
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
            probs = F.softmax(logits, dim=-1)  # KV
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


class CaptionSampler(nn.Module):
    def __init__(self, args, model, sampler):
        super(CaptionSampler, self).__init__()

        self.model = model
        self.max_target_len = args.max_target_len
        self.num_samples = args.get('num_samples', 1)
        self.sampling_method = args.sampling_method
        self.truncate = sampler

    def forward(self, batch):
        if self.sampling_method == 'greedy':
            return self.forward_faster_greedy(batch)
        else:
            return self.forward_base(batch)

    def forward_base(self, batch):  # we only use features
        video = batch.video
        B, V = video.shape[:2]
        res = []
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.model.feature_names}.items()}
        keywords, reg_loss, stats = self.model.get_keyword(batch, features)
        with torch.no_grad():
            for i in range(B):
                vid = []
                for v in range(V):
                    feature = {k: val[i, v].unsqueeze(0) for k, val in features.items()}
                    c = self.model.rnn.init_c(B, self.model.context_dim, device=video.device)
                    keyword = keywords[i, v].unsqueeze(0) if keywords is not None else None
                    c, _, hypo = self.model.run_video(feature, c, v,
                                                      self.max_target_len,
                                                      sentences=None,
                                                      sampler=self.sample_token,
                                                      keyword=keyword)
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
        with torch.no_grad():
            vid = []
            for v in range(V):
                feature = {k: val[:, v] for k, val in features.items()}
                c = self.model.rnn.init_c(B, self.model.context_dim, device=video.device)
                keyword = keywords[:, v] if keywords is not None else None
                c, _, hypo = self.model.run_video(feature, c, v,
                                                    self.max_target_len,
                                                    sentences=None,
                                                    sampler=self.sample_token_faster_greedy,
                                                    keyword=keyword,
                                                    reduce_hypo=False)
                vid.append(hypo)
        res = []
        for i in range(B):
            vid_res = []
            for v in range(V):
                vid_res.append(vid[v][i])
            res.append(vid_res)

        return res, stats

    def sample_token(self, logits):
        logits = logits[:, -1]  # KV (get last token)
        probs = F.softmax(logits, dim=-1)  # KV
        idx = self.truncate(probs)  # KV -> K'2 (K, V)
        probs = probs[idx[:, 0], idx[:, 1]]  # K'
        probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize
        sample = torch.multinomial(probs, self.num_samples)  # N
        idx = idx[sample]  # N2
        tokens = idx[:, 1]
        return tokens, probs

    def sample_token_faster_greedy(self, logits):
        logits = logits[:, -1]  # BV (get last token)
        probs = F.softmax(logits, dim=-1)  # BV
        tokens = probs.argmax(dim=-1)  # B
        return tokens, probs


class TopKSampler:
    def __init__(self, args):
        self.k = args.get('sampling_k', 10)

    def __call__(self, probs):
        # KV
        V = probs.shape[1]
        idx = probs.contiguous().view(-1).topk(self.k, dim=-1)[1]
        idx = torch.stack((idx // V, idx % V), dim=-1)
        return idx


class GreedySampler(TopKSampler):
    def __init__(self, args):
        self.k = 1


class NucleusSampler:
    def __init__(self, args):
        self.p = args.get('sampling_p', 0.9)

    def __call__(self, probs):
        # get min card set with cumul prob > self.p
        V = probs.shape[1]
        probs, idx = probs.contiguous().view(-1).sort(dim=-1, descending=True)
        probs = probs / probs.sum(dim=-1)

        # pick first index with cumul prob > self.p
        probs = probs.cumsum(dim=-1)
        sentinel = (probs >= self.p).nonzero()[0].item()

        # cut idx
        idx = idx[:sentinel + 1]
        idx = torch.stack((idx // V, idx % V), dim=-1)
        return idx

# TODO: implement top-k sampling, nucleus sampling, etc.
import torch
from torch import nn
import torch.nn.functional as F


def get_sampler(args, model):
    return {
        'greedy': GreedySampler,
        'topk': TopKSampler,
        'nucleus': NucleusSampler
    }[args.sampling_method](args, model)


class Sampler(nn.Module):
    def __init__(self, args, model):
        super(Sampler, self).__init__()

        self.model = model
        self.max_target_len = args.max_target_len

    def forward(self, keywords):
        B = keywords.shape[0]
        res = []
        h_0 = self.model.encode(keywords)
        for i in range(B):
            res.append(self.sample_single_source(h_0[:, i]))
        return res

    def sample_single_source(self, h_0):
        hypo = torch.Tensor([self.model.tokenizer.cls_id]).unsqueeze(-1).long().to(h_0.device).unsqueeze(0)  # 11
        h = h_0.unsqueeze(1)
        s_probs = torch.ones(hypo.shape).squeeze(-1).float()
        s_probs = s_probs / s_probs.sum(dim=-1, keepdim=True)
        for i in range(1, self.max_target_len):
            # LK
            s = hypo  # KL
            s = self.model.wte(s)
            logits, h = self.model.decode(s, h)  # KL
            logits = logits[:, -1]  # KV (get last token)
            s_probs = s_probs.unsqueeze(dim=-1) * logits  # KV
            idx = self.sample(s_probs, logits)  # K KV -> K'2 (K, V)
            s_probs = s_probs[idx[:, 0], idx[:, 1]]
            hypo = hypo[idx[:, 0]]
            tokens = idx[:, 1]
            hypo = torch.cat((hypo, tokens.unsqueeze(-1)), dim=1)
            s_probs = s_probs / s_probs.sum(dim=-1, keepdim=True)  # renormalize

        return hypo[s_probs.argmax(dim=-1)]


class TopKSampler(Sampler):
    def __init__(self, args, model):
        super(TopKSampler, self).__init__(args, model)

        self.k = args.get('sampling_k', 10)

    def sample(self, prev_probs, logits):
        probs = F.softmax(logits, dim=-1)
        probs = prev_probs.unsqueeze(-1) * probs
        return probs.topk(self.k, dim=-1)[1]


class GreedySampler(TopKSampler):
    def __init__(self, args, model):
        super(GreedySampler, self).__init__(args, model)

        self.k = 1


class NucleusSampler(Sampler):
    def __init__(self, args, model):
        super(NucleusSampler, self).__init__(args, model)

        self.p = args.get('sampling_p', 0.9)

    def sample(self, prev_probs, logits):
        probs = F.softmax(logits, dim=-1)
        probs = prev_probs.unsqueeze(-1) * probs
        return (probs >= self.p).nonzero()

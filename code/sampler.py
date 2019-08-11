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
        self.num_samples = args.get('num_samples', 1)

    def forward(self, keywords):
        keywords = keywords.detach()
        with torch.no_grad():
            B = keywords.shape[0]
            res = []
            h_0 = self.model.encode(keywords)
            for i in range(B):
                res.append(self.sample_single_source(h_0[:, i]))
        return res

    def sample_single_source(self, h_0):
        hypo = torch.Tensor([self.model.tokenizer.cls_id]).long().to(h_0.device).unsqueeze(0)  # 11
        h = h_0.unsqueeze(1)
        for i in range(1, self.max_target_len):
            s = self.model.wte(hypo)  # KL
            logits, h = self.model.decode(s, h.contiguous())  # KL
            logits = logits[:, -1]  # KV (get last token)
            probs = F.softmax(logits, dim=-1)  # KV
            idx = self.truncate(probs)  # KV -> K'2 (K, V)
            probs = probs[idx[:, 0], idx[:, 1]]  # K'
            probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize
            sample = torch.multinomial(probs, self.num_samples)  # N
            idx = idx[sample]  # N2
            hypo = hypo[idx[:, 0]]
            tokens = idx[:, 1]
            h = h[:, idx[:, 0]]  # (layers)NC
            hypo = torch.cat((hypo, tokens.unsqueeze(-1)), dim=1)

        return hypo[probs.argmax(dim=-1)]


class TopKSampler(Sampler):
    def __init__(self, args, model):
        super(TopKSampler, self).__init__(args, model)

        self.k = args.get('sampling_k', 10)

    def truncate(self, probs):
        # KV
        V = probs.shape[1]
        idx = probs.contiguous().view(-1).topk(self.k, dim=-1)[1]
        idx = torch.stack((idx // V, idx % V), dim=-1)
        return idx


class GreedySampler(TopKSampler):
    def __init__(self, args, model):
        super(GreedySampler, self).__init__(args, model)

        self.k = 1


class NucleusSampler(Sampler):
    def __init__(self, args, model):
        super(NucleusSampler, self).__init__(args, model)

        self.p = args.get('sampling_p', 0.9)

    def truncate(self, probs):
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

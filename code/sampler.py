# TODO: implement top-k sampling, nucleus sampling, etc.
import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self, args, model):
        super(Sampler, self).__init__()

        self.model = model
        self.max_target_len = args.max_target_len

    def sample(self, x):
        return x.argmax(dim=-1)  # greedy

    def forward(self, keywords):
        h_0 = self.model.encode(keywords)

        s_0 = torch.Tensor([self.model.tokenizer.cls_id]).long().to(h_0.device)
        s_0 = s_0.unsqueeze(-1).expand(keywords.shape[0], 1)

        s = s_0
        h = h_0
        res = torch.full((keywords.shape[0], self.max_target_len), self.model.tokenizer.pad_id).long().to(keywords.device)
        eos_all = torch.zeros(keywords.shape[0]).byte().to(keywords.device)
        res[:, 0] = s
        for i in range(self.max_target_len - 1):
            s = self.model.wte(s)
            logits, h = self.model.decode(s, h)
            tokens = self.sample(logits).squeeze(-1)
            res[:, i + 1] = tokens
            eos_all = eos_all | (tokens == self.model.tokenizer.sep_id).byte()
            if eos_all.all():
                break
            s = tokens.unsqueeze(-1)  # B -> B1
        return res

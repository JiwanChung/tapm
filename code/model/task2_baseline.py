import math

import torch
from torch import nn
import torch.nn.functional as F

from utils import mean
from data.batcher import make_blank_filling_batch, pad_tensor
from .transformer_model import TransformerModel


class Task2Baseline(TransformerModel):
    transformer_name = 'gpt2'  # assign transformer_name = 'bert' to use BPE
    model_type = 'caption'
    use_keyword = True

    def make_batch(self, *args, **kwargs):
        return make_blank_filling_batch(*args, **kwargs)

    def __init__(self, args, transformer, tokenizer):
        super(Task2Baseline, self).__init__()

        self.dropout_ratio = args.get('dropout', 0.5)

        self.tokenizer = tokenizer

        self.net = transformer
        self.net.train()
        self.gpt_dim = self.net.transformer.config.n_embd

    def get_blanks(self, t, mask):
        # BL(C), BL
        res = []
        for b in range(t.shape[0]):
            # L(C), L
            m = mask[b]
            m = m.view(m.shape[0], *[1 for i in range(t.dim() - 2)]).contiguous()
            x = t[b].masked_select(m)
            if t.dim() > 2:
                x = x.view(-1, t.shape[-1]).contiguous()
            res.append(x)
        # list of length B: N(C)
        return res

    def get_acc(self, hypos, tgts):
        res = []
        for hypo, tgt in zip(hypos, tgts):
            if tgt.shape[0] > 1:  # 1x1 would be trivial
                hypo = torch.triu(hypo, diagonal=1)
                tgt = torch.triu(tgt, diagonal=1)
                acc = hypo == tgt
                mask = torch.triu(torch.ones_like(tgt).byte(), diagonal=1).float()
                acc = (acc.float() * mask).sum() / mask.sum()
                res.append(acc)
        return mean(res) if len(res) > 0 else None

    def cartesian(self, tgts):
        res = []
        for tgt in tgts:
            res.append(tgt.unsqueeze(0) == tgt.unsqueeze(-1))
        return res

    def get_relation(self, hypo, tgt):
        hypo = [str(h) for h in hypo]
        hypo = [f"\n{h[8:h.find('], device')]}" for h in hypo]
        tgt = [str(h) for h in tgt]
        tgt = [f"\n{h[8:h.find('], device')]}" for h in tgt]
        return [f"{h}\n---\n{t}" for h, t in zip(hypo, tgt)]

    def forward(self, batch, **kwargs):
        sentences = batch.sentences

        x = self.net.transformer(sentences)[0]
        logits = self.net.lm_head(x)

        blank_ids = batch.blank_ids
        reg_loss = self.get_loss(logits, batch.targets, blank_ids)

        stats = {'blank_loss': reg_loss.item(),
                 'blank_num': blank_ids.float().sum(dim=-1).mean().item()}

        with torch.no_grad():
            logit_argmax = logits.detach().argmax(dim=-1)
            blank_correct = (logit_argmax == batch.targets).masked_select(blank_ids)
            blank_words = [x.squeeze(0) for x in logit_argmax.split(1, dim=0)]
            masks = [x.squeeze(0) for x in blank_ids.split(1, dim=0)]
            blank_words = [x.masked_select(masks[i]) for i, x in enumerate(blank_words)]
            max_size = max([x.shape[0] for x in blank_words])
            storage = torch.full((len(blank_words), max_size), self.tokenizer.pad_id).to(blank_ids.device)
            for i, t in enumerate(blank_words):
                storage[i, :t.shape[0]] = t
            blank_words = storage

            target = self.get_blanks(batch.targets, blank_ids)
            target = self.cartesian(target)
            hypo = self.get_blanks(logit_argmax, blank_ids)
            hypo = self.cartesian(hypo)
            acc = self.get_acc(hypo, target)
            relation = self.get_relation(hypo, target)
            stats = {'blank_acc': acc.float().mean().item(), **stats}

        return None, batch.targets, reg_loss, stats, relation

    def get_loss(self, hypo, tgt, mask):
        hypo = hypo.contiguous()
        tgt = tgt.contiguous()
        mask = mask.contiguous()
        mask = mask.view(-1, 1)
        lprobs = F.log_softmax(hypo, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = tgt.view(-1, 1)
        nll_loss = -lprobs.gather(dim=-1, index=target)[mask]

        return nll_loss.mean()


class Task2Baseline2(Task2Baseline):
    def __init__(self, args, transformer, tokenizer):
        super(Task2Baseline2, self).__init__(args, transformer, tokenizer)

        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        small_dim = 128
        self.reduce_dim = nn.Linear(self.gpt_dim, small_dim)
        self.layer_norm = nn.LayerNorm(small_dim)

    def get_loss(self, hypos, tgts):
        res = []
        for hypo, tgt in zip(hypos, tgts):
            hypo = torch.triu(hypo, diagonal=1)
            tgt = torch.triu(tgt, diagonal=1)
            loss = self.loss(hypo, tgt.float())
            mask = torch.triu(torch.ones_like(tgt).byte(), diagonal=1).byte()
            loss = loss.masked_select(mask).sum()
            res.append(loss)
        return mean(res)

    def relation_net(self, hypos):
        res = []
        for hypo in hypos:
            # NC
            C = hypo.shape[-1]
            hypo = self.reduce_dim(hypo)
            hypo = self.layer_norm(hypo)
            hypo = torch.einsum('nic,imc->nm', hypo.unsqueeze(1), hypo.unsqueeze(0))
            res.append(hypo)
        return res

    def forward(self, batch, **kwargs):
        sentences = batch.sentences

        x = self.net.transformer(sentences)[0]

        # build nxn relation matrix
        blank_ids = batch.blank_ids
        target = self.get_blanks(batch.targets, blank_ids)
        target = self.cartesian(target)
        hypo = self.get_blanks(x, blank_ids)
        hypo = self.relation_net(hypo)

        reg_loss = self.get_loss(hypo, target)
        with torch.no_grad():
            hypo = [h >= 0.5 for h in hypo]
            acc = self.get_acc(hypo, target)
            relation = self.get_relation(hypo, target)

            stats = {'blank_loss': reg_loss.item(),
                    'blank_num': blank_ids.float().sum(dim=-1).mean().item(),
                    'blank_acc': acc.float().mean().item()}

        return None, batch.targets, reg_loss, stats, relation

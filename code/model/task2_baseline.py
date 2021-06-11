import math
from collections import OrderedDict
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from utils import mean
from data.batcher import make_blank_filling_batch, pad_tensor
from run_transformer import transformer_embed, transformer_run_cells
from .transformer_model import TransformerModel


class Task2Baseline(TransformerModel):
    transformer_name = 'gpt2'  # assign transformer_name = 'bert' to use BPE
    model_type = 'caption'
    use_keyword = True
    task = 2

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
        accs = []
        same_accs = []
        diff_accs = []
        for hypo, tgt in zip(hypos, tgts):
            if tgt.shape[0] > 1 and hypo is not None:  # 1x1 would be trivial
                hypo = torch.triu(hypo, diagonal=1)
                tgt = torch.triu(tgt, diagonal=1)
                acc = hypo == tgt
                mask = torch.triu(torch.ones_like(tgt).bool(), diagonal=1).float()
                same_mask = mask * tgt.float()
                diff_mask = mask * (~tgt.bool()).float()
                same_acc = (acc.float() * same_mask).sum() / same_mask.sum() if same_mask.sum() > 0 else None
                diff_acc = (acc.float() * diff_mask).sum() / diff_mask.sum() if diff_mask.sum() > 0 else None
                acc = (acc.float() * mask).sum() / mask.sum()
                accs.append(acc)
                same_accs.append(same_acc)
                diff_accs.append(diff_acc)
        accs = mean(accs) if len(accs) > 0 else None
        same_accs = mean(same_accs) if len(same_accs) > 0 else None
        diff_accs = mean(diff_accs) if len(diff_accs) > 0 else None

        return accs, same_accs, diff_accs

    def cartesian(self, tgts):
        res = []
        for tgt in tgts:
            res.append(tgt.unsqueeze(0) == tgt.unsqueeze(-1))
        return res

    def get_relation(self, hypo, tgt):
        res = []
        for h, t in zip(hypo, tgt):
            if h is not None:
                h = str(h)
                t = str(t)
                h = f"\n{h[8:h.find('], device')]}"
                t = f"\n{t[8:t.find('], device')]}"
                res.append(f"{h}\n---\n{t}")
            else:
                res.append(None)
        return res

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
            acc, same_acc, diff_acc = self.get_acc(hypo, target)
            relation = self.get_relation(hypo, target)
            stats = {'blank_acc': None if acc is None else acc.float().mean().item(),
                    'blank_same_acc': None if same_acc is None else same_acc.float().mean().item(),
                    'blank_diff_acc': None if diff_acc is None else diff_acc.float().mean().item(),
                     **stats}

        return None, batch.targets, reg_loss, stats, \
            {'text': relation, 'hypo': [h.detach().cpu().numpy() for h in hypo]}

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
            if hypo is not None:
                hypo = torch.triu(hypo, diagonal=1)
                tgt = torch.triu(tgt, diagonal=1)
                loss = self.loss(hypo, tgt.float())
                mask = torch.triu(torch.ones_like(tgt).bool(), diagonal=1).bool()
                loss_sum = loss.masked_select(mask).sum()
                if loss.nelement() > 0:
                    loss_sum = loss_sum / loss.nelement()
                res.append(loss_sum)
        return mean(res)

    def relation_net(self, hypos):
        res = []
        for hypo in hypos:
            # NC
            if hypo.nelement() != 0:
                hypo = self.reduce_dim(hypo)
                hypo = self.layer_norm(hypo)
                hypo = torch.einsum('nic,imc->nm', hypo.unsqueeze(1), hypo.unsqueeze(0))  # [-1, 1]
                res.append(hypo)
            else:
                res.append(None)
        return res

    def encode_sentence(self, batch):
        sentences = batch.sentences
        x = self.net.transformer(sentences)[0]
        return x

    def get_prob(self, hypo):
        return [torch.sigmoid(h) >= 0.5 if h is not None else None for h in hypo]

    def make_target(self, batch):
        blank_ids = batch.blank_ids
        target = self.get_blanks(batch.targets, blank_ids)
        target = self.cartesian(target)

        return target

    def forward(self, batch, **kwargs):
        x = self.encode_sentence(batch)

        # build nxn relation matrix
        blank_ids = batch.blank_ids
        target = self.make_target(batch)

        hypo = self.get_blanks(x, blank_ids)
        hypo = self.relation_net(hypo)

        reg_loss = self.get_loss(hypo, target)
        hypo_cont = [h.clone().detach() if torch.is_tensor(h) else h for h in hypo]
        with torch.no_grad():
            hypo = self.get_prob(hypo)
            acc, same_acc, diff_acc = self.get_acc(hypo, target)
            relation = self.get_relation(hypo, target)

            stats = {'blank_loss': reg_loss.item(),
                    'blank_num': blank_ids.float().sum(dim=-1).mean().item(),
                    'blank_acc': None if acc is None else acc.float().mean().item(),
                    'blank_same_acc': None if same_acc is None else same_acc.float().mean().item(),
                    'blank_diff_acc': None if diff_acc is None else diff_acc.float().mean().item()}

        return None, batch.targets, reg_loss, stats, \
            {'text': relation, 'hypo': [h.detach().cpu().numpy() if torch.is_tensor(h) else h for h in hypo_cont]}



class Task2BaselineCosSim(Task2Baseline2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self.layer_norm
        self.loss = nn.BCELoss(reduction='none')
        self.eps = 1e-12
        self.cossim_tolerance = 1e-4

    def get_prob(self, hypo):
        return [h >= 0.5 if h is not None else None for h in hypo]

    def relation_net(self, hypos):
        res = []
        for hypo in hypos:
            # NC
            if hypo.nelement() != 0:
                hypo = self.reduce_dim(hypo)
                hypo = F.normalize(hypo)
                hypo = torch.einsum('nic,imc->nm', hypo.unsqueeze(1), hypo.unsqueeze(0))  # [-1, 1]
                if hypo.max() > (1 + self.cossim_tolerance):
                    print('cossim max exceeded')
                    import ipdb; ipdb.set_trace()  # XXX DEBUG
                if hypo.min() < -(1 + self.cossim_tolerance):
                    print('cossim min exceeded')
                    import ipdb; ipdb.set_trace()  # XXX DEBUG
                assert hypo.max() <= 1 + self.cossim_tolerance \
                    and hypo.min() >= -(1 + self.cossim_tolerance), \
                    f'wrong hypo range, it should be within [-1, 1]'
                hypo = (hypo + 1 + self.eps) / 2  # [0, 1]
                res.append(hypo)
            else:
                res.append(None)
        return res


class Task2FeatureConcat(Task2Baseline2):
    def __init__(self, args, transformer, tokenizer):
        super(Task2FeatureConcat, self).__init__(args, transformer, tokenizer)

        self.loss = nn.BCELoss(reduction='none')
        self.feature_names = args.get('feature_names',
                                      ['video', 'image', 'flow', 'box'])
        self.feature_dims = {k: v for k, v in args.feature_dims.items() if k in self.feature_names}

        for feature in self.feature_names:
            setattr(self, feature, nn.Linear(self.feature_dims[feature], self.gpt_dim))

    def process_features(self, batch):
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.feature_names}.items()}
        res = {}
        for feature in self.feature_names:
            res[feature] = getattr(self, feature)(features[feature])

        return res

    def merge_context(self, features, cls_embd, sep_embd):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        return torch.cat((cls_embd, *chain(*[(rearrange(feature, 'b g l c -> b (g l) c'), sep_embd) for feature in features.values()])), dim=1)

    def encode_sentence(self, batch):
        features = self.process_features(batch)
        h, past, head_mask = transformer_embed(self.net.transformer, batch.sentences)

        cls_embd = self.net.transformer.wte(torch.LongTensor([self.tokenizer.cls_id]).to(h.device))
        sep_embd = self.net.transformer.wte(torch.LongTensor([self.tokenizer.sep_id]).to(h.device))
        B, L, C = h.shape
        cls_embd = cls_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        sep_embd = sep_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        context = self.merge_context(features, cls_embd, sep_embd)
        h = torch.cat((context, h), dim=1)

        o = transformer_run_cells(self.net.transformer, h, past=past, head_mask=head_mask)[0]
        context_embedded = o[:, :context.shape[1]]
        o = o[:, context.shape[1]:]

        return o

import math

import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import (
    make_mask_model_batch, make_subset_mask_batch,
    remove_pad
)

from .transformer_model import TransformerModel


class MaskModel(TransformerModel):
    transformer_name = 'bert'

    def __init__(self, args, transformer, tokenizer):
        super(MaskModel, self).__init__()

        self.complementary_mask = args.get('complementary_mask', False)

        self.transformer = transformer
        self.transformer.train()

        self.tokenizer = tokenizer

    def make_batch(self, *args, **kwargs):
        return make_mask_model_batch(*args, random_idx=self.training,
                                     complementary=self.complementary_mask, **kwargs)

    def forward(self, batch, **kwargs):
        if self.training:
            return self.forward_train(batch)
        else:
            return self.forward_eval(batch)

    def forward_eval(self, batch, **kwargs):
        sentences = batch.sentences
        targets = batch.targets
        mask_ids = batch.mask_ids
        # list of len(B) containing batch L*L
        loss_report = []
        scores = []
        ids = []
        for sentence, target in zip(sentences, targets):
            attention_mask = sentence != self.tokenizer.pad_id
            outputs = self.transformer(sentence, attention_mask=attention_mask)
            logits = outputs[0]
            # L*L*C
            L = logits.shape[0]
            C = logits.shape[-1]

            mask_ids = torch.arange(L).long().to(logits.device)
            target_resize = target[:logits.shape[0]]  # remove padding
            # LC, L
            if self.complementary_mask:
                idx_logit = logits
                idx_target = target_resize
                idx_target = idx_target.unsqueeze(0).expand(idx_target.shape[0], -1).contiguous()
            else:
                idx_logit = logits.gather(dim=1, index=mask_ids.contiguous().view(-1, 1, 1).expand(mask_ids.shape[0], 1, logits.shape[-1])).squeeze(1)
                idx_target = target_resize.gather(dim=0, index=mask_ids)

            losses = F.cross_entropy(idx_logit.contiguous().view(-1, C),
                                     idx_target.view(-1),
                                     reduction='none').contiguous().view(*idx_target.shape)
            with torch.no_grad():
                loss_report.append(losses.mean())
                # L
                probs = F.softmax(idx_logit, dim=-1)
                probs = probs.gather(dim=-1, index=idx_target.unsqueeze(-1)).squeeze(-1)

                probs = probs[1: -1]  # remove cls, sep
                losses = losses.detach()
                losses = losses[1: -1]  # remove cls, sep
                if len(probs.shape) > 1:
                    probs = probs[:, 1: -1]  # remove cls, sep
                    probs = probs.prod(dim=-1)
                if len(losses.shape) > 1:
                    losses = losses[:, 1: -1]  # remove cls, sep
                    losses = losses.sum(dim=-1)
                val, idx = losses.sort(dim=0, descending=False)
                idx = idx + 1  # remember cls
                idx = target[idx]
                ids.append(idx)
                scores.append(val)
        loss_report = torch.Tensor(loss_report).float().to(sentences[0].device).mean()
        return loss_report, scores, ids

    def forward_train(self, batch, **kwargs):
        sentence = batch.sentences
        targets = batch.targets
        mask_ids = batch.mask_ids
        attention_mask = sentence != self.tokenizer.pad_id
        outputs = self.transformer(sentence, attention_mask=attention_mask)
        logits = outputs[0]

        # BLC -> BC
        if self.complementary_mask:
            idx_logit = logits
            idx_target = targets
        else:
            mask_ids = mask_ids.unsqueeze(-1)
            idx_logit = logits.gather(dim=1, index=mask_ids.unsqueeze(-1).expand(mask_ids.shape[0], -1, logits.shape[-1])).squeeze(1)
            idx_target = targets.gather(dim=1, index=mask_ids).squeeze(1)

        with torch.no_grad():
            idx_prob = F.softmax(idx_logit.detach(), dim=-1)
            # BC -> B
            idx_prob = idx_prob.gather(dim=-1, index=idx_target.unsqueeze(-1)).squeeze(1)

            # BC -> B
            hypos = idx_logit.detach().argmax(dim=-1)
            hypos = torch.stack((hypos, idx_target), dim=-1)

        return idx_logit, idx_target, None, {'idx_prob': idx_prob.mean().item()}, None



class SubsetMaskModel(TransformerModel):
    transformer_name = 'bert'

    def __init__(self, args, transformer, tokenizer):
        super(SubsetMaskModel, self).__init__()

        self.extraction_min_words = args.extraction_min_words
        self.keyword_ratio = args.keyword_ratio

        self.net = transformer
        self.net.train()

        self.tokenizer = tokenizer

        bert_dim = self.net.bert.config.hidden_size

    def make_batch(self, *args, **kwargs):
        return make_subset_mask_batch(*args, random_idx=self.training, **kwargs)

    def forward(self, batch, **kwargs):
        if self.training:
            return self.forward_train(batch)
        else:
            return self.forward_eval(batch)

    def forward_train(self, batch, **kwargs):
        sentence = batch.sentences
        targets = batch.targets
        keyword_ids = batch.keyword_ids
        attention_mask = sentence != self.tokenizer.pad_id
        outputs = self.net(sentence, attention_mask=attention_mask)
        logits = outputs[0]

        with torch.no_grad():
            idx_prob = F.softmax(logits.detach(), dim=-1)
            # BC -> B
            idx_prob = idx_prob.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(1)

            # BC -> B
            hypos = logits.detach().argmax(dim=-1)
            hypos = torch.stack((hypos, targets), dim=-1)

        return logits, targets, None, {'idx_prob': idx_prob.mean().item()}, None

    '''
    def forward_eval(self, batch, **kwargs):
        sentences = batch.sentences
        targets = batch.targets
        keyword_ids = batch.keyword_ids
        # list of len(B) containing batch L*L
        scores = []
        loss_report = []
        ids = []
        for combs, sentence, target in zip(keyword_ids, sentences, targets):
            for comb_L in combs:
                sentence_comb = sentence.clone().unsqueeze(0).expand(comb_L.shape[0], -1)
                comb_mask = torch.zeros(*sentence_comb.shape).bool().to(sentence_comb.device)
                comb_mask.scatter_(dim=1, index=comb_L, src=1)
                sentence_comb = sentence_comb * comb_mask.long()

                attention_mask = sentence_comb != self.tokenizer.pad_id
                outputs = self.net(sentence_comb, attention_mask=attention_mask)
                logits = outputs[0]
                # B*L*C
                L = logits.shape[0]
                C = logits.shape[-1]

                # LC, L
                idx_logit = logits
                idx_target = target[:logits.shape[0]].unsqueeze(0).expand(logits.shape[0], -1).contiguous()

                losses = F.cross_entropy(idx_logit.contiguous().view(-1, C),
                                        idx_target.view(-1),
                                        reduction='none').contiguous().view(*idx_target.shape)
                # BL
                with torch.no_grad():
                    # BLV
                    probs = F.softmax(idx_logit, dim=-1)
                    probs = probs.gather(dim=-1, index=idx_target.unsqueeze(-1)).squeeze(-1)
                    probs = probs[:, 1: -1]  # remove cls, sep
                    # BL
                    losses = losses.detach()
                    losses = losses[:, 1: -1]  # remove cls, sep
                    losses = losses.mean(-1)  # B
                    i = losses.argmax(dim=0)
                    scores.append(losses[i])
                    ids.append(comb_L[i])
                    for i in range(losses.shape[0]):
                        loss_report.append(losses[i].item())

        loss_report = torch.Tensor(loss_report).float().to(sentences[0].device).mean()
        return loss_report, scores, ids
    '''

    def forward_eval(self, batch, **kwargs):
        sentences = batch.sentences
        targets = batch.targets
        lengths = batch.lengths
        scores = []
        loss_report = []
        ids = []
        for i, (sentence, target) in enumerate(zip(sentences, targets)):
            max_keyword_num = math.ceil(max(self.extraction_min_words, lengths[i].float().item() * self.keyword_ratio))
            storage = sentence.clone()
            storage = remove_pad(storage, self.tokenizer.pad_id)
            storage[1: -1] = self.tokenizer.mask_id  # mask except cls, sep
            row_scores = []
            row_ids = []
            prev_pos = []
            for j in range(max_keyword_num):
                # L -> LL
                x = storage.clone()
                x = x.unsqueeze(0).expand(x.shape[0], -1)
                scatter_mask = torch.eye(x.shape[0]).bool().to(x.device)
                x.masked_scatter_(scatter_mask, sentence)

                attention_mask = x != self.tokenizer.pad_id
                outputs = self.net(x, attention_mask=attention_mask)
                logits = outputs[0]
                C = logits.shape[-1]

                idx_logit = logits
                idx_target = target[:logits.shape[0]].unsqueeze(0).expand(logits.shape[0], -1).contiguous()

                losses = F.cross_entropy(idx_logit.contiguous().view(-1, C),
                                        idx_target.view(-1),
                                        reduction='none').contiguous().view(*idx_target.shape)
                # LL
                losses = losses.detach()
                losses = losses[:, 1: -1]  # remove cls, sep
                losses = losses.mean(-1)  # L
                # skip identity, cls, sep
                for idx in [0, -1, *prev_pos]:
                    losses[idx] = float('inf')
                idx = losses.argmin(dim=0)
                storage[idx] = sentence[idx]

                prev_pos.append(idx)
                row_scores.append(losses[idx])
                row_ids.append(sentence[idx])

            row_scores = torch.Tensor(row_scores).to(sentence.device).float()
            scores.append(row_scores)
            row_ids = torch.Tensor(row_ids).to(sentence.device).long()
            ids.append(row_ids)
            loss_report.append(row_scores.mean().item())

        loss_report = torch.Tensor(loss_report).float().to(sentences[0].device).mean()
        return loss_report, scores, ids

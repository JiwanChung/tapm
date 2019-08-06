import torch
from torch import nn
import torch.nn.functional as F


class MaskModel(nn.Module):
    transformer_name = 'bert'

    def __init__(self, args, transformer, tokenizer):
        super(MaskModel, self).__init__()

        self.bert = transformer
        self.bert.train()

        self.pad_id = tokenizer.pad_id

    def forward(self, sentence, lengths, mask_ids, target):
        if self.training:
            return self.forward_train(sentence, lengths, mask_ids, target)
        else:
            return self.forward_eval(sentence, lengths, mask_ids, target)

    def forward_eval(self, sentences, lengths, mask_ids, targets):
        # list of len(B) containing batch L*L
        loss_report = []
        scores = []
        ids = []
        for sentence, target in zip(sentences, targets):
            attention_mask = sentence != self.pad_id
            outputs = self.bert(sentence, attention_mask=attention_mask)
            logits = outputs[0]
            # L*L*C
            L = logits.shape[0]
            C = logits.shape[-1]
            target_resize = target[:logits.shape[0]]  # remove padding
            target_resize = target_resize.unsqueeze(0).expand(L, L)
            losses = F.cross_entropy(logits.contiguous().view(-1, C), target_resize.contiguous().view(-1),
                                     reduction='none').contiguous().view(L, L)
            loss_report.append(losses.mean())
            losses = losses.sum(dim=-1)
            # L
            losses = losses[1: -1]  # remove cls, sep
            val, idx = losses.sort(dim=0, descending=True)
            idx = idx + 1  # remember cls
            idx = target[idx]
            ids.append(idx)
            scores.append(val)
        loss_report = torch.Tensor(loss_report).float().to(sentences[0].device).mean()
        return loss_report, scores, ids

    def forward_train(self, sentence, lengths, mask_ids, targets):
        attention_mask = sentence != self.pad_id
        outputs = self.bert(sentence, attention_mask=attention_mask)
        logits = outputs[0]

        temp = logits.clone().detach()
        temp.requires_grad_(False)
        # BLC -> BC
        idx_logit = logits.gather(dim=1, index=mask_ids.contiguous().view(-1, 1, 1).expand(mask_ids.shape[0], 1, temp.shape[-1])).squeeze(1)
        idx_target = targets.gather(dim=1, index=mask_ids.unsqueeze(-1)).squeeze(1)

        with torch.no_grad():
            idx_prob = F.softmax(idx_logit.detach(), dim=-1)
            # BC -> B
            idx_prob = idx_prob.gather(dim=1, index=idx_target.unsqueeze(-1)).squeeze(1)

            # BC -> B
            hypos = idx_logit.detach().argmax(dim=-1)
            hypos = torch.stack((hypos, idx_target), dim=-1)

        return idx_logit, idx_target, None, idx_prob, None

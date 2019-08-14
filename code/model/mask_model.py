import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import make_mask_model_batch

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

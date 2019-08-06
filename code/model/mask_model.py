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
        attention_mask = sentence != self.pad_id
        outputs = self.bert(sentence, attention_mask=attention_mask)
        logits = outputs[0]

        temp = logits.clone().detach()
        temp.requires_grad_(False)
        with torch.no_grad():
            # BLC -> BC
            idx_logit = temp.gather(dim=1, index=mask_ids.contiguous().view(-1, 1, 1).expand(mask_ids.shape[0], 1, temp.shape[-1])).squeeze(1)
            idx_target = target.gather(dim=1, index=mask_ids.unsqueeze(-1)).squeeze(1)

            idx_prob = F.softmax(idx_logit, dim=-1)
            # BC -> B
            idx_prob = idx_prob.gather(dim=1, index=idx_target.unsqueeze(-1)).squeeze(1)

            # BC -> B
            hypos = idx_logit.argmax(dim=-1)
            hypos = torch.cat((hypos, idx_target), dim=-1)

        return logits, None, idx_prob, hypos

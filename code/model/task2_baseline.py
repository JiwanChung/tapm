import torch
from torch import nn
import torch.nn.functional as F

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

    def forward(self, batch, **kwargs):
        sentences = batch.sentences

        x = self.net.transformer(sentences)[0]
        logits = self.net.lm_head(x)

        blank_ids = sentences != batch.targets
        reg_loss = self.get_loss(logits, batch.targets, blank_ids)

        stats = {'blank_loss': reg_loss.item(),
                 'blank_num': blank_ids.float().sum(dim=-1).mean().item()}

        with torch.no_grad():
            blank_words = logits.detach().argmax(dim=-1)
            blank_words = [x.squeeze(0) for x in blank_words.split(1, dim=0)]
            masks = [x.squeeze(0) for x in blank_ids.split(1, dim=0)]
            blank_words = [x.masked_select(masks[i]) for i, x in enumerate(blank_words)]
            max_size = max([x.shape[0] for x in blank_words])
            storage = torch.full((len(blank_words), max_size), self.tokenizer.pad_id).to(blank_ids.device)
            for i, t in enumerate(blank_words):
                storage[i, :t.shape[0]] = t
            blank_words = storage

        return None, batch.targets, reg_loss, stats, blank_words

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

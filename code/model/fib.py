from collections import OrderedDict

from utils import mean
from data.batcher import make_fib_batch

from .pretrain_aux import PretrainAuxGroupRoll


class FibModel(PretrainAuxGroupRoll):
    transformer_name = 'bert_cased'

    def make_batch(self, *args, **kwargs):
        return make_fib_batch(*args, **kwargs)

    def _forward(self, batch, **kwargs):
        if batch.sentences.shape != batch.targets.shape:
            import ipdb; ipdb.set_trace()
        _, logits, targets, reg_loss, stats, batch = super()._forward(batch, **kwargs)
        source = batch['sentences']
        targets = self.mask_out_blank(targets, source)

        logits = logits.view(targets.shape[0], targets.shape[1], logits.shape[1], -1)
        hypo = logits.argmax(dim=-1)  # BGL

        stats['accuracy'], answers = self.get_blank_accuracy(hypo, source,
                                                    batch.answer, batch.group_mask)
        stats['num_samples'] = batch.group_mask.sum().item()
        batch.answers = answers
        return hypo, logits, targets, reg_loss, stats, batch

    def mask_out_blank(self, target, source):
        mask = source != self.tokenizer.mask_id
        target[mask] = self.tokenizer.pad_id  # ignore non masks
        return target

    def get_blank_accuracy(self, hypo, source, answer, group_mask):
        B, G, L = hypo.shape
        matches = []
        answers = []
        for b in range(B):
            for g in range(G):
                if group_mask[b, g] > 0:
                    blank_ids = (source[b, g] == self.tokenizer.mask_id).nonzero().squeeze(-1)
                    blank_hypo = hypo[b, g][blank_ids]
                    blank_hypo = self.tokenizer.decode(blank_hypo.detach().cpu())
                    answers.append((blank_hypo, answer[b][g]))
                    matches.append(int(blank_hypo == answer[b][g]))

        return mean(matches), answers

    def get_reg_loss(self, h, c, group_mask):
        # use only rank loss to guarantee independence between samples
        # note that the length of group must be bigger than 1 for ranking loss
        rank_loss, rank_stats = self.get_rank_loss(h, c, group_mask)
        loss = rank_loss

        stats = {**rank_stats,
                 'rank_loss': rank_loss.item()}

        return loss, stats

    def run_generation(self, batch, features, features_merged, keywords, **kwargs):
        return self.run_train(batch, features, features_merged, keywords)


class FibNoGtSos(FibModel):
    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss, stats = self.get_reg_loss(o.detach(), feature, group_mask)
        return reg_loss, stats

    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only

from collections import OrderedDict

from torch import nn

from data.batcher import make_multichoice_batch

from .pretrain_aux import PretrainAuxGroupRoll


class McModel(PretrainAuxGroupRoll):
    transformer_name = 'bert_cased'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.multichoice_linear = nn.Linear(self.gpt_dim, 1)

    def make_batch(self, *args, **kwargs):
        return make_multichoice_batch(*args, **kwargs)

    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only

    def _forward(self, batch, **kwargs):
        B, G = batch.sentences.shape[:2]
        o, _, targets, reg_loss, stats, batch = super()._forward(batch, **kwargs)

        o = o.view(B, G, o.shape[1], -1)
        o = self.mean_pool_text(o)
        o = self.multichoice_linear(o).squeeze(-1)  # B G
        hypo = o.argmax(dim=-1)

        stats['accuracy'] = (hypo == targets).float().mean().item()
        stats['num_samples'] = batch.group_mask.sum().item()

        return hypo, o, targets.long(), reg_loss, stats, batch

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


class McNoGtSos(McModel):
    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss, stats = self.get_reg_loss(o.detach(), feature, group_mask)
        return reg_loss, stats

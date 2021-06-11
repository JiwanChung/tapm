from collections import OrderedDict

from .temporal_corr import TemporalCorrGlobal
from .pretrain_aux import PretrainAuxGroupRoll, ConcatAux


class ConcatNoGtSos(ConcatAux):
    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss, stats = self.get_reg_loss(o.detach(), feature, group_mask)
        return reg_loss, stats

    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only


class NoGtSos(TemporalCorrGlobal):
    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss, stats = self.get_reg_loss(o.detach(), feature, group_mask)
        return reg_loss, stats

    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only


class NoGtSosOrig(PretrainAuxGroupRoll):
    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss, stats = self.get_reg_loss(o.detach(), feature, group_mask)
        return reg_loss, stats

    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only


class NoGtGen(TemporalCorrGlobal):
    auxloss_generate = True


'''
class NoGtEos(TemporalCorrGlobal):
    def run_token(B, self, hypo, features, keywords):
        # hypo: (BG)L
        if self.training:
            if self.net.transformer.weight_freezed:
                hypo = torch.LongTensor(

        o, context = self.run_transformer(hypo, features, keywords)
        # (BG)LC
        logits = self.net.lm_head(o)

        return logits, o, context

    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only
'''

from collections import OrderedDict

from exp import ex
from .temporal_corr import TemporalCorrGlobal


class PreAll(TemporalCorrGlobal):
    def _fix_gpt(self, flag=True):
        print(f"gpt param not freezed: {flag}")


class PreVis(TemporalCorrGlobal):
    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss, stats = self.get_reg_loss(feature.detach(), feature, group_mask)
        return reg_loss, stats

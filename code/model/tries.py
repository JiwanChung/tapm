import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from exp import ex

from .no_gt import NoGtSos
from .temporal_corr import TemporalEncoderGlobal
from .encoders import AttEncoder, DeepEncoder, NoPositionEncoder


class AttCorrGlobal(NoGtSos):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[TemporalEncoderGlobal(dim, self.gpt_dim, Encoder=AttEncoder)]))


class CatFeatures(NoGtSos):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[NoPositionEncoder(dim, self.gpt_dim, Encoder=DeepEncoder)]))


class PlanModel(NoGtSos):
    def run_train(self, batch, features, features_merged, keywords, **kwargs):
        hypo = batch.sentences
        B, G = hypo.shape[:2]
        hypo_added, seed_len = self.add_plan_seeds(hypo)
        hypo_added = rearrange(hypo_added.contiguous(), 'b g l -> (b g) l')
        hypo = rearrange(hypo.contiguous(), 'b g l -> (b g) l')
        logits, o, context = self.run_token(B, hypo_added,
                                            features_merged, keywords,
                                            infer=False)
        plan_stats = {}
        plan_loss = None
        if self.training:
            plan_loss, plan_stats = self.get_plan_loss(o[:, :seed_len], hypo)
        logits = logits[:, seed_len:]
        o = o[:, seed_len:]
        if torch.isnan(logits).any():
            import ipdb; ipdb.set_trace()  # XXX DEBUG
        frame = batch.frame if hasattr(batch, 'frame') else None
        reg_loss, stats = self.process_reg_loss(o, context, features,
                                                batch.group_mask, G,
                                                frame)
        stats = {**plan_stats, **stats}
        if plan_loss is not None:
            reg_loss = reg_loss + plan_loss

        return None, logits, reg_loss, stats

    def add_plan_seeds(self, hypo):
        # BGL
        B, G = hypo.shape[:2]
        seeds = torch.full((B, G, G), self.tokenizer.context_sep_id).long().to(hypo.device)
        hypo = torch.cat((seeds, hypo), dim=-1)  # B G (G+L)
        return hypo, seeds.shape[-1]

    def get_plan_loss(self, plan, hypo):
        # (BG)GC, (BG)L
        G = plan.shape[1]
        h, group_mask = self.get_hypo_context(hypo)  # (BG)C, (BG)
        loss = F.mse_loss(plan, h.unsqueeze(-2).expand(-1, G, -1), reduction='none')
        # (BG)GC
        loss = loss.masked_select(group_mask.unsqueeze(-1).unsqueeze(-1)).mean()
        return loss, {'plan_loss': loss.item()}

    def get_hypo_context(self, hypo):
        pad_mask = hypo == self.tokenizer.pad_id  # BL
        pad_mask = pad_mask.unsqueeze(-1)  # BL1
        h = self.net.transformer.word_embedding(hypo)  # BLC
        h.masked_fill_(pad_mask, 0)
        pad_mask_sum = pad_mask.float().sum(dim=1)
        h = h.sum(dim=1) / (pad_mask_sum + self.eps)  # BC
        pad_mask_sum = (pad_mask_sum > 0)  # B
        h.masked_fill_(~pad_mask_sum, 0)

        return h.detach(), pad_mask_sum

    def get_hypo_init(self, B, G):
        s0 = torch.Tensor([*[self.tokenizer.context_sep_id] * G, self.tokenizer.cls_id])
        hypo = s0.long().unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # B1G

        return hypo

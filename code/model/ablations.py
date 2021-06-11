from torch import nn
import torch.nn.functional as F

from exp import ex
from .temporal_corr import TemporalCorrGlobal
from .no_gt import NoGtSos
# from .ss_loss import calc_l2_loss


class AblationJointSS(TemporalCorrGlobal):
    def forward(self, batch, **kwargs):
        return self._forward(batch, **kwargs)


class AblationNoSS(TemporalCorrGlobal):
    def forward(self, batch, **kwargs):
        hypo, logit, target, reg_loss, stats, batch = self._forward(batch, **kwargs)
        reg_loss = None
        return hypo, logit, target, reg_loss, stats, batch


class AblationSplitGen(TemporalCorrGlobal):
    def forward(self, batch, **kwargs):
        if self.training:
            self.fix_gpt(kwargs.get('epoch', 0))
        hypo, logit, target, reg_loss, stats, batch = self._forward(batch, **kwargs)
        reg_loss = None
        return hypo, logit, target, reg_loss, stats, batch


class AblationNoPred(TemporalCorrGlobal):
    def get_reg_loss(self, h, c, group_mask):
        rank_loss, rank_stats = self.get_rank_loss(h, c, group_mask)

        return rank_loss, rank_stats


class AblationNoMatch(TemporalCorrGlobal):
    def get_reg_loss(self, h, c, group_mask):
        roll_loss, roll_stats = self.get_roll_losses(h, c, group_mask)

        return roll_loss, roll_stats


class AblationS(NoGtSos):
    def mean_pool_text(self, o):
        # BGLC
        return o.mean(dim=2)  # use the [sos] token only


class AblationLossL2(NoGtSos):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

    def calc_l2_loss(self, x1, x2, group_mask=None, margin=None, pool='mean', skip_idx=0):
        loss = F.mse_loss(x1, x2, reduction=pool)
        acc = 0

        return loss, acc

    def run_rank_loss(self, x1, x2, group_mask, skip_idx=0):
        x1 = x1.view(-1, x1.shape[-1])
        x2 = x2.view(-1, x2.shape[-1])
        group_mask = group_mask.view(-1)

        loss1, acc1 = self.calc_l2_loss(x1, x2, group_mask,
                                        margin=self.margin, pool='mean',
                                        skip_idx=skip_idx)
        loss2, acc2 = self.calc_l2_loss(x2, x1, group_mask,
                                        margin=self.margin, pool='mean',
                                        skip_idx=-skip_idx)

        return loss1, acc1, loss2, acc2


class AblationLossCycle(NoGtSos):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        dim = self.gpt_dim

        self.cycle_linears = nn.ModuleDict({
            'vis_to_txt': nn.Linear(dim, dim),
            'txt_to_vis': nn.Linear(dim, dim),
        })

    def calc_cycle_loss(self, x1, x2, group_mask=None, pool='mean', skip_idx=0):
        l1 = F.mse_loss(self.cycle_linears['vis_to_txt'](x2), x1.detach(), reduction=pool)
        l2 = F.mse_loss(self.cycle_linears['txt_to_vis'](x1), x2.detach(), reduction=pool)

        return l1, l2

    def get_rank_loss(self, h, c, group_mask, skip_idx=0):
        x1 = F.normalize(h)
        x2 = F.normalize(c)

        l1, l2 = self.run_rank_loss(x1, x2, group_mask, skip_idx)
        loss = l1 + l2

        # stats = {'rank_accuracy': acc}
        stats = {'loss_ttov': l1.item(), 'loss_vtot': l2.item()}
        return loss, stats

    def run_rank_loss(self, x1, x2, group_mask, skip_idx=0):
        x1 = x1.view(-1, x1.shape[-1])
        x2 = x2.view(-1, x2.shape[-1])
        group_mask = group_mask.view(-1)

        l1, l2 = self.calc_cycle_loss(x1, x2, group_mask, pool='mean', skip_idx=skip_idx)

        return l1, l2

    def get_reg_loss(self, h, c, group_mask):
        loss, stats = self.get_rank_loss(h, c, group_mask)

        return loss, stats


class AblationIterSS(TemporalCorrGlobal):
    @classmethod
    def get_args(cls):
        return {
            **super().get_args(),
            'iter_ss': 1
        }

    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch,
                 iter_ss):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        self.iter_ss = iter_ss
        self.current_epoch = -1

    def fix_gpt(self, epoch):
        if epoch != self.current_epoch:
            if (epoch + 1) % self.iter_ss == 0:
                # revert ss
                if not self.net.transformer.weight_freezed:
                    self._fix_gpt()
                    self.net.transformer.weight_freezed = True
                else:
                    self._fix_gpt(False)
                    self.net.transformer.weight_freezed = False
                    self.reset_optimizer = True
            self.current_epoch = epoch

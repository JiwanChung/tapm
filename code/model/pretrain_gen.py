from .pretrain_aux import PretrainAuxContext


class PretrainGenContext(PretrainAuxContext):
    def forward(self, batch, **kwargs):
        if self.training:
            self.fix_gpt(kwargs.get('epoch', 0))
        hypo, logit, target, reg_loss, stats, batch = self._forward(batch, **kwargs)
        reg_loss = None
        return hypo, logit, target, reg_loss, stats, batch

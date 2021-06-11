from torch import nn

from exp import ex
from .encoders import DeepEncoder
from .transformer_dis_concat_reg import TransformerDisConcatReg


class ConcatPretrain(TransformerDisConcatReg):
    @classmethod
    def get_args(cls):
        return {
            **super().get_args(),
            'fix_gpt_epoch': 5
        }

    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before)

        self.fix_gpt_epoch = fix_gpt_epoch
        self.net.transformer.weight_freezed = True
        self._fix_gpt()

        for feature in self.feature_names:
            dim = self.feature_dims[feature]
            setattr(self, feature,
                    nn.Sequential(*[DeepEncoder(dim, self.gpt_dim)]))

    def fix_gpt(self, epoch):
        if epoch < self.fix_gpt_epoch:
            if not self.net.transformer.weight_freezed:
                self._fix_gpt()
                self.net.transformer.weight_freezed = True
        else:
            if self.net.transformer.weight_freezed:
                self._fix_gpt(False)
                self.net.transformer.weight_freezed = False
                self.reset_optimizer = True

    def _fix_gpt(self, flag=True):
        print(f"gpt param freezed: {flag}")
        for name, param in self.net.transformer.named_parameters():
            param.requires_grad_(not flag)

    def forward(self, batch, **kwargs):
        if self.training:
            self.fix_gpt(kwargs.get('epoch', 0))
        logit, targets, reg_loss, stats, batch = super()._forward(batch, **kwargs)
        if self.training:
            if self.net.transformer.weight_freezed:
                logit = None
            else:
                reg_loss = None
        return logit, targets, reg_loss, stats, batch

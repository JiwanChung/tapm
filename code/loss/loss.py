from exp import ex

from .base import Loss, SmoothLoss
from .sparse import SparseLoss


@ex.capture
def get_loss(padding_idx, label_smoothing, normalizer_sparsity, use_multichoice):
    if use_multichoice:
        return SmoothLoss(padding_idx=-1, eps=label_smoothing)
    elif normalizer_sparsity is None:
        return SmoothLoss(padding_idx=padding_idx, eps=label_smoothing)
    else:
        return SparseLoss(padding_idx=padding_idx, eps=label_smoothing,
                      sparsity=normalizer_sparsity)

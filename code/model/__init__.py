from .model import Model
from .mask_model import MaskModel
from .variational_masking import VariationalMasking


def get_model(args):
    return {
        'autoencoder': Model,
        'mask_model': MaskModel,
        'variational_masking': VariationalMasking
    }[args.model.lower()]

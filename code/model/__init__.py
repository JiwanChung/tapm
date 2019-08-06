from .model import Model
from .mask_model import MaskModel


def get_model(args):
    return {
        'autoencoder': Model,
        'mask_model': MaskModel
    }[args.model.lower()]

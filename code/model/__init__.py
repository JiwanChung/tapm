from .model import Model
from .mask_model import MaskModel
from .variational_masking import VariationalMasking
from .keyword_lm import LSTMKeywordLM


def get_model(args):
    return {
        'autoencoder': Model,
        'mask_model': MaskModel,
        'variational_masking': VariationalMasking,
        'lstm_keyword_lm': LSTMKeywordLM
    }[args.model.lower()].build(args)

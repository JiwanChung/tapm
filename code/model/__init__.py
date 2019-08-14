from .model import Model
from .mask_model import MaskModel
from .variational_masking import VariationalMasking
from .deterministic_masking import DeterministicMasking
from .lstm_keyword_lm import LSTMKeywordLM
from .insert_keyword_lm import InsertKeywordLM


def get_model(args):
    model = {
        'autoencoder': Model,
        'mask_model': MaskModel,
        'variational_masking': VariationalMasking,
        'deterministic_masking': DeterministicMasking,
        'lstm_keyword_lm': LSTMKeywordLM,
        'insert_keyword_lm': InsertKeywordLM,
    }[args.model.lower()]
    if hasattr(model, 'get_args'):
        args = model.get_args(args)
    return model.build(args)

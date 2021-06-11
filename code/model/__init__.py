import os
import inspect
from pathlib import Path

from torch import nn

from inflection import underscore


model_dict = {}


def add_models():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != "__init__":
            __import__(f"{parent}.{name}")
            module = eval(name)
            for member in dir(module):
                # Add to dict all nn.Module classes
                member = getattr(module, member)
                if hasattr(member, '__mro__') and \
                        nn.Module in inspect.getmro(member):
                    model_dict[underscore(str(member.__name__))] = member


def get_model_options():
    if not model_dict:
        add_models()
    return list(model_dict.keys())


def get_model_class(model):
    if not model_dict:
        add_models()
    model = model_dict[model]
    if hasattr(model, 'get_args'):
        args = model.get_args()
    return args, model


def get_model(model, data, transformer_name=None):
    args, model = get_model_class(model)
    model, tokenizer = model.build(data, transformer_name)
    return model, tokenizer

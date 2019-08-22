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


def get_model(args, data):
    model = model_dict[args.model]
    if hasattr(model, 'get_args'):
        args = model.get_args(args)
    return model.build(args, data)


add_models()

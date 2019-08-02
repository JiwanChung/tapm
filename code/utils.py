from collections import defaultdict
from datetime import datetime

import six

import torch
from torch.nn.utils import clip_grad_norm_


def get_dirname_from_args(args):
    dirname = ''
    for key in sorted(args.log_keys):
        dirname += '_'
        dirname += key
        dirname += '_'
        dirname += str(args[key])

    return dirname[1:]


def get_now():
    now = datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S')


def add_attr(dt, name, val):
    for key, value in dt.items():
        setattr(value, name, val)
        dt[key] = value

    return dt


def transpose_dict(dt):
    d = defaultdict(dict)
    for key1, inner in dt.items():
        for key2, value in inner.items():
            d[key2][key1] = value
    return d


def peek_dict(dt):
    return next(iter(dt.items()))


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def check_equal(li):
    if isinstance(li[0], list):
        li = [list(i) for i in zip(*li)]
        return min([int(len(set(l)) <= 1) for l in li]) > 0
    else:
        return len(set(li)) <= 1


def clip_grad(model, max_norm=1):
    if max_norm is not None:
        for p in model.parameters():
            clip_grad_norm_(p, max_norm)


def wait_for_key(key="y"):
    text = ""
    while (text != key):
        text = six.moves.input(f"Press {key} to quit")
        if text == key:
            print("terminating process")
        else:
            print(f"key {key} unrecognizable")

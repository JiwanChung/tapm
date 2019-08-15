from collections import defaultdict
from datetime import datetime

import six

import torch
from torch.nn.utils import clip_grad_norm_


def add_keyword_paths(data_path, keyword_dir):
    keyword_dir = resolve_keyword_dir(keyword_dir, list(data_path.values())[0])
    for key, path in data_path.items():
        data_path[key] = get_keyword_path(data_path, key, keyword_dir)
    return data_path


def resolve_keyword_dir(keyword_dir, path):
    candidates = list(path.parent.glob(f'{keyword_dir}*'))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        x = candidates[0]
        print(f"choosing keyword path {x}")
        return x
    else:
        assert len(candidates) > 0, \
            f"no candidate for keyword dir: {keyword_dir}"


def get_keyword_path(data_path, key, dir_name=None, args=None):
    if dir_name is None:
        dir_name = f'keywords_{get_dirname_from_args(args)}'
    path = data_path[key].parent / dir_name
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"{data_path[key].name}.json"
    return path


def jsonl_to_json(x):
    keys = x[0].keys()
    res = {}
    for key in keys:
        res[key] = [i[key] for i in x]
    return res


def cut_sample(data, n=100):
    if isinstance(data, list):
        return data[:n]
    elif isinstance(data, dict):
        return {k: v for i, (k, v) in enumerate(data.items())
                if i < n}
    else:
        assert False, f'cutting not implemented for type {data.type}'


def get_dirname_from_args(args):
    dirname = ''
    for key in sorted(args.log_keys):
        dirname += '_'
        dirname += key
        dirname += '_'
        val = args[key]
        if isinstance(val, float):
            val = '{:.2f}'.format(val)
        else:
            val = str(val)
        dirname += val

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

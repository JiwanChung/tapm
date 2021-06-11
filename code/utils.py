from contextlib import contextmanager
import sys, os
import random
from collections import defaultdict
from datetime import datetime
from copy import copy as pycopy

import six

# import stanfordnlp
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_


def remove_nonascii(text):
    # return ''.join([i if ord(i) < 128 else ' ' for i in text])
    return text.encode('ascii', 'ignore').decode('ascii')


def flatten_dict(x):
    if isinstance(x, dict):
        li = [flatten_dict(v) for v in x.values()]
        return flatten_list(li)
    elif isinstance(x, list):
        return x
    else:
        return [x]


def flatten_list(x):
    return [it for subli in x for it in subli]


def recurse(shape, *args, vid='0', func=None):
    if len(shape) > 0:
        for i in range(shape[0]):
            if not isinstance(vid, str):
                if len(vid) > i:
                    vid_new = vid[i]
                    recurse(shape[1:], *list([v[i] if v is not None else v for v in args]),
                        vid=vid_new, func=func)
            else:
                vid_new = f'{vid}_{i}'
                recurse(shape[1:], *list([v[i] if v is not None else v for v in args]),
                        vid=vid_new, func=func)
    else:
        func(*args, vid=vid)


def jsonl_to_json(x):
    def get_key(t):
        if isinstance(t, dict):
            return t.keys()
        else:
            return get_key(t[0])
    keys = get_key(x)
    def merge_key(t, key):
        if isinstance(t[0], dict):
            return [i[key] for i in t if key in i]
        else:
            return [[k for k in merge_key(i, key)] for i in t]
    res = {}
    for key in keys:
        res[key] = merge_key(x, key)
    return res


def mean(x):
    x = list(x)
    x = [i for i in x if i is not None]
    if len(x) == 0:
        return None
    return sum(x) / len(x)


def cut_sample(data, n=800):
    if isinstance(data, list):
        return data[:n]
    elif isinstance(data, dict):
        return {k: v for i, (k, v) in enumerate(data.items())
                if i < n}
    else:
        assert False, f'cutting not implemented for type {data.type}'


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


def remove_duplicate(li, key=lambda x: x):
    keys = set([key(i) for i in li])
    keys = {k: False for k in keys}
    res = []
    for i in li:
        i_key = key(i)
        if not keys[i_key]:
            keys[i_key] = True
            res.append(i)
    return res


def remove_sep(string, sep='[SEP]'):
    return string[:string.find(sep)].strip()


def copy(x):
    if isinstance(x, list):
        res = []
        for i in range(len(x)):
            res.append(copy(x[i]))
    elif isinstance(x, dict):
        res = {}
        for k in x.keys():
            res[k] = copy(x[k])
    else:
        res = x
    return res


def recursive_insert_dict(dt, key_list, val):
    key = key_list[0]
    if len(key_list) == 1:
        dt[key] = val
        return dt
    if key not in dt:
        dt[key] = {}
    udt_dt = recursive_insert_dict(dt[key], key_list[1:], val)
    dt[key] = udt_dt
    return dt


def concatenate_group(text, groups):
    res = {}
    for group_key, vids in groups.items():
        t = [text[vid] for vid in vids]
        hypo, tgt = zip(*t)
        hypo = ' '.join(list(hypo))
        tgt = ' '.join(list(tgt))
        res[group_key] = (hypo, tgt)
    return res


def break_tie(li, longest=False):
    if longest:
        res = ''
        for l in li:
            if len(l) >= len(res):
                res = l
    else:
        len_res = 9999
        for l in li:
            if len(l) < len_res:
                res = l
                len_res = len(res)
    return res


def refactor_text(text, albums=None, images=None,
                  return_list=False, longest=False):
    if albums is None:
        return text
    else:
        reverse_albums = {vi: k for k, v in albums.items() for vi in v}
        text = [{'album_id': reverse_albums[k],
                 'photo_sequence': images[k],
                 'story_text_normalized': v.lower()} for k, v in text.items()]
        text = [(f"{v['album_id']}_{v['photo_sequence']}", v) for v in text]   # remove duplicate
        keys = [i[0] for i in text]
        keys = list(set(keys))
        res = []
        for key in keys:
            t = [v for k, v in text if k == key]
            if not return_list:
                t = break_tie(t, longest=longest)
            else:
                t = {**t[0],
                     'story_text_normalized': [i['story_text_normalized'] for i in t]}
            res.append(t)
        return {
            'team_name': "temp_team_name",
            "evaluation_info": {
                "additional_description": "none"
            },
            "output_stories": res}


def merge_vist_output(hypo, tgt, album=False):
    hypo = hypo['output_stories']
    tgt = tgt['output_stories']

    if album:
        res = defaultdict(list)
        for v in hypo:
            res[v['album_id']].append(v['story_text_normalized'])
        hypo = {}
        # tie breaking
        for k, v in res.items():
            hypo[k] = break_tie(v)
    else:
        hypo = {f"{v['album_id']}_{v['photo_sequence']}": v['story_text_normalized'] for v in hypo}
    res = defaultdict(list)
    for v in tgt:
        if album:
            res[v['album_id']].append(v['story_text_normalized'])
        else:
            res[f"{v['album_id']}_{v['photo_sequence']}"].append(v['story_text_normalized'])
    tgt = {k: flatten(v) for k, v in res.items()}
    '''
    if normalize:
        normalizer = VistTokenizer()
        hypo = {k: normalizer(v) for k, v in hypo.items()}
        tgt = {k: [normalizer(i) for i in v] for k, v in tgt.items()}
    '''

    return {k: (hypo[k], tgt[k]) for k in tgt.keys()}


'''
def normalize_text(t):
    t = t.replace('.', ' .')
    t = t.replace(',', ' ,')
    t = t.replace('?', ' ?')
    t = t.replace('!', ' !')
    t = t.replace("'s", " 's")
    t = t.replace("n't", " n't")

    return t
'''


'''
class VistTokenizer:
    def __init__(self):
        self.nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')

    def _tokenize(self, x):
        doc = self.nlp(x)
        res = []
        for sent in doc.sentences:
            res += [token.words[0].text for token in sent.tokens]
        res = ' '.join(res)
        return res

    def tokenize(self, x):
        with suppress_stdout():
            return self._tokenize(x)

    def __call__(self, x):
        return self.tokenize(x)
'''


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flatten(li):
    return [item for sublist in li for item in sublist]


@contextmanager
def suppress_stdout(do=True):
    if do:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

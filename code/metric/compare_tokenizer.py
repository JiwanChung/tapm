from itertools import chain
from pathlib import Path
import json


def compare(tokenizer):
    path = Path('~/projects/lsmdc/data/VIST/sis/train.story-in-sequence.json').expanduser()
    with open(path, 'r') as f:
        data = json.load(f)['annotations']
    texts = [v[0]['text'] for v in data]
    orig_texts = [v[0]['original_text'] for v in data]

    texts = [v.split() for v in texts]
    orig_texts = [tokenizer(v.lower()) for v in orig_texts]

    diff = [(i, v) for i, v in enumerate(zip(texts, orig_texts)) if len(v[0]) != len(v[1])]

    print(len(diff))

    return diff


def flatten_list(li):
    return list(chain(*li))


def cut_apst(w):
    w = w.split("'")
    w = [f"'{v}" for v in w]
    w[0] = w[0][1:]

    return w


def cut_dot(w):
    if w.endswith('.') and w != 'w' and not w.endswith('..'):
        return [w[:-1], w[-1]]
    else:
        return [w]


def cut_dash(w):
    if '--' in w and w != '--':
        w = w.split("--")
        w = flatten_list([['--', v] for v in w])
        w = w[1:]
        return w
    else:
        return [w]


def tok_test(s):
    s = s.split()
    s = flatten_list([[w[:-1], w[-1]] if w.endswith('!') else [w] for w in s])
    s = flatten_list([[w[:-1], w[-1]] if w.endswith('?') else [w] for w in s])
    s = flatten_list([[w[:-1], w[-1]] if w.endswith(',') and w != ',' else [w] for w in s])
    s = flatten_list([cut_dot(w) for w in s])
    s = flatten_list([cut_apst(w) if "'" in w and w != "'" else [w] for w in s])
    s = flatten_list([cut_dash(w) for w in s])

    s = [v for v in s if v != '']

    s = ' '.join(s)
    '''
    if s[-1] == '.':
        s = f"{s[:-1]} ."
        '''
    s = s.replace('!!!', '! ! !')
    s = s.replace('!!', '! !')
    s = s.replace('!?', '! ?')
    s = s.replace('?!', '? !')
    s = s.split(' ')

    return s

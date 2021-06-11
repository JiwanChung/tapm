import json
from pathlib import Path
from collections import defaultdict
import argparse

from metric_test import Metric
from normalize import normalize


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flatten(li):
    return [item for sublist in li for item in sublist]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--name', default='test_results.json', type=str)
    parser.add_argument('-v', '--vist', type=bool, default=False)
    parser.add_argument('-a', '--album', action='store_false')
    parser.add_argument('-l', '--longest', action='store_true')
    parser.add_argument('-c', '--cut', action='store_true')
    parser.add_argument('-s', '--eval-set', action='store_true')
    parser.add_argument('-dot', '--dot', action='store_true')  # shift last dot
    args = parser.parse_args()
    return args


def shift_dot(hypo):
    h = hypo['output_stories']
    h = [{**v, 'story_text_normalized': _shift_dot(v['story_text_normalized'])}
         for v in h]

    return {'output_stories': h}


def _shift_dot(text):
    if (not text.endswith(' .')) and text.endswith('.'):
        text = text[:-1] + ' .'
    return text


def break_tie(li, longest=True):
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


def load_file(name):
    path = Path('~/projects/lsmdc/data/VIST/sis/samples/full').expanduser() / name
    with open(path, 'r') as f:
        hypo = json.load(f)

    return hypo


def calc(hypo, album=True, longest=False):
    hypo = hypo['output_stories']
    if album:
        res = defaultdict(list)
        for v in hypo:
            res[v['album_id']].append(v['story_text_normalized'])
        hypo = {}
        # tie breaking
        for k, v in res.items():
            hypo[k] = break_tie(v, longest)
    else:
        hypo = {f"{v['album_id']}_{v['photo_sequence']}": v['story_text_normalized'] for v in hypo}
    tgt_name = 'albums' if album else 'images'
    path = Path('../../').resolve() / 'data' / 'VIST' / 'sis' / f'post_test.story-in-sequence_{tgt_name}.json'
    with open(path, 'r') as f:
        tgt = json.load(f)
    m = Metric(['cider', 'meteor', 'bleu', 'rouge'], False, use_vist=True, no_tokenizer=False)
    print(m({k: (hypo[k], tgt[k]) for k in hypo.keys()}))


def main():
    args = parse_args()
    print(args)
    if args.vist:
        run_vist(args)
    elif len(args.name.split('/')) > 1:
        run_ckpt(args)
    else:
        run_lsmdc(args)


def run_vist(args):
    if args.cut:
        hypo = normalize(args.name, args.longest)
    elif args.dot:
        hypo = load_file(args.name)
        hypo = shift_dot(hypo)
    else:
        hypo = load_file(args.name)

    calc(hypo, args.album, args.longest)


def run_lsmdc(args, hypo=None):
    root = Path('../../').resolve() / 'data' / 'LSMDC' / 'task1'
    tgt_path = root / 'LSMDC16_annos_test_someone.csv'
    tgt = {}
    keys = []
    with open(tgt_path, 'r') as f:
        for line in f:
            line = line.split('\t')
            tgt[line[0].strip()] = line[-1].strip()
            keys.append(line[0].strip())
    set_keys = build_set(keys)
    if hypo is None:
        hypo_path = root / 'samples' / 'full' / args.name
        with open(hypo_path, 'r') as f:
            hypo = json.load(f)

    if args.eval_set:
        hypo_set = {str(i): ' '.join(hypo[key] for key in keys) for i, keys in set_keys.items()}
        tgt_set = {str(i): ' '.join(tgt[key] for key in keys) for i, keys in set_keys.items()}
        hypo, tgt = hypo_set, tgt_set
    m = Metric(['cider', 'meteor', 'bleu', 'rouge'], False, use_vist=False, no_tokenizer=False)
    print(m({k: (hypo[k], tgt[k]) for k in hypo.keys()}))


def build_set(keys):
    res = defaultdict(list)
    for key in keys:
        name = key[:key.rfind('_')]
        res[name].append(key)
    res = {k: [(f"{k}_{i}", v2) for i, v2 in enumerate(list(chunks(v, 5)))] for k, v in res.items()}
    return dict(flatten(res.values()))


def run_ckpt(args):
    ckpt = Path('../../').resolve() / 'data' / 'ckpt'
    dir_name = args.name.split('/')
    dir_name, file_name = dir_name
    dir_path = list(ckpt.glob(dir_name))
    assert len(dir_path) > 0, f"nonexisting dir name {dir_name}"
    dir_path = dir_path[0]
    file_path = list(dir_path.glob(file_name))
    assert len(file_path) > 0, f"nonexisting file name {file_name}"
    file_path = file_path[0]
    print(f"Loading from {file_path}")
    with open(file_path, 'r') as f:
        hypo = json.load(f)
    hypo = dict({k: v[0] for k, v in hypo.items()})
    run_lsmdc(args, hypo)


if __name__ == '__main__':
    main()

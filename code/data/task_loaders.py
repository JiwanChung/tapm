import re
import copy
import json
import sys
from collections import defaultdict, OrderedDict
from itertools import chain

import numpy as np

from exp import ex
from utils import cut_sample, jsonl_to_json, peek_dict
from .vist_loaders import load_sis
from .actynetcap_loaders import load_actynetcap
from .feature_loaders import load_features


@ex.capture
def load_tasks(path, task_name):
    # */^[_]+_*/* -> $1
    path_task_name = path.parts[-2].split('_')[0]
    if task_name is None or 'task' not in path_task_name:
        task_name = path_task_name
    if 'fib' in path.stem:
        task_name = 'fib'
    elif '_MC_' in path.stem:
        task_name = 'multichoice'
    elif 'ActivityNet' in str(path):
        task_name = 'actynet'
    f = getattr(sys.modules[__name__], f"load_{task_name}")
    data, global_data = f(path)
    return data, global_data, task_name, path


@ex.capture
def load_task1(path, feature_names, keyword_name):
    if feature_names is not None and \
            len(feature_names) > 0:
        if keyword_name is not None:
            return load_task1_group_with_features_and_keyword(path)
        else:
            return load_task1_group_with_features(path)
    elif keyword_name is not None:
        return load_keywords(path)
    else:
        return load_task1_text_only(path)


def load_fib(path):
    return load_fib_group_with_features(path)


def load_multichoice(path):
    return load_multichoice_group_with_features(path)


def load_actynet(path):
    return load_actynet_group_with_features(path)


def load_task2(path):
    return load_task2_text(path)


def load_task2_text(path):
    blank = load_lsmdc_text(path.parent.parent / f'task2' / f'{path.stem}_blank{path.suffix}')
    all_feature = load_features(path, blank.keys())
    if len(list(all_feature.keys())) > 0:
        blank = OrderedDict([(k, {**v, **all_feature[k]}) for k, v in blank.items()])
    ids_path = path.parent / f'{path.stem}_onlyIDs{path.suffix}'
    if ids_path.is_file():
        ids = load_lsmdc_text(ids_path)
        data = OrderedDict([(k, (blank[k], ids[k]['target'])) for k in blank.keys()])
        def check_if_word(v):
            return '[...]' in v[0]['target'] and len(v[1].split(',')) > 0
        # data = OrderedDict([(k, v) for k, v in data.items() if check_if_word(v)])
    else:
        print(f"no gt IDs for file {path}")
        data = OrderedDict([(k, (blank[k], None)) for k in blank.keys()])

    group_keys = make_groups(data.keys())
    data = {k: [data[i] for i in v] for k, v in group_keys.items()}
    # skip no word sample

    def update_words(x):
        ex, words = zip(*x)
        res = {}
        res['target'] = [e['target'] for e in ex]
        res['blank_num'] = np.array([len(list(re.finditer(r'\[\.\.\.\]', i))) for i in res['target']])
        res['target'] = '\n'.join(res['target'])
        res['input'] = copy.deepcopy(res['target'])

        words = list(words)
        if len(words) > 0 and words[0] is not None:
            words = [word.split(',') for word in words]
            words = list(chain(*words))
            words = [w for w in words if w != '_']
            # make word map
            # first come first served
            word_map = []
            for word in words:
                if word not in word_map:
                    word_map.append(word)
            word_map = {v: f'[PERSON{i}]' for i, v in enumerate(word_map)}

            counter = 0
            for word in words:
                span = list(re.finditer(r'\[\.\.\.\]', res['target']))
                if len(span) == 0:
                    break
                span = span[0].span()
                res['target'] = res['target'][:span[0]] + word_map[word] + res['target'][span[1]:]
                counter += 1
            assert counter >= 0, print(f"no span for {res['target']} \n {words}")
        ex = jsonl_to_json(ex)
        ex.update(res)

        return ex

    data = {k: update_words(v) for k, v in data.items()}

    return data, {}


def load_task1_group_with_features(path):
    data, _ = load_task1_with_features(path)
    group_keys = make_groups(data.keys())
    data = {k: [data[i] for i in v] for k, v in group_keys.items()}

    return data, {}


def load_fib_group_with_features(path):
    data, _ = load_fib_with_features(path)
    group_keys = make_groups(data.keys())
    data = {k: [data[i] for i in v] for k, v in group_keys.items()}

    return data, {}


def load_multichoice_group_with_features(path):
    data, group_keys = load_multichoice_with_features(path)
    data = {k: [data[i] for i in v] for k, v in group_keys.items()}

    return data, {}


def load_actynet_group_with_features(path):
    data, _ = load_actynet_with_features(path)
    group_keys = make_groups(data.keys())
    data = {k: [data[i] for i in v] for k, v in group_keys.items()}

    return data, {}


def load_task1_group_with_features_and_keyword(path):
    data, _ = load_task1_group_with_features(path)
    keywords = load_keyword_only(path, None)
    total = load_keyword_only(path, name='keywords_gpt_total.json')

    return data, {'keywords': keywords, 'word_counter': total}


def make_groups(keys, chunk_size=5):
    movie_keys = defaultdict(list)
    for key in keys:
        name = key[:key.rfind('_')]
        movie_keys[name].append(key)
    movie_keys = {k: [v[i:i + chunk_size] for i in range(0, len(v), chunk_size)] \
                  for k, v in movie_keys.items()}
    res = {}
    for k, v in movie_keys.items():
        for i, chunk in enumerate(v):
            res[chunk[0]] = chunk
    return res


def load_task1_with_features(path):
    data = load_lsmdc_text(path)
    #data = dict(list(data.items())[:24])
    all_feature = load_features(path, data.keys())

    return {k: {**v, **all_feature[k]} for k, v in data.items()}, {}


def load_fib_with_features(path):
    data = load_fib_text(path)
    all_feature = load_features(path, [datum['vid'] for datum in data.values()])

    return {k: {**v, **all_feature[v['vid']]} for k, v in data.items()}, {}


def load_multichoice_with_features(path):
    data, group_keys = load_multichoice_text(path)
    all_feature = load_features(path, [datum['vid'] for datum in data.values()])

    return {k: {**v, **all_feature[v['vid']]} for k, v in data.items()}, group_keys


def load_actynet_with_features(path):
    data = load_actynet_text(path)
    all_feature = load_features(path, data.keys())

    return {k: {**v, **all_feature[k]} for k, v in data.items()}, {}


@ex.capture
def load_lsmdc_text(path, sample):
    '''
    LSMDC task csv files lack header,
    and have their delimiter='\t', lineterminator='\n'
    '''
    data = []
    keys = ['vid', 'vid_start', 'vid_end',
            'cap_start', 'cap_end', 'caption']
    if not path.is_file():
        path = path.parent / f"{path.stem}_someone{path.suffix}"

    with open(path, 'r') as f:
        for row in f:
            row = row.split('\t')
            if len(keys) > len(row):
                keys = keys[:len(row)]
            row = {k: v for k, v in zip(keys, row)}
            if 'caption' in row:
                row['target'] = row['caption'].strip()
            data.append(row)
    data = OrderedDict([(i['vid'], i) for i in data])
    if sample:
        data = cut_sample(data)
    return data


@ex.capture
def load_fib_text(path, sample):
    '''
    LSMDC task csv files lack header,
    and have their delimiter='\t', lineterminator='\n'
    '''
    data = []
    keys = ['vid', 'full_caption', 'masked_caption',
            'answer', 'vid_key', 'key']
    if not path.is_file():
        path = path.parent / f"{path.stem}_{path.suffix}"

    def match_length(src, tgt):
        src = src.split()
        tgt = tgt.split()
        length = min(len(src), len(tgt))
        if src[-1] != tgt[-1] and '_____' not in src[-1]:
            length -= 1
        src = ' '.join(src[:length])
        tgt = ' '.join(tgt[:length])
        return src, tgt

    def get_blank(src, tgt):
        src = src.split()
        tgt = tgt.split()
        index = [i for i, v in enumerate(src) if '_____' in v]
        if len(index) > 0:
            index = index[0]
        else:
            import ipdb; ipdb.set_trace()
        answer = tgt[index]
        src[index] = '[MASK]'
        src = ' '.join(src)
        tgt = ' '.join(tgt)
        return src, tgt, answer

    if_header = True
    with open(path, 'r') as f:
        for row in f:
            if if_header:
                if_header = False
                continue
            row = row.split('\t')
            if len(keys) > len(row):
                keys = keys[:len(row)]
            row = {k: v for k, v in zip(keys, row)}
            tgt = row['full_caption'].strip()
            src = row['masked_caption'].strip()
            src, tgt = match_length(src, tgt)
            src, tgt, answer = get_blank(src, tgt)
            row['source'] = src
            row['target'] = tgt
            row['answer'] = answer

            row['key'] = f"FIB_{row['key'].strip()}"
            data.append(row)
    data = OrderedDict([(i['key'], i) for i in data])
    if sample:
        data = cut_sample(data)
    return data


@ex.capture
def load_multichoice_text(path, sample, sample_length=100):
    data = []
    keys = ['vid', 'a1', 'a2', 'a3', 'a4', 'a5',
            'answer', 'vid_key', 'key']
    if not path.is_file():
        path = path.parent / f"{path.stem}_{path.suffix}"

    if_header = True
    group_keys = defaultdict(list)
    with open(path, 'r') as f:
        for row in f:
            if if_header:
                if_header = False
                continue
            row = row.strip().split('\t')
            if len(keys) > len(row):
                keys = keys[:len(row)]
            row = {k: v for k, v in zip(keys, row)}
            for i, a in enumerate(['a1', 'a2', 'a3', 'a4', 'a5']):
                key = f"{row['key']}_{i}"
                row_answer = {'vid': row['vid'], 'target': row[a],
                              'answer': row['answer'],
                              'group_key': row['key'],
                              'key': key}
                group_keys[row['key']].append(key)
                data.append(row_answer)

            if sample:
                if len(data) > sample_length:
                    break
    data = OrderedDict([(i['key'], i) for i in data])
    group_keys = dict(group_keys)
    return data, group_keys


@ex.capture
def load_actynet_text(path, sample):
    '''
    ActivityNet pretraining data loader
    '''
    data = []

    if not path.is_file():
        path = path.parent / f"{path.stem}_{path.suffix}"

    if_header = True
    with open(path, 'r') as f:
        rows = json.load(f)
        for line in rows:
            vid = line[0].split('/')[-1]
            num = "{:03d}".format(line[1])
            key = "{}_{}".format(vid, num)
            row = {'key': key, 'vid': vid, 'target': '[DUMMY]'}
            data.append(row)
    data = OrderedDict([(i['key'], i) for i in data])
    if sample:
        data = cut_sample(data)
    return data


def load_task1_text_only(path):
    data = load_lsmdc_text(path)
    k, v = peek_dict(data)
    if 'target' in v:
        data = {k: {'target': v['target']} for k, v in data.items()}
    return data, {}


def load_keywords(path):
    data, _ = load_task1_text_only(path)
    keywords = load_keyword_only(path, None)

    return data, {'keywords': keywords}


@ex.capture
def load_keyword_only(path, name, keyword_name):
    if name is None:
        name = keyword_name
    paths = path.parent.glob(f"keywords/{name}*")
    paths = list(sorted(list(paths)))
    assert len(paths) > 0, f"no keyword candidate for {name}"
    path = paths[0]
    print(f"loading keyword from {path}")
    assert path.is_file(), f"keyword {path} is not a file"
    with open(path, 'r') as f:
        keywords = json.load(f)  # (k, count)
    return keywords


def load_actynet_cap(path):
    with open(path, 'r') as f:
        x = json.load(f)
    data = {}
    for k, v in x.items():
        for i, sent in enumerate(v['sentences']):
            data[f"{k}/{i}"] = {'target': sent}
    return data, {}

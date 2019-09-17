import re
import copy
import json
import sys
from collections import defaultdict
from itertools import chain

from tqdm import tqdm
import numpy as np

from utils import cut_sample, jsonl_to_json


def load_tasks(args, path):
    # */^[_]+_*/* -> $1
    name = args.get('task_name', path.parts[-2].split('_')[0])
    f = getattr(sys.modules[__name__], f"load_{name}")
    return f(args, path)


def load_task1(args, path):
    if args.feature_names is not None and \
            len(args.feature_names) > 0:
        if args.keyword_name is not None:
            return load_task1_group_with_features_and_keyword(args, path)
        else:
            return load_task1_group_with_features(args, path)
    elif args.keyword_name is not None:
        return load_keywords(args, path)
    else:
        return load_task1_text_only(args, path)


def load_task2(args, path):
    return load_task2_text(args, path)


def load_task2_text(args, path):
    blank = load_lsmdc_text(args, path.parent / f'{path.stem}_blank{path.suffix}')
    all_feature = load_features(args, path, blank.keys())
    blank = {k: {**v, **all_feature[k]} for k, v in blank.items()}
    ids_path = path.parent / f'{path.stem}_onlyIDs{path.suffix}'
    if ids_path.is_file():
        ids = load_lsmdc_text(args, ids_path)
        data = {k: (blank[k], ids[k]['target']) for k in blank.keys()}
        def check_if_word(v):
            return '[...]' in v[0]['target'] and len(v[1].split(',')) > 0
        data = {k: v for k, v in data.items() if check_if_word(v)}
    else:
        data = {k: (blank[k], None) for k in blank.keys()}

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
            assert counter > 0, print(f"no span for {res['target']} \n {words}")
        ex = jsonl_to_json(ex)
        ex.update(res)

        return ex

    data = {k: update_words(v) for k, v in data.items()}

    return data, {}


def load_task1_group_with_features(args, path):
    data, _ = load_task1_with_features(args, path)
    group_keys = make_groups(data.keys())
    data = {k: [data[i] for i in v] for k, v in group_keys.items()}

    return data, {}


def load_task1_group_with_features_and_keyword(args, path):
    data, _ = load_task1_group_with_features(args, path)
    keywords = load_keyword_only(args, path)
    total = load_keyword_only(args, path, name='keywords_gpt_total.json')

    return data, {'keywords': keywords, 'word_counter': total}


def make_groups(keys, chunk_size=5):
    movie_keys = defaultdict(list)
    for key in keys:
        name = key[:key.rfind('_')]
        movie_keys[name].append(key)
    movie_keys = {k: sorted(v) for k, v in movie_keys.items()}
    movie_keys = {k: [v[i:i + chunk_size] for i in range(0, len(v), chunk_size)] \
                  for k, v in movie_keys.items()}
    res = {}
    for k, v in movie_keys.items():
        for i, chunk in enumerate(v):
            res[f"{k}_{i}"] = chunk
    return res


def load_task1_with_features(args, path):
    data = load_lsmdc_text(args, path)
    all_feature = load_features(args, path, data.keys())

    return {k: {**v, **all_feature[k]} for k, v in data.items()}, {}


def load_features(args, path, data_keys):
    path = path.parent.parent.glob(f"features/*")
    path = [p for p in list(path) if p.name in args.feature_names or \
            (p.name in args.feature_name_map and args.feature_name_map[p.name] in args.feature_names)]
    features = {}
    feature_names = copy.deepcopy(args.feature_names)
    reverse_map = {v: k for k, v in args.feature_name_map.items()}
    feature_names = [reverse_map[x] if x in reverse_map else x for x in feature_names]
    for p in path:
        if p.stem in feature_names and p.stem != 'rcnn':
            features[p.name] = load_feature(p, data_keys)
        elif p.stem in feature_names and p.stem == 'rcnn':
            features[p.name] = load_feature_box(p, data_keys)
    all_feature = {}
    for k in features[list(features.keys())[0]].keys():
        all_feature[k] = {p: v[k] for p, v in features.items()}
    return all_feature


def load_feature(path, keys):
    path = path.glob('*')
    path = {p.name: {p2.stem: p2 for p2 in p.glob('*')} for p in path}
    res = {}
    for k in tqdm(keys):
        f1_name = '_'.join(k.split('_')[:-1])
        f2_name = k
        p = path[f1_name][f2_name]
        res[k] = mean_pool_segments(np.load(p))
    return res


def load_feature_box(path, keys):
    path = path.glob('*')
    path = {p.name: {p2.stem: p2 for p2 in p.glob('*')} for p in path}
    res = {}
    for k in tqdm(keys):
        f1_name = '_'.join(k.split('_')[:-1])
        f2_name = k
        p = path[f1_name][f2_name]
        feat = np.load(p).tolist()
        feat = list(map(box_preprocess, feat))
        res[k] = mean_pool_segments(np.asarray(feat, dtype=np.float32))
    return res


def box_preprocess(feat):
    # T=1, Object(20), Info(6 - prob, idx, bbox)
    ret = [0 for i in range(1600)]
    for i in range(len(feat)):
        if ret[int(feat[i][1])-1] < feat[i][0]:
            ret[int(feat[i][1])-1] = feat[i][0]
    return ret
    #return np.asarray(ret, dtype=np.float32)


def load_lsmdc_text(args, path):
    '''
    LSMDC task csv files lack header,
    and have their delimiter='\t', lineterminator='\n'
    '''
    data = []
    keys = ['vid', 'vid_start', 'vid_end',
            'cap_start', 'cap_end', 'caption']
    with open(path, 'r') as f:
        for row in f:
            row = row.split('\t')
            row = {k: v for k, v in zip(keys, row)}
            row['target'] = row['caption'].strip()
            data.append(row)
    data = {i['vid']: i for i in data}
    if args.sample:
        data = cut_sample(data)
    return data


def load_task1_text_only(args, path):
    data = load_lsmdc_text(args, path)
    data = {k: {'target': v['target']} for k, v in data.items()}
    return data, {}


def load_keywords(args, path):
    data, _ = load_task1_text_only(args, path)
    keywords = load_keyword_only(args, path)

    return data, {'keywords': keywords}


def load_keyword_only(args, path, name=None):
    if name is None:
        name = args.keyword_name
    paths = path.parent.glob(f"keywords/{name}*")
    paths = list(sorted(list(paths)))
    assert len(paths) > 0, f"no keyword candidate for {name}"
    path = paths[0]
    print(f"loading keyword from {path}")
    assert path.is_file(), f"keyword {path} is not a file"
    with open(path, 'r') as f:
        keywords = json.load(f)  # (k, count)
    return keywords


def load_actynet_cap(args, path):
    with open(path, 'r') as f:
        x = json.load(f)
    data = {}
    for k, v in x.items():
        for i, sent in enumerate(v['sentences']):
            data[f"{k}/{i}"] = {'target': sent}
    return data, {}


# mean pool the features across $max_seg segments
def mean_pool_segments(features, max_seg=3):
    if features.shape[0] >= max_seg:
        tmp_feat = []
        nps = int(np.floor(features.shape[0] // max_seg))  # numbers per segment
        for i in range(max_seg):
            if i != max_seg - 1:
                segment = features[nps * i:nps * (i + 1)]
            else:
                segment = features[nps * i:]
            segment = segment.mean(axis=0)
            tmp_feat.append(segment)
        features = np.array(tmp_feat)
    else:
        # 0 pad frames
        features = zero_pad(features, max_seg)
    return features


def zero_pad(features,n_feat):
    if features.shape[0] < n_feat:
        features = np.vstack((features,np.zeros((n_feat - features.shape[0], features.shape[1]))))
    return features

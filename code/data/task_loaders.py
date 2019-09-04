import json
import sys
from collections import defaultdict

from tqdm import tqdm
import numpy as np

from utils import cut_sample


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
    data = load_task1_text(args, path)
    path = path.parent.glob(f"features/*")
    path = [p for p in list(path) if p.name in args.feature_names or \
            (p.name in args.feature_name_map and args.feature_name_map[p.name] in args.feature_names)]
    features = {}
    for p in path:
        print(f"loading feature {p}")
        features[p.name] = load_feature(p, data.keys())
    all_feature = {}
    for k in features[list(features.keys())[0]].keys():
        all_feature[k] = {p: v[k] for p, v in features.items()}
    return {k: {**v, **all_feature[k]} for k, v in data.items()}, {}


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


def load_task1_text(args, path):
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
    data = load_task1_text(args, path)
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

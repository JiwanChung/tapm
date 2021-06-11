import json

import torch
import numpy as np
import h5py
from tqdm import tqdm

from exp import ex
from utils import cut_sample, transpose_dict


def load_actynetcap_feature(path, keys=None):
    # load hdf5
    data = h5py.File(path, 'r')
    vids = list(data.keys())
    if keys is not None:
        keys = list(keys)
        vids = list(set(keys) & set(vids))

    res = {}
    for vid in tqdm(vids):
        res[vid] = np.asarray(list(data[vid].values())[0])

    return res


def load_actynetcap_features(path, keys=None):
    paths = path.glob('*')
    features = {}
    for p in paths:
        name = p.stem.split('.')[-1]
        print(f"loading feature {name}")
        features[name] = load_actynetcap_feature(p, keys)  # frames, channels

    # features = dict(transpose_dict(features))
    return features


def merge_stride(f, segment):
    # LC
    L_new = f.shape[0] // segment
    if L_new < 1:
        diff = segment - f.shape[0]
        last = f[-1]
        last = np.repeat(last[np.newaxis, :], diff, axis=0)
        f = np.concatenate((f, last), axis=0)
        L_new = 1
    down_length = L_new * segment
    f = f[:down_length]
    f = f.reshape((segment, L_new, f.shape[1]))
    f = f.mean(axis=1)
    return f


@ex.capture
def load_actynetcap_text(path, sample, max_target_sent=5):
    if path.is_file():
        with open(path, 'r') as f:
            data = json.load(f)
        data = {k: [{
                        'frame': timestamp,
                        'target': sentence.strip(),
                        'vid': f"{k}_{timestamp[0]}",
                        'duration': v['duration'],
                    } for timestamp, sentence in zip(v['timestamps'], v['sentences'])]
                for k, v in data.items()}
    else:
        path = path.parent / f"{path.stem}_ids{path.suffix}"
        with open(path, 'r') as f:
            data = json.load(f)
        data = {k: [{'vid': f"{k}_{i}"} for i in range(max_target_sent)]
                for k in data}
    if sample:
        data = cut_sample(data, n=100)

    return data


def merge_features(data, features):
    # data: VID/'frame':[],'target':[]
    # prepare dense captioning as span prediction
    res = {}
    for group_id, group in data.items():
        group = [merge_feature(g, {f_name: f[group_id] for f_name, f in features.items()})
                  for g in group]
        res[group_id] = group
    return res


'''
dense version
@ex.capture
def merge_feature(g, f, max_segments=10):
    # sample_feature = f[list(f.keys())[0]]
    # video_length = sample_feature.shape[0]
    # c3d fps 16
    # fps = 16
    eps = 1e-10
    if 'frame' in g:  # train
        # print(g['frame'][0] / g['duration'], g['frame'][0] / g['duration'])
        g['frame'] = [min(g['frame'][0], g['duration']), min(g['frame'][1], g['duration'])]
        g['frame'] = [g['frame'][0] * max_segments // (g['duration'] + eps),
                      g['frame'][1] * max_segments // (g['duration'] + eps)]
        g['frame'] = torch.Tensor(g['frame']).long()
    for k, v in f.items():
        f[k] = merge_stride(v, max_segments)
    return {**g, **f}
'''


def merge_feature(g, f):
    if 'frame' in g:  # train
        sample_feat = f[list(f.keys())[0]]
        feature_len = sample_feat.shape[0]
        # print(g['frame'][0] / g['duration'], g['frame'][0] / g['duration'])
        g['frame'] = [min(g['frame'][0], g['duration']), min(g['frame'][1], g['duration'])]
        g['frame'] = [min(g['frame'][0], g['frame'][1]), max(g['frame'][0], g['frame'][1])]
        g['frame'] = [max(int(round(g['frame'][0] * feature_len / g['duration']) - 1), 0),
                      max(int(round(g['frame'][1] * feature_len / g['duration']) - 1), 0)]
        g['frame'] = torch.Tensor(g['frame']).long()
    for k, v in f.items():
        f[k] = segment_feature(v, g['frame'])
    return {**{k: v for k, v in g.items() if k != 'frame'}, **f}


@ex.capture
def segment_feature(v, frame, max_segments=10):
    if frame[0] == frame[1]:
        v = v[frame[0]][np.newaxis, :]
    else:
        v = v[frame[0]: frame[1]]   # L'C
    v = merge_stride(v, max_segments)
    return v


def load_actynetcap(path):
    data = load_actynetcap_text(path)
    feature_path = path.parent.parent / 'features'
    features = load_actynetcap_features(feature_path, data.keys())
    # data = {k: {**data[k], **features[k]} for k in data.keys()}
    data = merge_features(data, features)

    return data, {}

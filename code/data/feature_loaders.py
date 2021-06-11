import json
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import os
import numpy as np
import pickle as pkl

from exp import ex
from utils import copy


@ex.capture
def load_features(path, data_keys,
                  segment_pool_type, feature_names,
                  feature_name_map):
    path = list(path.parent.parent.glob("features/*"))
    if segment_pool_type is not None:
        pooler = PoolSegments([p for p in path if 'boundary' in p.stem], data_keys)
    else:
        pooler = lambda key, val: val
    path = [p for p in path if p.name in feature_names or \
            (p.name in feature_name_map and feature_name_map[p.name] in feature_names)]
    features = {}
    feature_names = copy(feature_names)
    reverse_map = {v: k for k, v in feature_name_map.items()}
    feature_names = [reverse_map[x] if x in reverse_map else x for x in feature_names]
    for p in path:
        if p.stem in feature_names:
            feature_loader_dict = {
                #'rcnn': partial(load_feature_rcnn_like, class_num=1600),
                #'scene': partial(load_feature_rcnn_like, class_num=365),
                #'vggish': load_feature_sound,
                'human_i3d': load_feature_human_id,
            }
            if p.stem in feature_loader_dict:
                d = feature_loader_dict[p.stem](p, data_keys, pooler=pooler)
            else:
                d = load_feature(p, data_keys, pooler)
            features[p.name] = d
    all_feature = {}
    if len(list(features.keys())) > 0:
        for k in features[list(features.keys())[0]].keys():
            all_feature[k] = {p: v[k] for p, v in features.items()}
    return all_feature


def load_feature(path, keys, pooler=lambda x: x):
    path_orig = path
    path = path.glob('*')
    path = {p.name: {p2.stem: p2 for p2 in p.glob('*')} for p in path}
    res = {}
    for k in tqdm(keys, desc=path_orig.stem):
        f1_name = '_'.join(k.split('_')[:-1])
        f2_name = k
        p = path[f1_name][f2_name]
        res[k] = pooler(k, np.load(p, allow_pickle=True))
    return res


def load_feature_human_id(path, keys, pooler=lambda x: x, feat_dim=1536, max_id=10):
    path_orig = path
    path = path.glob('*')
    path = {p.name: {p2.name: p2 for p2 in p.glob('*')} for p in path}
    res = {}
    for k in tqdm(keys, desc=path_orig.stem):
        f1_name = '_'.join(k.split('_')[:-1])
        f2_name = k
        p = path[f1_name][f2_name]
        feat = np.zeros([max_id, feat_dim])
        try:
            dic = pkl.load(open(os.path.join(p,'i3d_rgb.pkl') , 'rb'))
        except:
            dic = {}
        try:
            head_dic = pkl.load(open(os.path.join(str(p).replace(path_orig.name,'human_head') ,'head_feature.pkl') , 'rb'))
        except:
            head_dic = {}
        temp = []
        for key,v in dic.items():
            traj_len = v['len']
            head_feat = np.zeros([512])
            if (int(key) in head_dic.keys()) and len(head_dic[int(key)]) > 0:
                head_feat = np.squeeze(head_dic[int(key)])
            human_feat = np.concatenate([v['feat'], head_feat])
            temp.append((traj_len, human_feat))
        if len(temp) > 0:
            temp.sort(key=lambda x: x[0], reverse=True)
            temp = temp[:10]
        for i, t in enumerate(temp):
            feat[i] = t[1]
        res[k] = feat #pooler(k, np.load(p))
    return res


@ex.capture
def load_feature_rcnn_like(path, keys, class_num, pooler, num_workers):
    path_orig = path
    path = path.glob('*')
    path = {p.name: {p2.stem: p2 for p2 in p.glob('*')} for p in path}
    res = {}
    box_preprocess = BoxPreprocess(class_num)
    with Pool(num_workers) as pool:
        for k in tqdm(keys, desc=path_orig.stem):
            f1_name = '_'.join(k.split('_')[:-1])
            f2_name = k
            p = path[f1_name][f2_name]
            feat = np.load(p).tolist()
            feat = list(pool.map(box_preprocess, feat))
            res[k] = pooler(k, np.asarray(feat, dtype=np.float32))
    return res


@ex.capture
def load_feature_sound(path, keys, pooler, num_workers):
    path_orig = path
    path = path.glob('*')
    path = {p.name: {p2.stem: p2 for p2 in p.glob('*')} for p in path}
    res = {}
    with Pool(num_workers) as pool:
        for k in tqdm(keys, desc=path_orig.stem):
            f1_name = '_'.join(k.split('_')[:-1])
            f2_name = k
            p = path[f1_name][f2_name]
            res[k] = pooler(k, np.load(p).astype(np.float32))
    return res


def scale_down(v):
    return [i // 8 for i in v]  # temporary measure


def load_boundary(path, keys):
    path = path.glob('*.json')
    data = {}
    for p in path:
        with open(p, 'r') as f:
            video = json.load(f)
        data = {**data, **video}
    data = {k: scale_down(v) for k, v in data.items() if k in keys}
    return data


class BoxPreprocess:
    def __init__(self, class_num=1600):
        self.class_num = class_num

    def __call__(self, feat):
        # T=1, Object(20), Info(6 - prob, idx, bbox)
        ret = [0 for i in range(self.class_num)]
        for i in reversed(range(len(feat))):
            ret[int(feat[i][1])-1] = feat[i][0]
        return ret


class PoolSegments:
    @ex.capture
    def __init__(self, path, keys, segment_pool_type, max_segments):

        self.segment_pool_type = segment_pool_type
        self.max_seg = max_segments

        if self.segment_pool_type == 'boundary':
            assert len(path) > 0, "no boundary data file!"
            path = path[0]
            self.boundary = load_boundary(path, keys)

        self.mean_pool_segments = {
            'interval': self.mean_pool_segments_interval,
            'boundary': self.mean_pool_segments_boundary,
        }[self.segment_pool_type]

    def __call__(self, key, features):
        return self.mean_pool_segments(key, features)

    # mean pool the features across $max_seg segments
    def mean_pool_segments_interval(self, key, features):
        if features.shape[0] >= self.max_seg:
            tmp_feat = []
            nps = int(np.floor(features.shape[0] // self.max_seg))  # numbers per segment
            for i in range(self.max_seg):
                if i != self.max_seg - 1:
                    segment = features[nps * i:nps * (i + 1)]
                else:
                    segment = features[nps * i:]
                segment = segment.mean(axis=0)
                tmp_feat.append(segment)
            features = np.array(tmp_feat)
        else:
            # 0 pad frames
            features = zero_pad(features, self.max_seg)
        return features

    # mean pool the features across scene boundary
    def mean_pool_segments_boundary(self, key, features):
        boundary = self.boundary[key]
        if features.shape[0] >= self.max_seg:
            tmp_feat = []
            boundary = [0, *[i - 1 for i in boundary if i > 0]]  # starting from 0
            boundary = np.unique(boundary)  # no duplicate
            for i in range(self.max_seg):
                if len(boundary) > i:
                    if boundary[i] >= features.shape[0]:
                        break
                    if i != len(boundary) - 1:
                        segment = features[boundary[i]: boundary[i + 1]]
                    else:
                        segment = features[boundary[i]:]
                    segment = segment.mean(axis=0)
                    tmp_feat.append(segment)
            if len(tmp_feat) == 0:
                tmp_feat.append(features.mean(axis=0))  # one segment scene
            features = np.array(tmp_feat)
        # 0 pad frames
        features = zero_pad(features, self.max_seg)
        return features


def zero_pad(features, n_feat):
    if features.shape[0] < n_feat:
        features = np.vstack((features, np.zeros((n_feat - features.shape[0], features.shape[1]))))
    return features

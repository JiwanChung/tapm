import json
import pickle
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

import numpy as np

from exp import ex
from utils import cut_sample, flatten_dict, copy


@ex.capture
def load_vist_text(path, sample):
    with open(path, 'r') as f:
        data = json.load(f)
    data = data['annotations']
    res = {}
    image_ids = []
    story = defaultdict(dict)
    for i, row in enumerate(data):
        row = row[0]  # weird structure of VIST... : 1 element list, always
        idx = res['storylet_id'] if 'storylet_id' in res else i
        if 'story_id' in row:
            story[row['story_id']][int(row['worker_arranged_photo_order'])] = idx
        res[idx] = {'target': row['text'].strip(), 'image_id': row['photo_flickr_id'],
                    'album_id': row['album_id'], 'story_id': row['story_id']}
        image_ids.append(row['photo_flickr_id'])
    # id/{text, image_id}
    story = {idx: [v for k, v in sorted(row.items())] for idx, row in story.items()}
    if sample:
        story = cut_sample(story, n=20)
        keys = flatten_dict(story)
        res = {k: v for k, v in res.items() if k in keys}
        keys = [v['image_id'] for k, v in res.items()]
        image_ids = [v for v in image_ids if v in keys]
    image_ids = list(set(image_ids))
    #temp
    # image_ids.append('33116693')
    return res, image_ids, story


@ex.capture
def load_vist_features(path, mode, image_ids, feature_names, feature_name_map, cross_modal_names):
    path = list(path.glob('*'))
    feature_names = copy(feature_names)
    reverse_map = {v: k for k, v in feature_name_map.items()}
    feature_names = [reverse_map[x] if x in reverse_map else x for x in feature_names]

    features = {}
    skip_count = 0
    cross_modal_feature = []
    for p in path:
        name = p.stem
        if name in feature_names:
            if name in cross_modal_names:
                cross_modal_feature.append(name)
            else:
                print(f"loading feature {name}")
                features[name], feature_skip_count = load_vist_feature(name, p / mode, image_ids)
                skip_count = max(skip_count, feature_skip_count)
    all_feature = {}
    # accounting for missing values
    pivot = [f for f in feature_names if f not in cross_modal_names][0]
    prototypes = {k: np.zeros_like(list(v.values())[0]) for k, v in features.items()}
    zero_fill_count = 0

    if len(list(features.keys())) > 0:
        for k in features[pivot].keys():
            feat = {}
            for p, v in features.items():
                if k in v:
                    feat[p] = v[k]
                else:
                    # fill with zeros
                    feat[p] = prototypes[p]
                    zero_fill_count += 1
            all_feature[k] = feat

    print(f"zero_filled features: {zero_fill_count}")
    return all_feature, skip_count, cross_modal_feature


@ex.capture
def load_vist_feature(name, path, image_ids, repeat_vist_image=1, cut_feature_temporal_dim={}):
    images = {}
    skip_count = 0
    cut_temporal_dim = cut_feature_temporal_dim[name] if name in cut_feature_temporal_dim else None
    for idx in tqdm(image_ids):
        p = path / f"{idx}.npy"
        if p.is_file():
            images[idx] = load_single_feat(p, repeat_vist_image, cut_temporal_dim=cut_temporal_dim)
        else:
            p = path / f"{idx}.pkl"
            if p.is_file():
                images[idx] = load_single_feat(p, repeat_vist_image, numpy=False, cut_temporal_dim=cut_temporal_dim)
            else:
                skip_count += 1
    return images, skip_count


def load_single_feat(p, repeat_vist_image=1, numpy=True, cut_temporal_dim=None):
    if numpy:
        feat = np.load(p)
    else:
        with open(p, 'rb') as f:
            feat = pickle.load(f)
    assert feat.ndim < 3, f"too many feature dimensions: {feat.shape}"

    if feat.ndim == 1:
        # unsqueeze 0 dim to create temporal dim
        feat = np.repeat(feat[np.newaxis, :],
                                repeat_vist_image, axis=0)
    elif cut_temporal_dim is not None:
        feat = feat[:cut_temporal_dim]

    return feat


@ex.capture
def load_vilbert_feature(path, image_ids, repeat_vist_image=1):
    images = {}
    skip_count = 0
    for idx in tqdm(image_ids):
        ps = list(path.glob(f"*/{idx}.npy"))  # storylet_id/image_id
        story = {}
        if len(ps) > 0:
            for p in ps:
                if p.is_file():
                    # unsqueeze 0 dim to create temporal dim
                    story_name = p.parent.stem
                    story[story_name] = load_single_feat(p, repeat_vist_image)
        if len(story) > 0:
            images[idx] = story
        else:
            skip_count += 1
    return images, skip_count


@ex.capture
def get_cross_modal_feature(path, data, mode, feature_names, repeat_vist_image=1):
    skip_count = 0
    for feature_name in feature_names:
        feature_skip_count = 0
        feat = {}
        for k, v in data.items():
            p = path / feature_name / mode / v['story_id'] / f"{v['image_id']}.pkl"
            if p.is_file():
                with open(p, 'rb') as f:
                    feat[k] = np.repeat(pickle.load(f)[np.newaxis, :],
                                                    repeat_vist_image, axis=0)
            else:
                feature_skip_count += 1
        # dt[feature_name] = feat
        skip_count = max(skip_count, feature_skip_count)
        data = {k: {**v, feature_name: feat[k]} for k, v in data.items() if k in feat}

    return data, skip_count


def make_story(data, story_map):
    story = {}
    for idx, row in story_map.items():
        if len([k for k in row if k in data]) == len(row):
            story[idx] = [{**data[k], 'vid': k} for k in row]
    return story


def load_sis(path):
    # e.g. data/VIST/sis/train.story-in-sequence.json
    mode = path.name.split('.')[0]  # train, test, val
    feature_path = path.parent.parent
    data, image_ids, story = load_vist_text(path)
    features, skip_count, cross_modal_feature = load_vist_features(feature_path, mode, image_ids)
    former_length = len(list(data.keys()))
    # skip broken images
    data = {k: {**v, **features[v['image_id']]} for k, v in data.items()
            if v['image_id'] in features}
    # supporting only vilbert at the moment
    if len(cross_modal_feature) > 0:
        data, cross_modal_skip_count = get_cross_modal_feature(feature_path, data, mode, cross_modal_feature)
        print(f"cross_model_skip_count: {cross_modal_skip_count}")
    skipped_length = len(list(data.keys()))
    print(f"skipped {former_length - skipped_length} images")

    data = make_story(data, story)

    return data, {}

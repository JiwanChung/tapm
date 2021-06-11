import json
from pathlib import Path
from collections import defaultdict, OrderedDict

#from exp import ex


def leave_unique(dt):
    for k, v in dt.items():
        dt[k] = list(set(v))
    return dt


def save_file(obj, path, name):
    obj_path = path.parent / f"post_{path.stem}_{name}{path.suffix}"
    print(f"saving to {obj_path}")
    with open(obj_path, 'w') as f:
        json.dump(obj, f, indent=4)


#@ex.capture()
def main(filename):
    root = Path('~/projects/lsmdc/data/VIST/sis').expanduser()
    path = root / filename
    with open(path, 'r') as f:
        data = json.load(f)
    data = data['annotations']
    stories = defaultdict(list)
    for d in data:
        d = d[0]
        stories[d['story_id']].append(
            {'album_id': d['album_id'],
             'photo_flickr_id': d['photo_flickr_id'],
             'worker_arranged_photo_order': d['worker_arranged_photo_order'],
             'text': d['text']})
    albums = defaultdict(list)
    images = defaultdict(list)
    for k, v in stories.items():
        v = OrderedDict([(i['worker_arranged_photo_order'], i) for i in v])
        v = sorted(v.items())
        v = [i for k, i in v]
        texts = ' '.join([i['text'] for i in v])
        stories[k] = [texts]
        aid = v[0]['album_id']
        albums[aid].append(texts)
        iid = f"{aid}_{[i['photo_flickr_id'] for i in v]}"
        images[iid].append(texts)
    stories = leave_unique(stories)
    albums = leave_unique(albums)
    images = leave_unique(images)

    save_file(stories, path, 'stories')
    save_file(albums, path, 'albums')
    save_file(images, path, 'images')


if __name__ == "__main__":
    main('test.story-in-sequence.json')

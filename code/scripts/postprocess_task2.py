from collections import defaultdict
from pathlib import Path
import pickle
from tqdm import tqdm

from exp import ex


THRESHOLD = 0.5


def process_video(vids, blank_nums, relation):
    if relation is not None:
        N = relation.shape[0]
        vid = vids[0]
        video_name = vid[:vid.find('_')]
        blank_num = sum(blank_nums)

        people = [f'[{video_name}_PERSON_{i}]' for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if relation[i, j] >= THRESHOLD:
                    people[j] = people[i]

        assert len(people) == blank_num, f'people num {len(people)} does not match blank num {blank_num}'
        res = []
        for length in list(blank_nums):
            res.append(people[:length])
            people = people[length:]
        res = {vid: person for vid, person in zip(vids, res)}
    else:
        res = {vid: [] for vid in vids}
    return res


def make_groups(keys, chunk_size=5):
    movie_keys = defaultdict(list)
    for key in keys:
        name = key[:key.rfind('_')]
        movie_keys[name].append(key)
    movie_keys = {k: v for k, v in movie_keys.items()}
    movie_keys = {k: [v[i:i + chunk_size] for i in range(0, len(v), chunk_size)] \
                  for k, v in movie_keys.items()}
    res = {}
    for k, v in movie_keys.items():
        for i, chunk in enumerate(v):
            res[f"{k}_{i}"] = chunk
    return res


def load_blank_data(path):
    res = {}
    key_list = []
    with open(path / 'LSMDC16_annos_test_blank.csv', 'r') as f:
        for line in f:
            line = line.split('\t')
            key = line[0]
            text = line[-1].strip()
            res[key] = (key, text.count('[...]'))
            key_list.append(key)
    return res, key_list


@ex.capture
def main(path, relation_file):
    path = Path(path).parent
    blank_data, key_list = load_blank_data(path)
    with open(path / relation_file, 'rb') as f:
        relation_data = pickle.load(f)
    print(f"blanks before:{sum([i[1] for i in blank_data.values()])}, after:{sum([t.shape[0] for t in relation_data.values() if t is not None])}")

    group_keys = make_groups(blank_data.keys())
    blank_data = {k: [blank_data[i] for i in v] for k, v in group_keys.items()}
    blank_data = {k: (v[0][0], v) for k, v in blank_data.items()}
    blank_data = {k: v for k, v in blank_data.items() if v[0] in relation_data}

    data = {k: (*zip(*v[1]), relation_data[v[0]]) for k, v in blank_data.items()}
    outputs = {}
    data_keys = list(data.keys())
    for key in tqdm(data_keys, total=len(data_keys)):
        vids, blank_nums, relation = data[key]
        res = process_video(vids, blank_nums, relation)
        outputs = {**outputs, **res}

    outputs = [(key, outputs[key]) if key in outputs else (key, []) for key in key_list]

    task2_out_path = path / 'test_results.csv'
    # save file
    with open(task2_out_path, 'w') as f:
        for line in outputs:
            person = ','.join(line[1]) if len(line[1]) > 0 else '_'
            f.write(f"{line[0]}\t{person}\n")

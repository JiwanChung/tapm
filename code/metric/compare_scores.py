import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from metric_test import Metric


def main():
    root = Path('../../').resolve() / 'data' / 'LSMDC' / 'task1'
    tgt_path = root / 'LSMDC16_annos_test_someone.csv'
    tgt = {}
    key_list = defaultdict(list)
    with open(tgt_path, 'r') as f:
        for line in f:
            line = line.split('\t')
            vid = line[0].strip()
            tgt[vid] = line[-1].strip()
            video = '_'.join(vid.split('_')[:-1])
            key_list[video].append(vid)
    hypo1_path = root / 'publictest_with_aux_results.json'
    with open(hypo1_path, 'r') as f:
        hypo1 = json.load(f)
    hypo2_path = root / 'publictest_without_aux_results.json'
    with open(hypo2_path, 'r') as f:
        hypo2 = json.load(f)
    m = Metric(['meteor'], False, use_vist=False, no_tokenizer=False)
    key_list = list(key_list.values())
    key_list = [chunks(video, 5) for video in key_list]
    key_list = flatten(key_list)
    scores = []
    for group in tqdm(key_list):
        t = ' '.join([tgt[g] for g in group])
        h1 = ' '.join([hypo1[g] for g in group])
        h2 = ' '.join([hypo2[g] for g in group])
        s1 = m({'0': (h1, t)})['METEOR']
        s2 = m({'0': (h2, t)})['METEOR']
        scores.append((s1 - s2, group))
    scores = sorted(scores)
    print(scores[:10])
    with open('score_comparison.json', 'w') as f:
        json.dump(scores, f, indent=4)
    import ipdb; ipdb.set_trace()  # XXX DEBUG
    scores = scores


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flatten(li):
    return [item for sublist in li for item in sublist]



if __name__ == '__main__':
    main()

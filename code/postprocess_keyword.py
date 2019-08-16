import json
import argparse
from itertools import chain
from collections import Counter
from pathlib import Path
from nltk.corpus import stopwords

import nltk


def main(path, n=1000):
    path = Path(path).resolve()
    path = path.parent.parent.glob(f'{path.parent.name}/{path.name}')
    path = list(sorted(list(path)))[0]
    print(f"Loading {path}")
    with open(path, 'r') as f:
        keywords = json.load(f)
    keywords = chain(*[v['keyword'] for v in keywords.values()])
    x = Counter()
    for k in keywords:
        x[k] += 1
    cut_stop_words = get_stop_words()
    for sw in cut_stop_words:
        if sw in x:
            del x[sw]
    x_cut = Counter()
    for k, v in x.most_common(n):
        x_cut[k] = v

    save_path = path.parent / f"keyword_{n}.json"
    print(f"Saving to {save_path}")
    with open(save_path, 'w') as f:
        json.dump(x_cut, f, indent=4)


def get_stop_words():
    x = list(stopwords.words('english'))
    res = []
    for w in x:
        if not nltk.pos_tag([w])[0][1].startswith('PRP'):
            res.append(w)
    res += ["'", "[", "]", ")", "(", ".", '"']
    return res


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--n', default=1000, type=int)
    parser.add_argument('-p', '--path', default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    main(args.path, args.n)

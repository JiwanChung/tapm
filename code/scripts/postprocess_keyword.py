import json
import argparse
from itertools import chain
from collections import Counter
from pathlib import Path
from nltk.corpus import stopwords

import nltk

from common import process, parse, GetWords


def main():
    args = parse()
    path = args.path
    n = args.n
    path = Path(path).resolve()
    path = path.parent.parent.glob(f'{path.parent.name}/{path.name}')
    path = list(sorted(list(path)))[0]
    print(f"Loading {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    data = {k: v['keyword'] for k, v in data.items()}
    file_name = get_file_name(path.parent.name) if args.filename is None \
        else args.filename
    process(path.parent, data, GetTokenWords(), n, file_name)


def get_file_name(full_name):
    n = full_name.split('_')
    model_name_idx = [i for i, w in enumerate(n) if i == 'model'][0]
    return n[model_name_idx + 1]


class GetTokenWords(GetWords):
    def __init__(self):
        super(GetTokenWords, self).__init__()

    def __call__(self, sent):
        sent = ' '.join(sent)
        return super().__call__(sent)


if __name__ == '__main__':
    main()

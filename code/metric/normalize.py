import json
from collections import defaultdict
from pathlib import Path
import argparse

from vist_tokenizer import VistTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--name', default='test_results.json', type=str)
    parser.add_argument('-l', '--longest', action='store_true')
    args = parser.parse_args()
    return args


def break_tie(li, longest=False):
    if longest:
        res = ''
        for l in li:
            if len(l) >= len(res):
                res = l
    else:
        len_res = 9999
        for l in li:
            if len(l) < len_res:
                res = l
                len_res = len(res)
    return res


def main():
    args = parse_args()
    path = Path('~/projects/lsmdc/data/VIST/sis').expanduser()
    output = normalize(args.name, args.longest)
    with open(path / f"cut_{args.name}", 'w') as f:
        json.dump(output, f, indent=4)


def normalize(name, longest=False):
    path = Path('~/projects/lsmdc/data/VIST/sis').expanduser() / name
    with open(path, 'r') as f:
        data = json.load(f)['output_stories']
    albums = defaultdict(dict)
    for row in data:
        albums[row['album_id']]['_'.join(row['photo_sequence'])] = \
            row['story_text_normalized']

    # get shortest generation in album
    albums = {k: break_tie(list(v.values()),
                           longest=longest) for k, v in albums.items()}

    # tokenize
    tokenizer = VistTokenizer()
    albums = {k: [{'caption': v}] for k, v in albums.items()}
    albums = tokenizer.tokenize(albums)

    output = {'output_stories': [{'album_id': k, 'photo_sequence': [],
                                  'story_text_normalized': v[0]}
              for k, v in albums.items()]}

    return output


if __name__ == '__main__':
    main()

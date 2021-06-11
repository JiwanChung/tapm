import json
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model', default=None, type=str)
    args = parser.parse_args()
    return args


def load_json(path):
    with open(path, 'r') as f:
        x = json.load(f)
    return x


def load_ablation(path):
    x = load_json(path)
    return {k: v[0] if isinstance(v, list) else v for k, v in x.items()}


def main():
    args = parse_args()
    root = Path('./').resolve() / args.model
    key_list = load_json('key_list.json')

    ss = load_ablation(root / 'ss.json')
    joint_ss = load_ablation(root / 'joint_ss.json')
    no_ss = load_ablation(root / 'no_ss.json')

    x = {video: [[{'vid': vid, 'ss': ss[vid], 'joint_ss': joint_ss[vid], 'no_ss': no_ss[vid]}
          for vid in group]
         for group in groups]
         for video, groups in key_list.items()}

    with open(root / f'{args.model}_ablations.json', 'w') as f:
        json.dump(x, f, indent=4)
    print('done')


if __name__ == "__main__":
    main()

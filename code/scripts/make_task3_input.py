import json
from pathlib import Path

from exp import ex


def cut_key(key):
    k1 = key
    timestamps = key.split('_')[-1]
    k2, k3 = timestamps.split('-')

    return (k1, k2, k3, k2, k3)


@ex.capture(prefix='scripts')
def main(filename):
    root = Path('~/projects/lsmdc/data/LSMDC/task1').expanduser()
    path = root / filename
    with open(path, 'r') as f:
        data = json.load(f)
    out_path = Path(str(path.parent).replace('task1', 'task3')) / f"task3_input_{path.stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_path}")
    with open(out_path, 'w') as f:
        for key, val in data.items():
            ks = cut_key(key)
            ks = '\t'.join(ks)
            val = val.replace('SOMEONE', '[...]')
            f.write(f"{ks}\t{val}\n")

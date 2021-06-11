import json
import argparse
from pathlib import Path

from exp import ex


@ex.capture
def main(filename):
    root = Path('~/projects/lsmdc/data/LSMDC/task1').expanduser()
    with open(root / filename, 'r') as f:
        hypo = json.load(f)
    keyfile_name = 'LSMDC16_annos_blindtest.csv' if 'blind' in filename else 'LSMDC16_annos_test_someone.csv'
    aligned_vids = []
    with open(root / keyfile_name, 'r') as f:
        for row in f:
            row = row.split('\t')
            aligned_vids.append(row[0])
    res = [{"video_id": i + 1, "caption": hypo[vid]} for i, vid in enumerate(aligned_vids)]
    out_path = root / f'postprocessed_{filename}'
    print(f"saving to {out_path}")
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)

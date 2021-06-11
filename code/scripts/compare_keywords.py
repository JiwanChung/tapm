import json
from pathlib import Path
from collections import Counter
from itertools import product

from .common import load_data, get_counter, GetWords

from exp import ex


@ex.capture
def main(path):
    path = str(path).replace('task2', 'task1')
    path = Path(path)
    keyword_path = path.parent / 'keywords'
    keyword_files = list(keyword_path.glob('*.json'))
    keyword_files = [p for p in keyword_files if not p.name.startswith('keywords_total')]
    assert len(keyword_files) > 0, 'no keyword files'
    data = {}
    for p in keyword_files:
        name = p.name.split('.')[0]
        name, n = name.split('_')[1:3]
        n = int(n)
        with open(p, 'r') as f:
            datum = json.load(f)
        assert len(datum) == n, f"data length {len(datum)} does not match n {n}"
        data[name] = Counter(datum)

    '''
    print("pairwise word set difference")
    for k1, k2 in product(data.keys(), data.keys()):
        if k1 != k2:
            diff = set(data[k1]) - set(data[k2])
            print(f"    {k1}-{k2}: {len(diff)}/{len(data[k1])}")
    '''
    print("pairwise word set similarity")
    for (i1, k1), (i2, k2) in product(enumerate(data.keys()), enumerate(data.keys())):
        if i1 < i2:
            sim = set(data[k1]) & set(data[k2])
            print(f"    {k1}-{k2}: {len(sim)}/{len(data[k1])}")

    total_path = keyword_path / 'keywords_total.json'
    if total_path.is_file():
        with open(total_path, 'r') as f:
            whole = json.load(f)
        whole = Counter(whole)
    else:
        whole = load_data(path)
        whole = get_counter(whole, GetWords())
        with open(total_path, 'w') as f:
            json.dump(whole, f, indent=4)
    total = sum(whole.values())
    print("\n")
    print("left words")
    for k in data.keys():
        left = 0
        for word in data[k]:
            if word in whole:
                left += whole[word]
        print(f"    {k}: {left}/{total}({left/total:.3f})")

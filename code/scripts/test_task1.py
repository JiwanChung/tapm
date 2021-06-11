import json
from metric.metric import Metric

from exp import ex


@ex.capture()
def main(scripts, metrics, debug):
    filename = scripts['filename']
    root = f'../data/LSMDC/task1'
    with open(f'{root}/{filename}', 'r') as f:
        hypo = json.load(f)
    keyfile_name = 'LSMDC16_annos_blindtest.csv' if 'blind' in filename else 'LSMDC16_annos_test_someone.csv'
    data = {}
    with open(f'{root}/{keyfile_name}', 'r') as f:
        for row in f:
            row = row.split('\t')
            vid = row[0]
            tgt = row[-1].strip()
            data[vid] = tgt
    data = {k: (hypo[k], data[k]) for k in data.keys()}

    metric = Metric(metrics, debug)

    score_stats = metric.calculate(data)
    print(score_stats)

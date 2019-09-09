import torch
import numpy as np
from tqdm import tqdm

from itertools import chain
from collections import defaultdict

from tensor_utils import move_device
from data.batcher import decode_tensor
from sampler import get_sampler
from utils import jsonl_to_json, remove_sep
from metric.metric import Metric

def infer(args, model, loss_fn, optimizer, tokenizer, dataloaders,
             logger, print_output=False, epoch=0, subset=None):
    print("infering")
    infer_dict = {
        'task2_baseline': infer_task2,
    }
    if args.model.lower() in infer_dict:
        func = infer_dict[args.model.lower()]
    else:
        func = infer_task2
    return func(args, model, loss_fn, tokenizer, dataloaders,
                logger, print_output, epoch, subset)


def infer_task2(args, model, loss_fn, tokenizer, dataloaders,
                  logger, print_output=False, epoch=-1, subset=None):
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
    dataloader = dataloaders['test']
    total_length = len(dataloader)
    if subset is not None:
        subset = (len(dataloader) * subset) // args.batch_sizes['val']
        subset = max(1, subset)
        total_length = subset

    def erase_name(vid):
        vid_prefix = vid[:vid.find('_')]
        vid = vid[vid.find('_') + 1:]
        vid = vid[vid.find('.') + 1:]
        return f"{vid_prefix}_{vid}"

    def postprocess_relation(relation, blank_num, vids):
        # back to matrix
        # model/task2_baseline:Task2Baseline2/get_relation
        vid = vids[0]
        vid_prefix = vid[:vid.find('_')]
        relation = relation[:relation.find('---')].strip().split('\n')
        relation = [i.strip() for i in relation]
        relation = [i[1:i.find(']')] for i in relation]
        relation = [[int(j.strip()) for j in i.split(',')] for i in relation]
        relation = np.array(relation)
        N = relation.shape[0]
        people = [f'[{vid_prefix}_PERSON_{i}]' for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if relation[i, j] == 1:
                    people[j] = people[i]

        assert len(people) == blank_num.sum(), f'people num {len(people)} does not match blank num {blank_num.sum()}'
        res = []
        for length in list(blank_num):
            res.append(people[:length])
            people = people[:length]
        res = {erase_name(vid): person for vid, person in zip(vids, res)}
        res = sorted(res.items())

        return res

    total_relations = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=subset):
            batch = move_device(batch,
                                to=args.device)
            B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
            targets = batch['targets']
            _, _, _, _, relations = model(batch,
                                        batch_per_epoch=args.batch_per_epoch['train'])

            relations = [postprocess_relation(relation, blank_num, vid)
                         for relation, blank_num, vid
                         in zip(relations, batch.blank_num, batch.vid)]
            relations = chain(*relations)

            total_relations = [*relations, *total_relations]

    task2_out_path = args.data_path['train'].parent / 'test_results.csv'
    # save file
    with open(task2_out_path, 'w') as f:
        for line in total_relations:
            f.write(f"{line[0]}\t{','.join(line[1])}\n")

    return total_relations

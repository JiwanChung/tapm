import json

import torch
import numpy as np
from tqdm import tqdm
import pickle

from itertools import chain
from collections import defaultdict

from exp import ex
from tensor_utils import move_device
from data.batcher import decode_tensor
from sampler import get_sampler
from beam import TransformerBeamSearch
from utils import jsonl_to_json, remove_sep, recurse, concatenate_group, refactor_text


@ex.capture
def _infer(model, loss_fn, optimizer, tokenizer, dataloaders, logger,
           model_name, new_beam_search=False):
    print("infering")
    infer_dict = {
        'task2_baseline': infer_task2,
        'task2_ensemble': infer_task2,
        'task2_baseline2': infer_task2,
        'task2_baseline_cos_sim': infer_task2,
    }
    if model_name.lower() in infer_dict:
        func = infer_dict[model_name.lower()]
    else:
        if new_beam_search:
            func = infer_task1_beam
        else:
            func = infer_task1
    return func(model, loss_fn, tokenizer, dataloaders, logger)


@ex.capture
def infer_task2(model, loss_fn, tokenizer, dataloaders, logger,
                print_output=False, epoch=-1, subset=None,
                batch_sizes=None, device=None, sample=False,
                data_path=None, use_data=None):
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
    dataloader = dataloaders['test']
    total_length = len(dataloader)
    if subset is not None:
        subset = (len(dataloader) * subset) // batch_sizes['val']
        subset = max(1, subset)
        total_length = subset

    total_relations = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, total=subset):
            batch = move_device(batch,
                                to=device)
            _, _, _, _, relations = model(batch)
                                          # batch_per_epoch=args.batch_per_epoch[args.use_data[0]])
            hypo = relations['hypo']

            relations = {k[0]: v for k, v in zip(batch.vid, hypo)}
            total_relations = {**relations, **total_relations}

    task2_out_name = 'relation_text_only.pkl'
    if sample:
        task2_out_name = 'sample_relation_text_only.pkl'
    task2_out_path = data_path[use_data[0]].parent / task2_out_name
    with open(task2_out_path, 'wb') as f:
        pickle.dump(total_relations, f)

    return total_relations


@ex.capture
def infer_task1(model, loss_fn, tokenizer, dataloaders,
                logger, print_output=False, epoch=-1, subset=None,
                device=None, data_path=None, eval_set=False, sample=False,
                model_name=None, num_samples=1, sampling_method='',
                sample_eval_at_last=False, postprocess_duplicates=1,
                concat_group=False, vist_sample_longest=False, log_tag='',
                use_vist=False, sample_ema_coeff=1):
    model.eval()
    dataloader = dataloaders['test']

    texts = {}
    sampler = get_sampler(model)
    groups = {}
    albums = defaultdict(list)
    images = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, total=subset):
            batch = move_device(batch,
                                to=device)
            for idx, vids in zip(batch.id, batch.vid):
                groups[idx] = vids
            if hasattr(batch, 'album_id'):
                for group_id, album_ids in zip(batch.id, batch.album_id):
                    for album_id in album_ids:
                        albums[album_id].append(group_id)
            if hasattr(batch, 'image_id'):
                for group_id, image_ids in zip(batch.id, batch.image_id):
                    for image_id in image_ids:
                        images[group_id].append(image_id)

            if hasattr(batch, 'sentences'):
                _, logits, _, _, _, sampler_input = model(batch)
                            #batch_per_epoch=args.batch_per_epoch[args.use_data[0]])
            else:
                sampler_input = batch
            if logits is not None:
                hypos = sampler(sampler_input)
                if isinstance(hypos, tuple):
                    hypos, _, _ = hypos

                def calc(target, hypo, vid):
                    nonlocal texts
                    if not (target == tokenizer.pad_id).all():
                        target = decode_tensor(tokenizer, target, use_vist=use_vist,
                                               remove_past_sep=True)
                        hypo = decode_tensor(tokenizer, hypo, use_vist=use_vist,
                                             remove_past_sep=True)
                        target = remove_sep(target, tokenizer.sep_token)
                        hypo = remove_sep(hypo, tokenizer.sep_token)
                        if len(target) > 0:
                            texts[vid] = (hypo, target)
                        return target, hypo

                vid = [[v] for v in batch.id] if concat_group else batch.vid
                recurse(batch.targets.shape[:-1], batch.targets, hypos,
                        vid=vid, func=calc)
            else:
                import ipdb; ipdb.set_trace()  # XXX DEBUG

    if len(texts.keys()) > 0:
        if eval_set and not concat_group:
            texts = concatenate_group(texts, groups)
        if len(albums) > 0:
            albums = {k: sorted(list(set(v))) for k, v in albums.items()}
        else:
            albums = None
        if albums is not None and len(albums) > 0:
            targets = {k: v[1] for k, v in texts.items()}
            targets = refactor_text(targets, albums, images, return_list=True)
            texts = {k: v[0] for k, v in texts.items()}  # remove target
            texts = refactor_text(texts, albums, images, longest=vist_sample_longest)
        else:
            texts = {k: v[0] for k, v in texts.items()}  # remove target

    # filename = 'blindtest' if 'blind' in data_path['test'].name else 'publictest'

    subdir = 'small' if sample else 'full'
    task1_out_path = data_path['test'].parent / 'samples' / subdir
    task1_out_path.mkdir(exist_ok=True, parents=True)

    out_name = f'{log_tag}_{model_name}_{sampling_method}_{num_samples}_ema_coeff_{sample_ema_coeff}'
    if sample_eval_at_last:
        out_name += '_eval_at_last'
    if postprocess_duplicates != 1:
        out_name += '_penalty_duplicates'
    out_name += '.json'
    task1_out_path = task1_out_path / out_name
    # save file
    print(f"saving result to {task1_out_path}")
    with open(task1_out_path, 'w') as f:
        json.dump(texts, f, indent=4)
    model.train()

    return texts


@ex.capture
def infer_task1_beam(model, loss_fn, tokenizer, dataloaders,
                logger, print_output=False, epoch=-1, subset=None,
                device=None, data_path=None, eval_set=False, sample=False,
                model_name=None, num_samples=1, sampling_method='',
                sample_eval_at_last=False, postprocess_duplicates=1,
                concat_group=False, vist_sample_longest=False, log_tag='',
                use_vist=False, sample_ema_coeff=1):
    model.eval()
    dataloader = dataloaders['test']

    texts = {}
    groups = {}
    albums = defaultdict(list)
    images = defaultdict(list)
    beam_searcher = TransformerBeamSearch(model, dataloader.batch_size)
    with torch.no_grad():
        for batch in tqdm(dataloader, total=subset):
            batch = move_device(batch,
                                to=device)
            for idx, vids in zip(batch.id, batch.vid):
                groups[idx] = vids
            if hasattr(batch, 'album_id'):
                for group_id, album_ids in zip(batch.id, batch.album_id):
                    for album_id in album_ids:
                        albums[album_id].append(group_id)
            if hasattr(batch, 'image_id'):
                for group_id, image_ids in zip(batch.id, batch.image_id):
                    for image_id in image_ids:
                        images[group_id].append(image_id)

            if hasattr(batch, 'sentences'):
                _, logits, _, _, _, sampler_input = model(batch)
                            #batch_per_epoch=args.batch_per_epoch[args.use_data[0]])
            else:
                sampler_input = batch
            if logits is not None:
                hypos = beam_searcher(sampler_input)
                sg = [tokenizer.decode(hypos[0][i].cpu().numpy()) for i in range(5)]

                def calc(target, hypo, vid):
                    nonlocal texts
                    if not (target == tokenizer.pad_id).all():
                        target = decode_tensor(tokenizer, target, use_vist=use_vist,
                                               remove_past_sep=True)
                        hypo = decode_tensor(tokenizer, hypo, use_vist=use_vist,
                                             remove_past_sep=True)
                        target = remove_sep(target, tokenizer.sep_token)
                        hypo = remove_sep(hypo, tokenizer.sep_token)
                        if len(target) > 0:
                            texts[vid] = (hypo, target)
                        return target, hypo

                vid = [[v] for v in batch.id] if concat_group else batch.vid
                recurse(batch.targets.shape[:-1], batch.targets, hypos,
                        vid=vid, func=calc)
            else:
                import ipdb; ipdb.set_trace()  # XXX DEBUG

    if len(texts.keys()) > 0:
        if eval_set and not concat_group:
            texts = concatenate_group(texts, groups)
        if len(albums) > 0:
            albums = {k: sorted(list(set(v))) for k, v in albums.items()}
        else:
            albums = None
        if albums is not None and len(albums) > 0:
            targets = {k: v[1] for k, v in texts.items()}
            targets = refactor_text(targets, albums, images, return_list=True)
            texts = {k: v[0] for k, v in texts.items()}  # remove target
            texts = refactor_text(texts, albums, images, longest=vist_sample_longest)
        else:
            texts = {k: v[0] for k, v in texts.items()}  # remove target

    # filename = 'blindtest' if 'blind' in data_path['test'].name else 'publictest'

    subdir = 'small' if sample else 'full'
    task1_out_path = data_path['test'].parent / 'samples' / subdir
    task1_out_path.mkdir(exist_ok=True, parents=True)

    task1_out_path = task1_out_path / \
        f'{log_tag}_{model_name}_newbeam_{num_samples}_ema_coeff_{sample_ema_coeff}_eval_at_last_{sample_eval_at_last}.json'
    # save file
    print(f"saving result to {task1_out_path}")
    with open(task1_out_path, 'w') as f:
        json.dump(texts, f, indent=4)
    model.train()

    return texts

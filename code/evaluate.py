import copy

import torch
from tqdm import tqdm

from collections import defaultdict

from exp import ex
from tensor_utils import move_device
from data.batcher import decode_tensor
from sampler import get_sampler
from utils import (
    remove_sep, recurse, concatenate_group,
    refactor_text, merge_vist_output
)
from metric.metric import Metric


@ex.capture
def _evaluate(model, loss_fn, optimizer, tokenizer, dataloaders, logger,
              model_name,
              epoch=0, subset=None, key=None, **kwargs):
    print("evaluating")
    eval_prefix_dict = {
        'transformer_dis': evaluate_sample,
        'task2': evaluate_base,
    }
    eval_dict = {
        'mask_model': evaluate_mask,
        'subset_mask_model': evaluate_mask,
        'autoencoder': evaluate_sample,
        'variational_masking': evaluate_sample,
        'deterministic_masking': evaluate_sample,
        'lstm_keyword_lm': evaluate_sample,
        'hybrid_dis': evaluate_sample,
        'transformer_dis': evaluate_sample,
        'transformer_dis_ce': evaluate_sample,
        'transformer_dis_concat': evaluate_sample,
        'transformer_dis_concat_reg': evaluate_sample,
        'transformer_dis_pool': evaluate_sample,
        'transformer_dis_rank': evaluate_sample,
        'transformer_dis_rank_roll': evaluate_sample,
        'transformer_dis_rank_split': evaluate_sample,
        'transformer_dis_group': evaluate_sample,
        'transformer_dis_group_reg': evaluate_sample,
        'transformer_dis_group_reg_feat': evaluate_sample,
        'transformer_dis_ptr_gen': evaluate_sample,
        'transformer_dis_ptr_gen2': evaluate_sample,
        'task2_baseline': evaluate_base,
        'task2_ensemble': evaluate_base,
        'task2_baseline2': evaluate_base,
        'task2_baseline_cos_sim': evaluate_base,
        'task2_feature_concat': evaluate_base,
        'fib_model': evaluate_fib,
        'fib_no_gt_sos': evaluate_fib,
        'mc_model': evaluate_fib,
        'mc_no_gt_sos': evaluate_fib,
    }
    model_name = model_name.lower()
    func = evaluate_sample
    for k, v in eval_prefix_dict.items():
        if model_name.startswith(k):
            func = v
    if model_name in eval_dict:
        func = eval_dict[model_name]
    if 'print_output' in kwargs:
        kwargs.pop('print_output')
    return func(model, loss_fn, tokenizer, dataloaders, logger,
                epoch=epoch, subset=subset, key=key, **kwargs)


@ex.capture
def evaluate_base(model, loss_fn, tokenizer, dataloaders, logger,
                  batch_sizes, device, reg_coeff,
                  epoch=-1, subset=None, key=None, **kwargs):
    if key is None:
        key = 'val'
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
    dataloader = dataloaders[key]
    text_logging_step = 0
    total_length = len(dataloader)
    if subset is not None:
        subset = (len(dataloader) * subset) // batch_sizes[key]
        subset = max(1, subset)
        total_length = subset
    with torch.no_grad():
        for batch in tqdm(dataloader, total=subset):
            batch = move_device(batch,
                                to=device)
            B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
            targets = batch['targets']
            logits, targets, reg_loss, added_stats, words = model(batch)  # batch_per_epoch=batch_per_epoch[use_data[0]])
            if logits is not None:
                loss, stats = loss_fn(logits, targets, model)
                stats = {'language_loss': loss.item(), **stats}
                if reg_loss is not None:
                    final_loss = loss + reg_loss.sum() * reg_coeff
                    stats = {**stats, **{
                        'reg_loss': reg_loss.mean().item(),
                        'final_loss': final_loss.item()
                    }}
                else:
                    final_loss = loss
            else:
                final_loss = reg_loss
                stats = {
                    'reg_loss': reg_loss.mean().item(),
                    'final_loss': final_loss.item()
                }

            if added_stats is not None:
                stats = {**stats, **added_stats}

            n_step += 1

            for k, v in stats.items():
                if v is not None:
                    epoch_stats[k] = epoch_stats[k] + B * v
            epoch_stats['num'] = epoch_stats['num'] + B

            # log text for batch 1 ~ 5
            if n_step <= 5:
                if isinstance(words, dict):
                    words = words['text']
                    if (torch.is_tensor(words) or isinstance(words, list)):
                        for i in range(len(targets)):
                            string = ''
                            word = words[i]
                            if torch.is_tensor(word):
                                word = decode_tensor(tokenizer, word, remove_past_sep=True)
                            string += f"word: {word}"
                            string += "\n---"
                            target = decode_tensor(tokenizer, targets[i], remove_past_sep=True)
                            string += f"\ntarget: \n{target}"
                            logger(f"{key}/keyword/epoch{epoch}", string, text_logging_step)
                            text_logging_step += 1

            if subset is not None and n_step > subset:
                break

    num = epoch_stats.pop('num')
    epoch_stats = {k: v / num for k, v in epoch_stats.items()}

    return epoch_stats, words, None


@ex.capture
def evaluate_sample(model, loss_fn, tokenizer, dataloaders, logger,
                    metrics, batch_sizes, device, reg_coeff, eval_metric,
                    epoch=-1, subset=None, key=None, eval_generate=True,
                    eval_set=False, sample=False, concat_group=False,
                    use_vist=False, sampling_method='greedy'):
    print(f"sampling_method: {sampling_method}")
    if key is None:
        key = 'val'
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
    dataloader = dataloaders[key]
    metric = Metric(metrics, sample)
    text_logging_step = 0
    texts = {}
    total_length = len(dataloader)
    if subset is not None:
        subset = (len(dataloader) * subset) // batch_sizes[key]
        subset = max(1, subset)
        total_length = subset
    groups = {}
    albums = defaultdict(list)
    images = defaultdict(list)
    stories = defaultdict(list)
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
            if hasattr(batch, 'image_id') and hasattr(batch, 'album_id'):
                for group_id, album_ids, image_ids in zip(batch.id, batch.album_id, batch.image_id):
                    stories[f"{album_id}_{image_ids}"].append(group_id)

            B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
            targets = batch['targets']
            _, logits, targets, reg_loss, added_stats, sampler_input = model(batch)

            if logits is not None:
                loss, stats = loss_fn(logits, targets, model)
                stats = {'language_loss': loss.item(), **stats}
                if reg_loss is not None:
                    final_loss = loss + reg_loss.sum() * reg_coeff
                    stats = {**stats, **{
                        'reg_loss': reg_loss.mean().item(),
                        'final_loss': final_loss.item()
                    }}
                else:
                    final_loss = loss
                    stats = {'final_loss': loss.item(), **stats}
            else:
                final_loss = reg_loss
                stats = {
                    'reg_loss': reg_loss.mean().item(),
                    'final_loss': final_loss.item()
                }

            if added_stats is not None:
                stats = {**stats, **added_stats}

            n_step += 1

            for k, v in stats.items():
                if v is not None:
                    epoch_stats[k] = epoch_stats[k] + B * v
            epoch_stats['num'] = epoch_stats['num'] + B

            if eval_generate:
                if eval_metric and logits is not None:
                    hypos = logits.argmax(dim=-1)
                    sampler = get_sampler(model)
                    with torch.no_grad():
                        hypos = sampler(sampler_input)
                    if isinstance(hypos, tuple):
                        hypos, _, classifier_stats = hypos
                        stats = {**stats, **classifier_stats}

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

                def log_text(targets, hypos, keywords=None, vid=None):
                    nonlocal text_logging_step

                    target = decode_tensor(tokenizer, targets, use_vist=use_vist,
                                           remove_past_sep=True)
                    string = f"vid:{vid}\n"
                    if keywords is not None:
                        if isinstance(keywords, tuple):
                            keywords, scores = keywords
                            keyword = decode_tensor(tokenizer, keywords, remove_past_sep=True)
                            score = '/'.join(["%.2f" % j for j in scores.detach().cpu().numpy()])
                            string += f"keywords:{[f'({i}/{j})' for i, j in zip(keyword.split(), score.split('/'))]}"
                        else:
                            keyword = decode_tensor(tokenizer, keywords, remove_past_sep=True)
                            string += f"keywords:{keyword}"
                    hypo = decode_tensor(tokenizer, hypos, use_vist=use_vist,
                                         remove_past_sep=True)
                    string += f"\nhypo: {hypo}"
                    string += f"\ntarget: {target}"
                    logger(f"{key}/keyword/epoch{epoch}", string, text_logging_step)
                    text_logging_step += 1

                if n_step <= 5 and logits is not None:
                    hypos = logits.argmax(dim=-1)
                    sampler = get_sampler(model)
                    hypos = sampler(sampler_input)
                    if isinstance(hypos, tuple):
                        hypos, _, _ = hypos

                    keywords = None
                    targets = copy.deepcopy(batch.targets)  # avoiding shm leak
                    recurse(targets.shape[:-1], targets, hypos, keywords,
                            vid=batch.vid, func=log_text)
                    del batch.targets

                if subset is not None and n_step > subset:
                    break

    num = epoch_stats.pop('num')
    epoch_stats = {k: v / num for k, v in epoch_stats.items()}

    if len(texts.keys()) > 0:
        # lowercase all
        texts = {k: [v[0].lower(), v[1].lower()] for k, v in texts.items()}
        if eval_set and not concat_group:
            texts = concatenate_group(texts, groups)
        if len(albums) > 0:
            albums = {k: sorted(list(set(v))) for k, v in albums.items()}
        else:
            albums = None
        score_texts = texts
        if albums is not None and len(albums) > 0:
            targets = {k: v[1] for k, v in texts.items()}
            targets = refactor_text(targets, albums, images, return_list=True)
            texts = {k: v[0] for k, v in texts.items()}  # remove target
            texts = refactor_text(texts, albums, images)
            score_texts = merge_vist_output(texts, targets)
            album_score_texts = merge_vist_output(texts, targets, album=True)
            score_stats = metric.calculate(score_texts)
            score_stats = {f"photo_{k}": v for k, v in score_stats.items()}
            album_score_stats = metric.calculate(album_score_texts)
            epoch_stats = {**epoch_stats, **score_stats, **album_score_stats}
        else:
            score_stats = metric.calculate(score_texts)
            epoch_stats = {**epoch_stats, **score_stats}

    return epoch_stats, sampler_input, texts


@ex.capture
def evaluate_mask(model, loss_fn, tokenizer, dataloaders, logger,
                  device,
                  print_output=False, epoch=0, subset=None, key=None):
    if key is None:
        key = 'val'
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
    dataloader = dataloaders[key]
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = move_device(batch,
                                to=device)
            B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
            targets = batch['targets']
            loss, scores, ids = model(batch)

            n_step += 1

            stats = {'language_loss': loss.item()}
            for k, v in stats.items():
                epoch_stats[k] = epoch_stats[k] + B * v
            epoch_stats['num'] = epoch_stats['num'] + B
            # log text for batch 1 ~ 5
            if n_step <= 5 or print_output:
                for i in range(B):
                    keywords = decode_tensor(tokenizer, ids[i])
                    score = '/'.join(["%.2f" % j for j in scores[i].detach().cpu().numpy()])
                    target = decode_tensor(tokenizer, targets[i])
                    string = f"keywords:{[f'({i}/{j})' for i, j in zip(keywords.split(), score.split('/'))]}\ntarget:{target}"
                    logger(f"{key}/keyword/epoch{epoch}", string, (n_step - 1) * B + i)

    model.train()

    return epoch_stats, keywords, target


@ex.capture
def evaluate_fib(model, loss_fn, tokenizer, dataloaders, logger,
                    metrics, batch_sizes, device, reg_coeff, eval_metric,
                    epoch=-1, subset=None, key=None, **kwargs):
    if key is None:
        key = 'val'
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
    dataloader = dataloaders[key]
    groups = {}
    albums = defaultdict(list)
    images = defaultdict(list)
    stories = defaultdict(list)
    all_answers = []
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
            if hasattr(batch, 'image_id') and hasattr(batch, 'album_id'):
                for group_id, album_ids, image_ids in zip(batch.id, batch.album_id, batch.image_id):
                    stories[f"{album_id}_{image_ids}"].append(group_id)

            B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
            targets = batch['targets']
            _, logits, targets, reg_loss, added_stats, sampler_input = model(batch)
            if hasattr(sampler_input, 'answers'):
                all_answers = [*all_answers, *sampler_input.answers]

            if logits is not None:
                loss, stats = loss_fn(logits, targets, model)
                stats = {'language_loss': loss.item(), **stats}
                if reg_loss is not None:
                    final_loss = loss + reg_loss.sum() * reg_coeff
                    stats = {**stats, **{
                        'reg_loss': reg_loss.mean().item(),
                        'final_loss': final_loss.item()
                    }}
                else:
                    final_loss = loss
                    stats = {'final_loss': loss.item(), **stats}
            else:
                final_loss = reg_loss
                stats = {
                    'reg_loss': reg_loss.mean().item(),
                    'final_loss': final_loss.item()
                }

            if added_stats is not None:
                stats = {**stats, **added_stats}

            n_step += 1

            for k, v in stats.items():
                if v is not None:
                    if k == 'num_samples':
                        epoch_stats[k] = epoch_stats[k] + v
                    else:
                        epoch_stats[k] = epoch_stats[k] + B * v
            epoch_stats['num'] = epoch_stats['num'] + B
            epoch_stats['gt_accuracy'] = stats['accuracy'] * stats['num_samples']

    epoch_stats['gt_accuracy'] = epoch_stats['gt_accuracy'] / epoch_stats['num_samples']
    num = epoch_stats.pop('num')
    epoch_stats = {k: v / num for k, v in epoch_stats.items() if k != 'gt_accuracy'}
    if len(all_answers) > 0:
        all_answers = '\n'.join([f'hypo: {answer[0]}, tgt: {answer[1]}' for answer in all_answers])
        logger('val/epoch/answers', all_answers, epoch)

    return epoch_stats, None, None

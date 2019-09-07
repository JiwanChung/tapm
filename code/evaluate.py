import torch
from tqdm import tqdm

from collections import defaultdict

from tensor_utils import move_device
from data.batcher import decode_tensor
from sampler import get_sampler
from utils import jsonl_to_json, remove_sep
from metric.metric import Metric


def evaluate(args, model, loss_fn, optimizer, tokenizer, dataloaders,
             logger, print_output=False, epoch=0, subset=None):
    print("evaluating")
    eval_dict = {
        'mask_model': evaluate_mask,
        'subset_mask_model': evaluate_mask,
        'autoencoder': evaluate_sample,
        'variational_masking': evaluate_sample,
        'deterministic_masking': evaluate_sample,
        'lstm_keyword_lm': evaluate_sample,
        'task2_baseline': evaluate_base,
    }
    if args.model.lower() in eval_dict:
        func = eval_dict[args.model.lower()]
    else:
        func = evaluate_base
    return func(args, model, loss_fn, tokenizer, dataloaders,
                logger, print_output, epoch, subset)


def evaluate_base(args, model, loss_fn, tokenizer, dataloaders,
                  logger, print_output=False, epoch=-1, subset=None):
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
    dataloader = dataloaders['val']
    metric = Metric(args)
    text_logging_step = 0
    texts = {}
    total_length = len(dataloader)
    if subset is not None:
        subset = (len(dataloader) * subset) // args.batch_sizes['val']
        subset = max(1, subset)
        total_length = subset
    with torch.no_grad():
        for batch in tqdm(dataloader, total=subset):
            batch = move_device(batch,
                                to=args.device)
            B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
            targets = batch['targets']
            logits, targets, reg_loss, added_stats, words = model(batch,
                                                                     batch_per_epoch=args.batch_per_epoch['train'])
            if logits is not None:
                loss, stats = loss_fn(logits, targets)
                stats = {'nll_loss': loss.item(), **stats}
                if reg_loss is not None:
                    final_loss = loss + reg_loss.sum() * args.reg_coeff
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
                targets = batch['targets']
                for i in range(targets.shape[0]):
                    word = words[i]
                    if torch.is_tensor(word):
                        word = decode_tensor(tokenizer, word, remove_past_sep=True)
                    target = decode_tensor(tokenizer, targets[i], remove_past_sep=True)
                    string = ''
                    if word is not None:
                        string += f"word: {word}"
                    string += "\n---"
                    string += f"\ntarget: \n{target}"
                    logger(f"eval/keyword/epoch{epoch}", string, text_logging_step)
                    text_logging_step += 1

            if subset is not None and n_step > subset:
                break

    num = epoch_stats.pop('num')
    epoch_stats = {k: v / num for k, v in epoch_stats.items()}

    return epoch_stats, words, texts

def evaluate_sample(args, model, loss_fn, tokenizer, dataloaders,
                  logger, print_output=False, epoch=-1, subset=None):
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
    dataloader = dataloaders['val']
    metric = Metric(args)
    text_logging_step = 0
    texts = {}
    total_length = len(dataloader)
    if subset is not None:
        subset = (len(dataloader) * subset) // args.batch_sizes['val']
        subset = max(1, subset)
        total_length = subset
    with torch.no_grad():
        for batch in tqdm(dataloader, total=subset):
            batch = move_device(batch,
                                to=args.device)
            B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
            targets = batch['targets']
            logits, targets, reg_loss, added_stats, sampler_input = model(batch,
                                                                     batch_per_epoch=args.batch_per_epoch['train'])
            if logits is not None:
                loss, stats = loss_fn(logits, targets)
                stats = {'nll_loss': loss.item(), **stats}
                if reg_loss is not None:
                    final_loss = loss + reg_loss.sum() * args.reg_coeff
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

            if args.eval_metric and logits is not None:
                hypos = logits.argmax(dim=-1)
                sampler = get_sampler(args, model)
                hypos= sampler(sampler_input)
                if isinstance(hypos, tuple):
                    hypos, classifier_stats = hypos
                    stats = {**stats, **classifier_stats}

                def calc(target, hypo, vid):
                    nonlocal texts
                    if not (target == tokenizer.pad_id).all():
                        target = decode_tensor(tokenizer, target, remove_past_sep=True)
                        hypo = decode_tensor(tokenizer, hypo, remove_past_sep=True)
                        target = remove_sep(target, tokenizer.sep_token)
                        hypo = remove_sep(hypo, tokenizer.sep_token)
                        if len(target) > 0:
                            texts[vid] = (hypo, target)
                        return target, hypo

                def recurse(shape, *args, vid='0', func=None):
                    if len(shape) > 0:
                        for i in range(shape[0]):
                            if not isinstance(vid, str):
                                if len(vid) > i:
                                    vid_new = vid[i]
                                    recurse(shape[1:], *list([v[i] if v is not None else v for v in args]),
                                        vid=vid_new, func=func)
                            else:
                                vid_new = f'{vid}_{i}'
                                recurse(shape[1:], *list([v[i] if v is not None else v for v in args]),
                                        vid=vid_new, func=func)
                    else:
                        func(*args, vid=vid)

                recurse(batch.targets.shape[:-1], batch.targets, hypos,
                        vid=batch.vid, func=calc)

            for k, v in stats.items():
                epoch_stats[k] = epoch_stats[k] + B * v
            epoch_stats['num'] = epoch_stats['num'] + B

            def log_text(targets, hypos, keywords=None, vid=None):
                nonlocal text_logging_step

                target = decode_tensor(tokenizer, targets, remove_past_sep=True)
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
                hypo = decode_tensor(tokenizer, hypos, remove_past_sep=True)
                string += f"\nhypo: {hypo}"
                string += f"\ntarget: {target}"
                logger(f"eval/keyword/epoch{epoch}", string, text_logging_step)
                text_logging_step += 1

            # log text for batch 1 ~ 5
            if n_step <= 5 and logits is not None:
                hypos = logits.argmax(dim=-1)
                sampler = get_sampler(args, model)
                hypos = sampler(sampler_input)
                if isinstance(hypos, tuple):
                    hypos, _ = hypos

                keywords = None
                recurse(batch.targets.shape[:-1], batch.targets, hypos, keywords,
                        vid=batch.vid, func=log_text)

            if subset is not None and n_step > subset:
                break

    num = epoch_stats.pop('num')
    epoch_stats = {k: v / num for k, v in epoch_stats.items()}

    # remove tensor padding
    if len(texts.keys()) > 0:
        score_stats = metric.calculate(texts)
        epoch_stats = {**epoch_stats, **score_stats}

    return epoch_stats, sampler_input, texts


def evaluate_mask(args, model, loss_fn, tokenizer, dataloaders, logger,
                  print_output=False, epoch=0, subset=None):
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
    dataloader = dataloaders['val']
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = move_device(batch,
                                to=args.device)
            B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
            targets = batch['targets']
            loss, scores, ids = model(batch)

            n_step += 1

            stats = {'nll_loss': loss.item()}
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
                    logger(f"eval/keyword/epoch{epoch}", string, (n_step - 1) * B + i)

    model.train()

    return epoch_stats, keywords, target

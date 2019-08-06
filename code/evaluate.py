import torch

from collections import defaultdict

from tensor_utils import move_device
from transformers import decode_tensor


def evaluate(args, model, loss_fn, optimizer, tokenizer, dataloaders, logger, print_output=False):
    return {
        'mask_model': evaluate_mask,
        'autoencoder': evaluate_base
    }[args.model.lower()](args, model, loss_fn, tokenizer, dataloaders, logger, print_output)


def evaluate_base(args, model, loss_fn, tokenizer, dataloaders, logger, print_output=False):
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
    with torch.no_grad():
        for batch in dataloaders['val']:
            batch = move_device(*batch,
                                to=args.device)
            B = batch[0].shape[0]
            targets = batch[-1]
            logits, reg_loss, scores, keywords = model(*batch)
            loss, stats = loss_fn(logits, targets)

            if reg_loss is not None:
                final_loss = loss + reg_loss.sum() * args.reg_coeff
            else:
                final_loss = loss
            n_step += 1

            if reg_loss is not None:
                stats = {**stats, **{
                    'nll_loss': loss.item(),
                    'reg_loss': reg_loss.mean().item(),
                    'final_loss': final_loss.item()
                }}
            else:
                stats = {**stats, **{
                    'nll_loss': loss.item()}}
            if scores is not None:
                stats = {**stats, **{
                        'scores': scores.mean().item()}}
            for k, v in stats.items():
                epoch_stats[k] = epoch_stats[k] + B * v
            epoch_stats['num'] = epoch_stats['num'] + B
            # log text for batch 0
            if n_step == 0:
                hypos = logits.argmax(dim=-1)
                for i in range(B):
                    if keywords is not None:
                        keyword = decode_tensor(tokenizer, keywords[i])
                        logger(f"train/keyword", keyword, n_step)
                    hypo = decode_tensor(tokenizer, hypos[i])
                    logger(f"train/hypo", hypo, n_step)
                    target = decode_tensor(tokenizer, targets[i])
                    logger(f"train/target", target, n_step)

    num = epoch_stats.pop('num')
    epoch_stats = {k: v / num for k, v in epoch_stats.items()}
    for name, val in epoch_stats.items():
        logger(f"train/epoch/{name}", val, n_step)
    model.train()

    return epoch_stats, keywords, target


def evaluate_mask(args, model, loss_fn, tokenizer, dataloaders, logger, print_output=False):
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
    with torch.no_grad():
        for batch in dataloaders['val']:
            batch = move_device(*batch,
                                to=args.device)
            B = batch[0].shape[0] if torch.is_tensor(batch) else len(batch[0])
            targets = batch[-1]
            loss, scores, ids = model(*batch)

            n_step += 1

            stats = {'nll_loss': loss.item()}
            for k, v in stats.items():
                epoch_stats[k] = epoch_stats[k] + B * v
            epoch_stats['num'] = epoch_stats['num'] + B
            # log text for batch 0
            if n_step == 0:
                for i in range(B):
                    keywords = decode_tensor(tokenizer, ids[i])
                    logger(f"train/keyword", keywords, n_step)
                    score = '/'.join([str(j) for j in scores[i].detach().cpu().numpy()])
                    logger(f"train/score", score, n_step)
                    target = decode_tensor(tokenizer, targets[i])
                    logger(f"train/target", target, n_step)
            if print_output:
                for i in range(B):
                    keywords = decode_tensor(tokenizer, ids[i])
                    score = '/'.join([str(j) for j in scores[i].detach().cpu().numpy()])
                    target = decode_tensor(tokenizer, targets[i])
                    print(f"keywords:{keywords}")
                    print(f"score:{score}")
                    print(f"target:{target}")

    num = epoch_stats.pop('num')
    epoch_stats = {k: v / num for k, v in epoch_stats.items()}
    for name, val in epoch_stats.items():
        logger(f"train/epoch/{name}", val, n_step)
    model.train()

    return epoch_stats, keywords, target

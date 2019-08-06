import torch

from collections import defaultdict

from tensor_utils import move_device
from transformers import decode_tensor


def evaluate(args, model, loss_fn, tokenizer, dataloaders, logger):
    epoch_stats = defaultdict(float)
    model.eval()
    n_step = 0
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

        stats = {**stats, **{
            'loss': loss.item(),
            'final_loss': final_loss.item()
        }}
        if reg_loss is not None:
            stats = {**stats, **{
                    'reg_loss': reg_loss.mean().item()}}
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
                    logger(f"Text/train/keyword", keyword, n_step)
                hypo = decode_tensor(tokenizer, hypos[i])
                logger(f"Text/train/hypo", hypo, n_step)
                target = decode_tensor(tokenizer, targets[i])
                logger(f"Text/train/target", target, n_step)

    num = epoch_stats.pop('num')
    epoch_stats = {k: v / num for k, v in epoch_stats.items()}
    for name, val in epoch_stats.items():
        logger(f"train/epoch/{name}", val, n_step)
    model.train()

    return epoch_stats

import torch

from collections import defaultdict

from tensor_utils import move_device
from transformers import decode_tensor


def evaluate(args, model, loss_fn, optimizer, tokenizer, dataloaders, logger, print_output=False):
    print("evaluating")
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
            # log text for batch 1 ~ 5
            if n_step <= 5:
                hypos = logits.argmax(dim=-1)
                for i in range(B):
                    if keywords is not None:
                        keyword = decode_tensor(tokenizer, keywords[i])
                        logger(f"eval/keyword", keyword, n_step)
                    hypo = decode_tensor(tokenizer, hypos[i])
                    logger(f"eval/hypo", hypo, n_step)
                    target = decode_tensor(tokenizer, targets[i])
                    logger(f"eval/target", target, n_step)

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
            # log text for batch 1 ~ 5
            if n_step <= 5:
                for i in range(B):
                    keywords = decode_tensor(tokenizer, ids[i])
                    score = '/'.join(["%.2f" % j for j in scores[i].detach().cpu().numpy()])
                    target = decode_tensor(tokenizer, targets[i])
                    string = f"keywords:{[f'({i}/{j})' for i, j in zip(keywords.split(), score.split('/'))]}\ntarget:{target}"
                    logger(f"eval/keyword", string, n_step)
            if print_output:
                for i in range(B):
                    keywords = decode_tensor(tokenizer, ids[i])
                    score = '/'.join(["%.2f" % j for j in scores[i].detach().cpu().numpy()])
                    target = decode_tensor(tokenizer, targets[i])
                    string = f"keywords:{[f'({i}/{j})' for i, j in zip(keywords.split(), score.split('/'))]}\ntarget:{target}"
                    logger(f"eval/keyword", string, n_step)

    model.train()

    return epoch_stats, keywords, target

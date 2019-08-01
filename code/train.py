import torch

from collections import defaultdict

from tensor_utils import move_device
from transformers import decode_tensor


def train(args, model, loss_fn, optimizer, tokenizer, dataloaders, logger):
    print(f"training steps: {len(dataloaders['train'])}")
    for epoch in range(args.max_epoch):
        epoch_stats = defaultdict(float)
        model.train()
        for n_step, (sentences, lengths, targets) in enumerate(dataloaders['train']):
            sentences, lengths, targets = move_device(sentences, lengths, targets,
                                                      to=args.device)
            B = sentences.shape[0]
            logits, reg_loss, scores, keywords = model(sentences, lengths)
            loss, stats = loss_fn(logits, targets)

            if reg_loss is not None:
                final_loss = loss + reg_loss.sum() * args.reg_coeff
            else:
                final_loss = loss
            final_loss.backward()
            optimizer[1].step()
            optimizer[0].step()

            stats = {**stats, **{
                'loss': loss.item(),
                'final_loss': final_loss.item()
            }}
            if reg_loss is not None:
                stats = {**stats, **{
                'reg_loss': reg_loss.mean().item()}}
            for k, v in stats.items():
                epoch_stats[k] = epoch_stats[k] + B * v
            epoch_stats['num'] = epoch_stats['num'] + B

            for name, val in stats.items():
                logger(f"iters/{name}", val, n_step)
            if args.log_text_every > 0 and \
                    ((n_step + 1) % args.log_text_every == 0):
                hypos = logits.argmax(dim=-1)
                for i in range(B):
                    if keywords is not None:
                        keyword = decode_tensor(tokenizer, keywords[i])
                        logger(f"Text/keyword", keyword, n_step)
                    hypo = decode_tensor(tokenizer, hypos[i])
                    logger(f"Text/hypo", hypo, n_step)
                    target = decode_tensor(tokenizer, targets[i])
                    logger(f"Text/target", target, n_step)


        num = epoch_stats.pop('num')
        epoch_stats = {k: v / num for k, v in epoch_stats.items()}
        for name, val in epoch_stats.items():
            logger(f"epoch/{name}", val, n_step)

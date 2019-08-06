import torch

from collections import defaultdict

from tensor_utils import move_device
from transformers import decode_tensor
from evaluate import evaluate


def train(args, model, loss_fn, optimizer, tokenizer, dataloaders, logger):
    print(f"training steps: {len(dataloaders['train'])}")
    n_step = 0
    for epoch in range(args.max_epoch):
        print(f"training {epoch}th epoch")
        epoch_stats = defaultdict(float)
        model.train()
        for batch in dataloaders['train']:
            batch = move_device(*batch,
                                to=args.device)
            B = batch[0].shape[0]
            targets = batch[-1]
            logits, targets, reg_loss, added_stats, keywords = model(*batch)
            loss, stats = loss_fn(logits, targets)

            if reg_loss is not None:
                final_loss = loss + reg_loss.sum() * args.reg_coeff
            else:
                final_loss = loss
            final_loss.backward()
            optimizer[1].step()
            optimizer[0].step()
            n_step += 1

            if reg_loss is not None:
                stats = {**stats, **{
                    'nll_loss': loss.item(),
                    'reg_loss': reg_loss.mean().item() * args.reg_coeff,
                    'final_loss': final_loss.item()
                }}
            else:
                stats = {**stats, **{
                    'nll_loss': loss.item()}}
            if added_stats is not None:
                stats = {**stats, **added_stats}
            for k, v in stats.items():
                epoch_stats[k] = epoch_stats[k] + B * v
            epoch_stats['num'] = epoch_stats['num'] + B

            # log lr
            stats['lr'] = optimizer[0].get_lr()

            for name, val in stats.items():
                logger(f"train/iters/{name}", val, n_step)
            if args.log_text_every > 0 and \
                    ((n_step + 1) % args.log_text_every == 0):
                for i in range(B):
                    if keywords is not None:
                        keyword = {k: decode_tensor(tokenizer, v[i])
                                   if v[i].dtype == torch.long
                                   else '/'.join(["%.2f" % j for j in v[i].detach().cpu().numpy()])
                                    for k, v in keywords.items()}
                        string = '\n'.join(list([f"{k}:{v}"
                                                 for k, v in keyword.items()]))
                        logger(f"train/keyword", string, n_step)

        num = epoch_stats.pop('num')
        epoch_stats = {k: v / num for k, v in epoch_stats.items()}
        for name, val in epoch_stats.items():
            logger(f"train/epoch/{name}", val, epoch)
        eval_stats, _, _ = evaluate(args, model, loss_fn, optimizer, tokenizer,
                                    dataloaders, logger, epoch=epoch)

        num = eval_stats.pop('num')
        eval_stats = {k: v / num for k, v in eval_stats.items()}
        for name, val in eval_stats.items():
            logger(f"eval/epoch/{name}", val, epoch)

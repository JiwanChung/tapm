import torch
from tqdm import tqdm

from collections import defaultdict

from tensor_utils import move_device
from data.batcher import decode_tensor
from evaluate import evaluate
from ckpt import save_ckpt
from extract_keyword import extract_and_save_all


def train(args, model, loss_fn, optimizer, tokenizer, dataloaders, logger):
    dataloader = dataloaders['train']
    print(f"training steps: {len(dataloader)}")
    n_step = 0
    lowest_loss = float('inf')
    for epoch in range(args.max_epoch):
        print(f"training {epoch}th epoch")
        epoch_stats = defaultdict(float)
        epoch_stats['num'] = 0
        model.train()
        if hasattr(model, 'epoch_update'):
            model.epoch_update(epoch)
        for batch in tqdm(dataloader, total=len(dataloader)):
            optimizer.zero_grad()
            batch = move_device(batch, to=args.device)
            B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
            targets = batch['targets']
            logits, targets, reg_loss, added_stats, keywords = model(batch,
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
                final_loss = reg_loss * args.reg_coeff
                stats = {
                    'reg_loss': reg_loss.mean().item(),
                    'final_loss': final_loss.item()
                }

            final_loss.backward()
            optimizer.clip_grad()
            optimizer.step()
            optimizer.scheduler.step()

            if added_stats is not None:
                stats = {**stats, **added_stats}

            n_step += 1

            for k, v in stats.items():
                if v is not None:
                    epoch_stats[k] = epoch_stats[k] + B * v
            epoch_stats['num'] = epoch_stats['num'] + B

            # log lr
            stats['lr'] = optimizer.get_lr()

            for name, val in stats.items():
                logger(f"train/iters/{name}", val, n_step)
            '''
            if args.log_text_every > 0 and \
                    ((n_step + 1) % args.log_text_every == 0):
                if keywords is not None and model.use_keyword:
                    if isinstance(keywords, tuple):
                        keywords, scores = keywords
                    for i in range(B):
                        keyword = decode_tensor(tokenizer, keywords[i], remove_past_sep=True)
                        target = decode_tensor(tokenizer, batch['targets'][i], remove_past_sep=True)
                        string = f"keyword:{keyword}\ntarget:{target}"
                        logger(f"train/keyword/epoch{epoch}", string, (n_step - 1) * B + i)
            '''

        num = epoch_stats.pop('num')
        epoch_stats = {k: v / num for k, v in epoch_stats.items()}
        for name, val in epoch_stats.items():
            logger(f"train/epoch/{name}", val, epoch)
        eval_stats, _, _ = evaluate(args, model, loss_fn, optimizer, tokenizer,
                                    dataloaders, logger, epoch=epoch,
                                    subset=None if (epoch + 1) % args.eval_every == 0 else args.eval_subset)

        for name, val in eval_stats.items():
            logger(f"eval/epoch/{name}", val, epoch)

        current_loss = eval_stats['final_loss']
        save_ckpt(args, epoch, current_loss, model, tokenizer)

        if args.extract_keyword and current_loss < lowest_loss:
            extract_and_save_all(args, model, tokenizer, dataloaders)
        lowest_loss = min(lowest_loss, current_loss)

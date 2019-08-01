from collections import defaultdict


def train(args, model, loss_fn, optimizer, tokenizer, dataloaders, logger):

    for epoch in range(args.max_epoch):
        epoch_stats = defaultdict(lambda x: 0)
        for n_step, (sentences, lengths, targets) in enumerate(dataloaders['train']):
            B = sentences.shape[0]
            logits, reg_loss, scores, keywords = model(sentences, lengths)
            loss, stats = loss_fn(logits, targets)

            final_loss = loss + reg_loss * args.reg_coeff
            final_loss.backward()
            optimizer.step()

            stats = {**stats, **{
                'loss': loss.item(),
                'reg_loss': reg_loss.item(),
                'final_loss': final_loss.item()
            }}
            epoch_stats = {epoch_stats[k] + B * v for k, v in stats.items()}
            epoch_stats['num'] = epoch_stats['num'] + B
            for name, val in stats.items():
                logger(f"iters/{name}", val, n_step)

        num = epoch_stats.pop('num')
        epoch_stats = {k: v / num for k, v in epoch_stats.items()}
        for name, val in epoch_stats.items():
            logger(f"epoch/{name}", val, n_step)

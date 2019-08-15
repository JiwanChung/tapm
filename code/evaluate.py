import torch
from tqdm import tqdm

from collections import defaultdict

from tensor_utils import move_device
from data.batcher import decode_tensor
from sampler import get_sampler


def evaluate(args, model, loss_fn, optimizer, tokenizer, dataloaders,
             logger, print_output=False, epoch=0):
    print("evaluating")
    eval_dict = {
        'mask_model': evaluate_mask,
        'subset_mask_model': evaluate_mask,
        'autoencoder': evaluate_base,
        'variational_masking': evaluate_base,
        'deterministic_masking': evaluate_base,
        'lstm_keyword_lm': evaluate_base
    }
    if args.model.lower() in eval_dict:
        func = eval_dict[args.model.lower()]
    else:
        func = evaluate_base
    return func(args, model, loss_fn, tokenizer, dataloaders,
                logger, print_output, epoch)


def evaluate_base(args, model, loss_fn, tokenizer, dataloaders,
                  logger, print_output=False, epoch=-1):
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
            logits, targets, reg_loss, added_stats, keywords = model(batch,
                                                                     batch_per_epoch=args.batch_per_epoch['train'])
            loss, stats = loss_fn(logits, targets)

            if loss is not None:
                if reg_loss is not None:
                    final_loss = loss + reg_loss.sum() * args.reg_coeff
                else:
                    final_loss = loss
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
                # log text for batch 1 ~ 5
                if n_step <= 5:
                    hypos = logits.argmax(dim=-1)
                    if args.eval_generate:
                        sampler = get_sampler(args, model)
                        hypos = sampler(keywords)

                    if keywords is not None:
                        if isinstance(keywords, tuple):
                            keywords, scores = keywords
                        for i in range(B):
                            keyword = decode_tensor(tokenizer, keywords[i], remove_past_sep=True)
                            score = '/'.join(["%.2f" % j for j in scores[i].detach().cpu().numpy()])
                            target = decode_tensor(tokenizer, batch['targets'][i], remove_past_sep=True)
                            string = f"keywords:{[f'({i}/{j})' for i, j in zip(keyword.split(), score.split('/'))]}\ntarget: {target}"
                            logger(f"eval/keyword/epoch{epoch}", string, (n_step - 1) * B + i)

    return epoch_stats, keywords, None


def evaluate_mask(args, model, loss_fn, tokenizer, dataloaders, logger, print_output=False, epoch=0):
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
            if n_step <= 5:
                for i in range(B):
                    keywords = decode_tensor(tokenizer, ids[i])
                    score = '/'.join(["%.2f" % j for j in scores[i].detach().cpu().numpy()])
                    target = decode_tensor(tokenizer, targets[i])
                    string = f"keywords:{[f'({i}/{j})' for i, j in zip(keywords.split(), score.split('/'))]}\ntarget:{target}"
                    logger(f"eval/keyword/epoch{epoch}", string, (n_step - 1) * B + i)
            if print_output:
                for i in range(B):
                    keywords = decode_tensor(tokenizer, ids[i])
                    score = '/'.join(["%.2f" % j for j in scores[i].detach().cpu().numpy()])
                    target = decode_tensor(tokenizer, targets[i])
                    string = f"keywords:{[f'({i}/{j})' for i, j in zip(keywords.split(), score.split('/'))]}\ntarget:{target}"
                    logger(f"eval/keyword/epoch{epoch}", string, (n_step - 1) * B + i)

    model.train()

    return epoch_stats, keywords, target

import torch
from tqdm import tqdm
import json

from collections import defaultdict

from exp import ex
from path import get_dirname_from_args
from tensor_utils import move_device
from evaluate import _evaluate
from ckpt import save_ckpt
from extract_keyword import extract_and_save_all
from sampler import get_sampler
from loss.base import RLLoss


@ex.capture
def _train(model, loss_fn, optimizers, tokenizer, dataloaders, logger,
           data_path, max_epoch, device, reg_coeff, grad_acc_steps,
           eval_every, eval_subset, ckpt_path,
           eval_after, eval_test, eval_every_iter,
           extract_keyword, use_vist,
           reinforce_epoch, reinforce_metrics, reinforce_group,
           store_ckpt):
    best_stats = {}
    if model.task == 1:
        ckpt_criterion = {'key': 'CIDEr', 'compare': lambda prev, new: prev <= new}
        if use_vist:
            ckpt_criterion = {'key': 'METEOR', 'compare': lambda prev, new: prev <= new}
    elif model.task == 2:
        ckpt_criterion = {'key': 'final_loss', 'compare': lambda prev, new: prev >= new}

    def eval_data(key, epoch, eval_generate):
        eval_stats, _, texts = _evaluate(model, loss_fn, None, tokenizer,
                                    dataloaders['target'], logger, epoch=epoch,
                                    subset=None if (epoch + 1) % eval_every == 0 else eval_subset,
                                    key=key, eval_generate=eval_generate)

        if texts is not None and len(texts) > 0:
            if key == 'test':
                filename = 'blindtest' if 'blind' in data_path['test'].name else 'publictest'
            else:
                filename = key
            task1_out_path = ckpt_path / get_dirname_from_args()
            task1_out_path.mkdir(parents=True, exist_ok=True)
            ckpt_key = ckpt_criterion['key']
            task1_out_path = task1_out_path / f'epoch_{epoch}_{ckpt_key}_{eval_stats[ckpt_key]}_{filename}_results.json'
            # save file
            print(f"saving result to {task1_out_path}")
            with open(task1_out_path, 'w') as f:
                json.dump(texts, f, indent=4)

        for name, val in eval_stats.items():
            logger(f"{key}/epoch/{name}", val, epoch)

        return eval_stats

    def train_data(name, max_epoch_per_name, optimizer, dataset_idx=None):
        nonlocal best_stats
        dataset_type = 'pretrain' if name in 'pretrain' else 'target'
        if dataset_idx is not None:
            name = "{}_{}".format(name, dataset_idx)
            dataloader = dataloaders[dataset_type]['train'][i]
        else:
            dataloader = dataloaders[dataset_type]['train']

        n_step = 0
        print(f"{name}: {dataloader.task} training steps: {len(dataloader)}")
        print(f"max_epoch: {max_epoch_per_name}")
        print(f"from {dataloader.path}")
        for epoch in range(max_epoch_per_name):
            epoch += model.ckpt_epoch
            print(f"training {epoch}th epoch")
            epoch_stats = defaultdict(float)
            epoch_stats['num'] = 0
            model.train()
            if hasattr(model, 'epoch_update') and name not in 'pretrain':
                model.epoch_update(epoch)

            optimizer.zero_grad()
            for batch in tqdm(dataloader, total=len(dataloader)):
                batch = move_device(batch, to=device)
                B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
                targets = batch['targets']
                _, logits, targets, reg_loss, added_stats, sampler_input = model(batch, epoch=epoch)
                                                                        # batch_per_epoch=batch_per_epoch['train'],
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
                    final_loss = final_loss / grad_acc_steps
                    final_loss.backward()
                elif reg_loss is not None:
                    final_loss = reg_loss * reg_coeff
                    stats = {
                        'reg_loss': reg_loss.mean().item(),
                        'final_loss': final_loss.item()
                    }
                    final_loss = final_loss / grad_acc_steps
                    final_loss.backward()
                else:
                    continue

                if (n_step + 1) % grad_acc_steps == 0:
                    optimizer.clip_grad()
                    optimizer.step()
                    optimizer.scheduler.step()
                    optimizer.zero_grad()

                if added_stats is not None:
                    stats = {**stats, **added_stats}

                n_step += 1

                for k, v in stats.items():
                    if v is not None:
                        epoch_stats[k] = epoch_stats[k] + B * v
                epoch_stats['num'] = epoch_stats['num'] + B

                # log lr
                stats['lr'] = optimizer.get_lr()

                for key, val in stats.items():
                    logger(f"{name}/train/iters/{key}", val, n_step)

                if n_step % eval_every_iter == 0:
                    eval_stats = eval_data('val', epoch)
                    for key, val in eval_stats.items():
                        logger(f"{name}/eval/iters/{key}", val, n_step)

                del targets

            num = epoch_stats.pop('num')
            epoch_stats = {k: v / num for k, v in epoch_stats.items()}
            for key, val in epoch_stats.items():
                logger(f"{name}/train/epoch/{key}", val, epoch)

            eval_generate = False
            if epoch >= eval_after and epoch % eval_every == 0:
                eval_generate = True
            if name in 'pretrain':
                eval_generate = False

            eval_name = 'test' if eval_test else 'val'
            eval_stats = eval_data(eval_name, epoch, eval_generate=eval_generate)

            ckpt_key = ckpt_criterion['key']
            if ckpt_key in eval_stats:
                key_dt = {ckpt_key: eval_stats[ckpt_key]}

                # store all ckpt
                if store_ckpt and name not in 'pretrain':
                    save_ckpt(epoch, key_dt, model, tokenizer)
                    if not best_stats:
                        best_stats = eval_stats
                    else:
                        if ckpt_criterion['compare'](best_stats[ckpt_key], eval_stats[ckpt_key]):
                            best_stats = eval_stats

    pretrain_epoch = 0
    if 'pretrain' in optimizers:
        optimizer = optimizers['pretrain']
        if isinstance(optimizer, list):
            pretrain_epoch = sum(op.max_epoch for op in optimizer)
        else:
            pretrain_epoch = optimizer.max_epoch

    if pretrain_epoch > 0:
        model.pretrain()
        if isinstance(dataloaders['pretrain']['train'], list):
            for i in range(len(dataloaders['pretrain']['train'])):
                optimizer = optimizers['pretrain'][i]
                train_data('pretrain', optimizer.max_epoch, optimizer,
                        dataset_idx=i)
        else:
            optimizer = optimizers['pretrain']
            train_data('pretrain', optimizer.max_epoch, optimizer)
        model.pretrain(False)
    optimizer = optimizers['target']
    train_data('target', optimizer.max_epoch, optimizer)

    return best_stats

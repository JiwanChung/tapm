import os

import torch
from torch import nn
import torch.nn.functional as F

from model import get_model
from utils import get_dirname_from_args


def get_ckpt_path(args, epoch, loss):
    ckpt_name = get_dirname_from_args(args)
    os.makedirs(args.ckpt_path, exist_ok=True)
    ckpt_path = args.ckpt_path / \
        f'loss_{loss}_epoch_{epoch}_{ckpt_name}.pickle'

    return ckpt_path


def save_ckpt(args, epoch, loss, model, tokenizer):
    print(f'saving epoch {epoch}')
    dt = {
        'args': args,
        'epoch': epoch,
        'loss': loss,
        'model': model.state_dict(),
        'tokenizer': tokenizer
    }

    ckpt_path = get_ckpt_path(args, epoch, loss)
    print(f"Saving checkpoint {ckpt_path}")
    torch.save(dt, ckpt_path)


def get_model_ckpt(args):
    ckpt_available = args.ckpt_name is not None
    if ckpt_available:
        ckpt_paths = sorted(args.ckpt_path.glob(f'{args.ckpt_name}*'))
        assert len(ckpt_paths) > 0, f"no ckpt candidate for {ckpt_path}"
        ckpt_path = ckpt_paths[-1]  # monkey patch for choosing the best ckpt
        print(f"loading from {ckpt_path}")
        dt = torch.load(ckpt_path)
        args.update(dt['args'])

    model, tokenizer = get_model(args)

    if ckpt_available:
        model.load_state_dict(dt['model'])
        tokenizer = dt['tokenizer']
    return args, model, tokenizer, ckpt_available

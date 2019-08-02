import random
from pathlib import Path

from fire import Fire
from munch import Munch

import torch
import numpy as np

from config import config, debug_options, log_keys
from utils import wait_for_key
from train import train
from model.model import Model
from loss import Loss
from optimizer import get_optimizer
from transformers import get_transformer
from dataloader import get_dataloaders
from logger import Logger


class Cli:
    def __init__(self):
        self.defaults = config
        self.debug = debug_options

    def _default_args(self, **kwargs):
        args = self.defaults
        args['log_keys'] = log_keys
        if 'debug' in kwargs:
            args.update(self.debug)
        args.update(kwargs)
        args['log_keys'] = list(set([*args['log_keys'], *kwargs.keys()]))
        args = Munch(args)
        args.update(resolve_paths(config))
        args.update(fix_seed(args))
        args.update(get_device(args))

        return args

    def load_dataloader(self, **kwargs):
        args = self._default_args(**kwargs)
        data_paths = ['train', 'val', 'test']
        data_paths = {k: args[f"{k}_path"] for k in data_paths
                      if f"{k}_path" in args}

        models, tokenizer = get_transformer(args.transformer_name)
        dataloaders = get_dataloaders(args, data_paths, tokenizer)

        return args, models, tokenizer, dataloaders

    def train(self, **kwargs):
        args, models, tokenizer, dataloaders = \
            self.load_dataloader(**kwargs)
        model = Model(args, models, tokenizer)
        model.to(args.device)
        loss_fn = Loss(padding_idx=tokenizer.pad_id)
        logger = Logger(args)
        optimizer = get_optimizer(args, model, dataloaders)

        train(args, model, loss_fn, optimizer, tokenizer, dataloaders, logger)

        # hold process to keep tensorboard alive
        wait_for_key()

        '''
    def evaluate(self, **kwargs):
        args = self._default_args(**kwargs)

        evaluate(args)
        '''


def resolve_paths(config):
    paths = [k for k in config.keys() if k.endswith('_path')]
    res = {}
    for path in paths:
        res[path] = Path(config[path])
        if config['sample']:
            p = res[path].parts
            idx = [i for i, v in enumerate(p) if v == 'data']
            if len(idx) > 0:
                idx = idx[0]
                parts = [*p[:idx+1], 'sample', *p[idx+1:]]
                p = Path('/'.join(parts))
                res[path] = p

    return res


def fix_seed(args):
    if 'random_seed' not in args:
        args['random_seed'] = 0
    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])
    return args


def get_device(args):
    if hasattr(args, 'device'):
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {'device': device}


if __name__ == "__main__":
    Fire(Cli)

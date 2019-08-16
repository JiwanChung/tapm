import math
import random
from pathlib import Path

from fire import Fire
from munch import Munch

import torch
import numpy as np

from config import config, debug_options, log_keys
from utils import wait_for_key, add_keyword_paths
from train import train
from evaluate import evaluate
from extract_keyword import extract_and_save_all
from model import get_model
from ckpt import get_model_ckpt
from loss import Loss
from optimizer import get_optimizer
from transformers import get_transformer
from data.dataloader import get_dataloaders
from logger import Logger


class Cli:
    def __init__(self):
        self.defaults = config
        self.debug = debug_options

    def _default_args(self, **kwargs):
        torch.multiprocessing.set_sharing_strategy('file_system')  # https://github.com/pytorch/pytorch/issues/973
        args = self.defaults
        args['log_keys'] = log_keys
        if 'debug' in kwargs:
            args.update(self.debug)
        args.update(kwargs)
        args['log_keys'] = list(set([*args['log_keys'], *kwargs.keys()]))
        args = Munch(args)
        if hasattr(args, 'train_batch_size'):
            args.batch_sizes['train'] = args.train_batch_size
        if hasattr(args, 'val_batch_size'):
            args.batch_sizes['val'] = args.val_batch_size
        if hasattr(args, 'test_batch_size'):
            args.batch_sizes['test'] = args.test_batch_size

        args.update(resolve_paths(config))
        args.update(fix_seed(args))
        args.update(get_device(args))

        return args

    def prepare(self, **kwargs):
        args = self._default_args(**kwargs)
        args, model, tokenizer, ckpt = get_model_ckpt(args)
        model.to(args.device)
        args.update(kwargs)
        if 'keyword_dir' in args and args.keyword_dir is not None:
            args.data_path = add_keyword_paths(args.data_path, args.keyword_dir)
        dataloaders = get_dataloaders(args, args.data_path, model.make_batch, tokenizer)
        args.batch_per_epoch = {}
        for key in dataloaders.keys():
            args.batch_per_epoch[key] = \
                math.ceil(len(dataloaders[key]) / args.batch_sizes[key])
        loss_fn = Loss(padding_idx=tokenizer.pad_id)
        logger = Logger(args)
        optimizer = get_optimizer(args, model, dataloaders)

        return args, model, loss_fn, optimizer, tokenizer, dataloaders, logger

    def train(self, **kwargs):
        all_args = self.prepare(**kwargs)

        train(*all_args)

        # hold process to keep tensorboard alive
        wait_for_key()

    def evaluate(self, **kwargs):
        all_args = self.prepare(**kwargs)

        stats, keywords, target = evaluate(*all_args, print_output=False)

        print(stats)
        print(f"key:{keywords}, target:{target}")

        # hold process to keep tensorboard alive
        wait_for_key()

    def extract(self, **kwargs):
        args, model, _, _, tokenizer, \
            dataloaders, _ = self.prepare(**kwargs)
        for dataloader in dataloaders.values():
            dataloader.training = False

        extract_and_save_all(args, model, tokenizer, dataloaders)


def resolve_paths(config):
    paths = [k for k in config.keys() if k.endswith('_path')]
    res = {}
    res['root'] = Path('../').resolve()
    for path in paths:
        if config[path] is not None:
            res[path] = Path(config[path])
            # resolve root
            res[path] = res['root'] / res[path]
        '''
        if config['sample']:
            p = res[path].parts
            idx = [i for i, v in enumerate(p) if v == 'data']
            if len(idx) > 0:
                idx = idx[0]
                parts = [*p[:idx+1], 'sample', *p[idx+1:]]
                p = Path('/'.join(parts))
                res[path] = p
                '''
    res['data_path'] = {}
    for key in ['train', 'val', 'test']:
        if f"{key}_path" in res:
            res['data_path'][key] = res[f"{key}_path"]
            del res[f"{key}_path"]  # use data_path

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

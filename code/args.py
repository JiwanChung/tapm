import json
import random
from pathlib import Path, PosixPath

from munch import Munch
import numpy as np
import torch
from sacred.arg_parser import get_config_updates

from config import default_args, debug_args, reinforce_args, vist_args
from model import get_model_class

import tensorflow as tf


def get_args(options, fixed_args={}):
    post_args = {}
    new_args = get_new_args(options)
    post_args.update(new_args)
    post_args.update(fixed_args)
    args, Model = _get_args(post_args)
    args = process_args(args)
    args = Munch(update_data_path(args, Model))
    args = args.toDict()
    torch.multiprocessing.set_sharing_strategy('file_system')
    return args


def get_new_args(options):
    new_args, _ = get_config_updates(options['UPDATE'])
    return new_args


def process_args(args):
    tf.get_logger().setLevel('INFO')
    args = data_specific_args(args)

    for name in ['train', 'val', 'test']:
        if hasattr(args, f'{name}_batch_size') and args[f'{name}_batch_size'] is not None:
            args.batch_sizes[name] = args[f'{name}_batch_size']
    if hasattr(args, 'all_batch_size') and args.all_batch_size is not None:
        args.batch_sizes['train'] = args.all_batch_size
        args.batch_sizes['val'] = args.all_batch_size
        args.batch_sizes['test'] = args.all_batch_size

    args.update(fix_seed(args))
    args.update(get_device(args))
    args.update(resolve_paths(args))

    return args


def fix_seed(args):
    if 'random_seed' not in args or not isinstance(args['random_seed'], int):
        if 'seed' in args:
            args['random_seed'] = args['seed']
        else:
            args['random_seed'] = 0
    args['seed'] = args['random_seed']  # for sacred
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


def resolve_paths(config):
    path_dicts = ['data_path', 'vist_path', 'actynetcap_path',
                  'fib_path', 'multichoice_path']
    paths = [k for k in config.keys() if (k.endswith('_path') and k not in path_dicts)]
    res = {}
    res['root'] = Path('../').resolve()
    for key in ['val', 'test']:
        if config[f"{key}_path"] is None:
            config[f"{key}_path"] = config['train_path'].replace('training', key)
    for path in paths:
        if config[path] is not None:
            if isinstance(config[path], list):
                res[path] = []
                for i in range(len(config[path])):
                    res[path][i] = Path(config[path][i])
                    res[path][i] = res['root'] / res[path][i]
            if isinstance(config[path], dict):
                res[path] = {}
                for k in config[path].keys():
                    res[path][k] = Path(config[path][k])
                    res[path][k] = res['root'] / res[path][k]
            else:
                res[path] = Path(config[path])
                # resolve root
                res[path] = res['root'] / res[path]
    return res


def data_specific_args(args):
    if args.use_vist:
        # args.feature_names = ['images']
        args['eval_set'] = True
        if 'video' in args['feature_names']:
            args['feature_names'].remove('video')
        for name in ['train', 'val', 'test']:
            args[f"{name}_path"] = args['vist_path'][name]

    if args.use_fib:
        for name in ['train', 'val', 'test']:
            if name in args['fib_path']:
                args[f"{name}_path"] = args['fib_path'][name]

    if args.use_multichoice:
        for name in ['train', 'val', 'test']:
            if name in args['multichoice_path']:
                args[f"{name}_path"] = args['multichoice_path'][name]

    if args.use_actynetcap:
        args.feature_names = ['c3d']
        args.max_segments = 10
        args.grac_acc_steps = args.grad_acc_steps * 4
        all_batch_size = args.all_batch_size if args.all_batch_size is not None \
            else args.batch_sizes['train']
        args.all_batch_size = all_batch_size // 4
        for name in ['train', 'val', 'test']:
            args[f"{name}_path"] = args['actynetcap_path'][name]

    return args


def update_data_path(args, Model):
    if 'data_path' not in args:
        args['data_path'] = {}

    if 'pretrain_path' not in args:
        args['pretrain_path'] = {}

    for key in ['train', 'val', 'test']:
        if f"{key}_path" in args:
            args['data_path'][key] = args[f"{key}_path"]
            del args[f"{key}_path"]  # use data_path
        if f"{key}_pretrain_path" in args and args[f"{key}_pretrain_path"] is not None:
            args['pretrain_path'][key] = args[f"{key}_pretrain_path"]
            # del args[f"{key}_pretrain_path"]  # use pretrain_path

    for key, path in args.data_path.items():
        path = Path(path).resolve() \
                if str(path).startswith('/') else args.root / path
        args.data_path[key] = path

    for key, path in args.pretrain_path.items():
        if isinstance(path, list):
            li = []
            for p in path:
                p = Path(p).resolve() \
                    if str(p).startswith('/') else args.root / p
                li.append(p)
            args.pretrain_path[key] = li
        else:
            path = Path(path).resolve() \
                    if str(path).startswith('/') else args.root / path
            args.pretrain_path[key] = path

    if hasattr(Model, 'task'):
        def change_task(x):
            x = str(x)
            idx = x.find('task')
            if idx >= 0:
                x = list(x)
                x[idx + len('task')] = str(Model.task)
                x = ''.join(x)
            x = Path(x)
            return x

        args.task_name = f'task{Model.task}'
        args.data_path = {k: change_task(v) for k, v in args.data_path.items()}

    if args.use_lsmdc16:

        def change_data(x):
            task_name = x.parent.name
            if task_name == 'task1':
                x = x.parent.parent / 'task1_2016' / x.name
            return x

        args.data_path = {k: change_data(v) for k, v in args.data_path.items()}

    if 'task2' in str(args.data_path[args.use_data[0]]):
        args.data_path['train'] = Path(str(args.data_path['train']).replace('training_val', 'training'))
        args.data_path['val'] = Path(str(args.data_path['val']).replace('test', 'val'))

    if args['add_target_to_pretrain']:
        if 'train' in args['pretrain_path']:
            if not isinstance(args['pretrain_path']['train'], list):
                args['pretrain_path']['train'] = [args['pretrain_path']['train']]
            args['pretrain_path']['train'] = [*args['pretrain_path']['train'],
                                              args['data_path']['train']]

    return args


def merge_args(args1, args2, log_new=False):
    args1.update(args2)
    if log_new:
        args1['log_keys'] = list(set([*args1['log_keys'], *args2.keys()]))
    return Munch(args1)


def load_args(args):
    root = Path('../').resolve()
    if str(root) not in str(args.ckpt_path):
        args.ckpt_path = root / args.ckpt_path
    args_path = sorted(args.ckpt_path.glob(f'{args.ckpt_name}*'), reverse=False)[0].parent
    args_path = args_path / 'args.json'
    with open(args_path, 'r') as f:
        ckpt_args = json.load(f)
    eval_keys = [k for k, v in default_args.items() if not isinstance(v, str) and k != 'task_name']
    eval_keys.append('data_path')
    ckpt_args = {k: eval(v) if k in eval_keys else v for k, v in ckpt_args.items()}
    excludes = ['debug', 'sample', 'batch_sizes', 'all_batch_size', 'train_batch_size', 'val_batch_size', 'test_batch_size']
    ckpt_args = {k: v for k, v in ckpt_args.items() if k not in excludes and not k.endswith('_path')}

    return ckpt_args


def get_ckpt_args(args):
    ckpt_available = args.ckpt_name is not None
    return load_args(args) if ckpt_available else {}


def _get_args(new_args):
    args = Munch(default_args)
    command_args = Munch(new_args)
    args = merge_args(args, command_args, log_new=True)
    ckpt_args = Munch(get_ckpt_args(args))
    args = merge_args(args, ckpt_args, log_new=False)
    args = merge_args(args, command_args, log_new=False)
    model_args, Model = get_model_class(args['model_name'])
    model_args = Munch(model_args)
    args = merge_args(args, model_args, log_new=False)
    args = merge_args(args, command_args, log_new=False)

    if args['reinforce']:
        _reinforce_args = Munch(reinforce_args)
        args = merge_args(args, _reinforce_args, log_new=False)
    if args['use_vist']:
        _vist_args = Munch(vist_args)
        args = merge_args(args, _vist_args, log_new=False)
    if args['debug']:
        _debug_args = Munch(debug_args)
        args = merge_args(args, _debug_args, log_new=False)

    return args, Model

'''
1. command_args: dep[]
2. model_args: dep[default_args, command_args, ckpt_args]
3. ckpt_args: dep[default_args, command_args]
4. default_args: dep[]
---
(DeA -> CoA) -> CkA -> CoA -> MoA -> CoA
sacred
'''

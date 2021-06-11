import os
import json

from exp import ex

from args import get_args
from train import _train
from utils import wait_for_key, count_parameters
from evaluate import _evaluate
from infer import _infer
from vis_tsne import _tsne, _silhouette
from distance import _distance
from extract_keyword import extract_and_save_all
from model import get_model_options
from ckpt import get_model_ckpt
from loss.loss import get_loss
from optimizer import get_optimizer
from data.dataloader import get_dataloaders
from logger import get_logger
from scripts import run_script


@ex.capture
def prepare_model(model_name):
    return get_model_ckpt(model_name)


def prepare(no_logger=False):
    logger = get_logger(log_file=no_logger)

    model, tokenizer, ckpt, datasets, epoch = prepare_model()
    dataloaders = get_dataloaders(datasets, model.make_batch, tokenizer)
    '''
    args.batch_per_epoch = {}
    for key in dataloaders.keys():
        args.batch_per_epoch[key] = \
            math.ceil(len(dataloaders[key]) / args.batch_sizes[key])
    '''
    loss_fn = get_loss(padding_idx=tokenizer.pad_id)
    optimizers = get_optimizer(model, dataloaders)
    model.ckpt_epoch = epoch

    return model, loss_fn, optimizers, tokenizer, dataloaders, logger


@ex.command
def train():
    all_args = prepare()
    res = _train(*all_args)

    logger = all_args[-1]
    # hold process to keep tensorboard alive
    if 'tfboard' in logger.logger_dests:
        wait_for_key()

    return res


@ex.command
def evaluate(log_path):
    all_args = prepare(no_logger=True)
    stats, _, texts = _evaluate(*all_args, key='val', print_output=False)
    print(stats)
    model = all_args[0]
    assert hasattr(model, 'ckpt_path'), "no ckpt loaded"
    path = model.ckpt_path
    parent = path.parent.parent.parent
    dir_name = path.parent.stem
    parent = parent / "evals" / dir_name
    os.makedirs(parent, exist_ok=True)
    with open(parent / 'eval_stats.json', 'w') as f:
        json.dump(stats, f)
    with open(parent / 'eval_text.json', 'w') as f:
        json.dump(texts, f)


@ex.command
def tsne(log_path, test_path):
    # all_args = prepare({'use_data': 'val', 'sample': True})
    all_args = prepare()
    _tsne(*all_args, key='test')


@ex.command
def silhouette(log_path):
    # all_args = prepare({'use_data': 'val', 'sample': True})
    all_args = prepare()
    _silhouette(*all_args, key='test')


@ex.command
def distance(log_path):
    # all_args = prepare({'use_data': 'val', 'sample': True})
    all_args = prepare()
    _distance(*all_args, key='val')


@ex.command
def infer():
    #all_args = prepare({'use_data': 'val'})
    all_args = prepare()

    texts = _infer(*all_args)


@ex.command
def model_stats():
    #all_args = prepare({'use_data': 'val'})
    all_args = prepare(no_logger=True)
    model = all_args[0]

    stats = {}
    stats['parameter_counts'] = count_parameters(model)

    print(stats)


@ex.command
def extract():
    model, _, _, tokenizer, \
        dataloaders, _ = prepare()
    for dataloader in dataloaders.values():
        dataloader.training = False

    extract_and_save_all(model, tokenizer, dataloaders)


@ex.command
def scripts(script):
    run_script(script)


@ex.command
def print_models():
    print(sorted(get_model_options()))


@ex.option_hook
def update_args(options):
    args = get_args(options)
    print(sorted(args.items()))
    ex.add_config(args)


@ex.automain
def run():
    train()

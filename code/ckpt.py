import json

import torch

from exp import ex
from model import get_model
from path import get_dirname_from_args
from data.dataloader import get_datasets


@ex.capture
def get_ckpt_path(epoch, key_dt, ckpt_path):
    ckpt_name = get_dirname_from_args()
    ckpt_path = ckpt_path / ckpt_name
    ckpt_path.mkdir(exist_ok=True, parents=True)
    key, val = list(key_dt.items())[0]
    val = '{:.4f}'.format(val)
    ckpt_name = f'{key}_{val}_epoch_{epoch}.pickle'
    args_name = f'args.json'

    return ckpt_path, args_name, ckpt_name


@ex.capture
def save_ckpt(epoch, key_dt, model, tokenizer, _config):
    print(f'saving epoch {epoch}')
    args = _config
    dt = {
        'epoch': epoch,
        **key_dt,
        'model': model.state_dict(),
        'tokenizer': tokenizer
    }

    ckpt_path, args_name, ckpt_name = get_ckpt_path(epoch, key_dt)
    print(f"Saving checkpoint {ckpt_path / ckpt_name}")
    torch.save(dt, ckpt_path / ckpt_name)
    args_path = ckpt_path / args_name
    if not args_path.is_file():
        with open(args_path, 'w') as f:
            json.dump({k: str(v) for k, v in args.items()}, f)


@ex.capture
def load_ckpt(ckpt_path, ckpt_name):
    ckpt_paths = sorted(ckpt_path.glob(f'{ckpt_name}*'), reverse=False)
    assert len(ckpt_paths) > 0, f"no ckpt candidate for {ckpt_path / ckpt_name}"
    ckpt_path = ckpt_paths[0]  # monkey patch for choosing the best ckpt
    print(f"loading from {ckpt_path}")
    dt = torch.load(ckpt_path)

    return dt, ckpt_path


@ex.capture
def get_model_ckpt(model_name, ckpt_name, data_path, pretrain_path,
                   use_data, device, feature_names,
                   transformer_name=None):
    ckpt_available = ckpt_name is not None
    if ckpt_available:
        dt, ckpt_path = load_ckpt()

    datasets = get_datasets(data_path, pretrain_path)
    model, tokenizer = get_model(model_name, datasets['target'][use_data[0]].data,
                                 transformer_name=transformer_name)
    model = model.to(device)

    epoch = 0
    if ckpt_available:
        if 'net.transformer.word_embedding.weight' in dt['model']:
            model.net.transformer.word_embedding = model.net.transformer.wte
        elif hasattr(model.net.transformer, 'word_embedding'):
            del model.net.transformer.word_embedding

        new_dt = {}
        remove_keys = []
        for name in feature_names:
            for key in dt['model'].keys():
                if key.startswith(f"{name}."):
                    new_key = key.split('.')
                    if len(new_key) > 2 and new_key[2] in ['linear_in', 'linear2', 'res_layers']:
                        new_key = '.'.join(new_key[:2] + ['encoder']  + new_key[2:])
                        new_dt[new_key] = dt['model'][key]
                        remove_keys.append(key)
        dt['model'] = {**dt['model'], **new_dt}
        for key in remove_keys:
            del dt['model'][key]

        model.load_state_dict(dt['model'])
        tokenizer = dt['tokenizer']
        epoch = dt['epoch']
        model.ckpt_path = ckpt_path
    return model, tokenizer, ckpt_available, datasets, epoch

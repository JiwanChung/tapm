config = {
    'reg_coeff': 0.001,
    'keyword_threshold': 1/3,
    'transformer_pool': 'mean',
    'threshold_gap': 3,
    'transformer_name': 'gpt2',
    'use_keyword': True,

    'learning_rate': 6.25e-5,
    'warmup_steps': 4000,
    'batch_sizes': {'train': 16, 'val': 16},
    'max_epoch': 10,

    'train_path': '../data/ActyNetCap/train.json',
    'val_path': '../data/ActyNetCap/val_1.json',
    'num_workers': 32,

    'sample': False,
    'log_cmd': False,
    'log_path': '../log',
    'log_text_every': 2000,
    'hostname': 'snu.vision',
}

debug_options = {
    'num_workers': 0,
    'use_cmd': True,
}

log_keys = [
    'keyword_threshold',
    'threshold_gap',
]

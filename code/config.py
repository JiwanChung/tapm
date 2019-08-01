config = {
    'reg_coeff': 0.01,
    'keyword_threshold': 1/3,
    'transformer_pool': 'mean',
    'threshold_gap': 3,
    'transformer_name': 'gpt2',

    'learning_rate': 0.0001,
    'batch_sizes': {'train': 4, 'val': 4},
    'max_epoch': 10,

    'data_paths': {'train': '../data/ActyNetCap/train.json',
                   'val': '../data/ActyNetCap/val_1.json'},
    'num_workers': 32,

    'log_cmd': True,
    'log_path': '../log',
}

debug_options = {
}

log_keys = [
    'keyword_threshold',
    'threshold_gap',
]

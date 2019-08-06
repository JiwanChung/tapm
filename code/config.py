config = {
    'reg_coeff': 1,
    'keyword_threshold': 1/3,
    'transformer_pool': 'mean',
    'threshold_gap': 3,
    'model': 'mask_model',
    'use_keyword': True,
    'keyword_ratio': 0.3,
    'latent_std': 1,

    'learning_rate': 1e-5,
    'warmup_steps': 4000,
    'batch_sizes': {'train': 16, 'val': 16, 'test': 16},
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
    'model',
    'reg_coeff',
    'keyword_threshold',
    'threshold_gap',
    'learning_rate',
]

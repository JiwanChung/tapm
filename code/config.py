config = {
    'reg_coeff': 1,
    'keyword_threshold': 1/3,
    'transformer_pool': 'mean',
    'threshold_gap': 3,
    'model': 'mask_model',
    'use_keyword': True,
    'keyword_ratio': 0.4,
    'latent_std': 1,
    'extract_keyword': False,
    'extraction_threshold': 1,
    'extraction_min_words': 2,
    'eval_generate': False,

    'max_sentence_tokens': 50,
    'max_target_len': 30,  # max bpe token num in target: 69
    'sampling_method': 'greedy',
    'sampling_k': 8,
    'sampling_p': 0.9,

    'learning_rate': 1e-5,
    'warmup_steps': 4000,
    'batch_sizes': {'train': 16, 'val': 16, 'test': 16},
    'max_epoch': 30,

    'train_path': 'data/LSMDC/task1/LSMDC16_annos_training_someone.csv',
    'val_path': 'data/LSMDC/task1/LSMDC16_annos_val_someone.csv',
    'test_path': None,
    #'keyword_dir': 'keywords_top1000',
    'keyword_dir': None,
    'num_workers': 32,

    'sample': False,
    'log_cmd': False,
    'log_path': 'data/log',
    'log_text_every': 2000,
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'hostname': 'snu.vision',
}

debug_options = {
    'num_workers': 0,
    'use_cmd': True,
}

log_keys = [
    'sample',
    'model',
    'learning_rate',
    'keyword_ratio',
]

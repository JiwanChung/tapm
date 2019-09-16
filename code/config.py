config = {
    'reg_coeff': 1,
    'keyword_threshold': 1/3,
    'transformer_pool': 'mean',
    'threshold_gap': 3,
    'model': 'transformer_dis_small_vocab', # hybrid_dis
    'use_keyword': True,
    'keyword_ratio': 0.4,
    'latent_std': 1,
    'extract_keyword': False,
    'extraction_threshold': 1,
    'extraction_min_words': 2,
    'feature_names': ['video', 'image', 'flow'],
    'feature_name_map': {'i3d_rgb': 'video', 'resnet152_2': 'image', 'i3d_flow': 'flow'},
    'max_vocab': None,
    'pretrained_embedding': None,
    # 'pretrained_embedding': 'glove.840B.300d',

    'eval_generate': True,
    'eval_metric': True,
    'max_sentence_tokens': 50,
    'max_target_len': 30,  # max bpe token num in target: 69
    'sampling_method': 'greedy',
    'sampling_k': 8,
    'sampling_p': 0.9,
    'num_samples': 1,

    'learning_rate': 5e-5,
    'warmup_steps': 4000,
    'grad_clip': None,
    'grad_acc_steps': 1,
    'batch_sizes': {'train': 8, 'val': 8, 'test': 8},
    'max_epoch': 30,
    'eval_every': 1,
    'eval_subset': None,

    'train_path': 'data/LSMDC/task1/LSMDC16_annos_training_someone.csv',
    'val_path': 'data/LSMDC/task1/LSMDC16_annos_val_someone.csv',
    'test_path': None,
    #'keyword_dir': 'keywords_top1000',
    'keyword_name': 'keywords_gpt_top_1000.json',
    'num_workers': 32,

    'sample': False,
    'debug': False,
    'log_cmd': False,
    'log_multi': False, # Display multiple runs in a single board
    'log_path': 'data/log',
    'log_text_every': 2000,
    'log_tag': 'twostream', # Brief descriptive tag for logdir readibility
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'hostname': 'snu.vision',
}

debug_options = {
    'num_workers': 0,
    'use_cmd': True,
}

log_keys = [
    'log_tag',
    'sample',
    'model',
]

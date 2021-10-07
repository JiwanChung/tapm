default_args = {
    'logger_type': ['tfboard'],
    'store_ckpt': True,
    'random_seed': 0,
    'fix_gpt_epoch': 5,
    'reg_coeff': 1,
    'keyword_threshold': 1/3,
    'transformer_pool': 'mean',
    'threshold_gap': 3,
    'model_name': 'no_gt_sos', # hybrid_dis
    'task_name': None,
    'use_keyword': True,
    'use_gt_keyword': False,
    'use_word_subset': False,
    'concat_group': False,
    'ss_loss_type': 'ranking',  # ['ranking', 'bce', 'l2']
    'feature_names': ['video', 'images'],
    'feature_name_map': {
        'i3d_rgb': 'video',
        'resnet': 'images',
        'i3d_flow': 'flow',
        'rcnn': 'box',
        'scene': 'scene',
        'vggish': 'sound',
        'human_i3d': 'human_i3d',
        'c3d': 'c3d',
        'vilbert': 'vilbert',
        'vilbert_tune_5': 'vilbert_tune_5',
        'lxmert': 'lxmert',
        'rcnn_pad': 'object_feat'
    },
    'feature_dims': {
        'video': 1024,
        'images': 2048,
        'flow': 1024,
        'box': 1600,
        'scene': 365,
        'sound': 128,
        'human_i3d' : 1536,
        'c3d': 500,
        'vilbert': 1024,
        'vilbert_tune_5': 1024,
        'lxmert': 768,
        'object_feat': 2048,
    },
    'cross_modal_names': ['vilbert', 'lxmert', 'vilbert_tune_5'],
    'cut_feature_temporal_dim': {},  # e.g. rcnn_pad: 18
    'keyword_ratio': 0.4,
    'latent_std': 1,
    'extract_keyword': False,
    'extraction_threshold': 1,
    'extraction_min_words': 2,
    'use_data': ['train', 'val', 'test'],
    'metrics': ['meteor', 'bleu', 'rouge', 'cider'],
    'segment_pool_type': 'interval',  # ['interval', 'boundary', None]
    'max_vocab': None,
    'pretrained_embedding': None,
    # 'pretrained_embedding': 'glove.840B.300d',

    'force_ascii': True,
    'reinforce_epoch': 1000,
    'reinforce_metrics': ['meteor'],
    'reinforce_group': False,  # reinforce concatenated group
    'reinforce': False,
    'length_normalize_beam': False,
    'sample_ema_coeff': 1,
    'sample_eval_at_last': True,
    'nvlad_size': 64,
    'concentration': 100,
    'transformer_size': 'small',
    'transformer_name': None,  # 'gpt2',
    'new_beam_seach': False,
    'postprocess_duplicates': 1,  # 1 indicated no postprocessing

    'max_segments': 3,
    'eval_after': 5,
    'eval_every': 1,
    'eval_every_iter': 1e+10,
    'eval_generate': True,
    'eval_metric': True,
    'max_sentence_tokens': 50,
    'max_target_len': 30,  # max bpe token num in target: 69
    'max_target_sent': 5,  # max sent for ActivityNet Captions
    'sampling_method': 'greedy',  # ['beam', 'greedy', 'topk', 'max_nucleus']
    'sampling_k': 8,
    'sampling_p': 0.9,
    'num_samples': 1,
    'normalizer_sparsity': None,

    'learning_rate': 5e-5,
    'dropout': 0.5,
    'visual_dropout': 0.0,
    'change_transformer_dropout': False,
    'label_smoothing': 0,  # 0.1 for Attention is All You Need
    'warmup_steps': 4000,
    'grad_clip': None,
    'grad_acc_steps': 1,
    'batch_sizes': {'train': 8, 'val': 8, 'test': 8},
    'all_batch_size': None,
    'max_epoch': 30,
    'eval_subset': None,

    'eval_test': False,
    'eval_set': False,  # evaluate the group-concatenated text
    'use_lsmdc16': False,
    'use_vist': False,
    'use_fib': False,
    'use_multichoice': False,
    'repeat_vist_image': 1,  # previously 3, for unknown reasons
    'vist_sample_longest': False,
    'train_path': 'data/LSMDC/task1/LSMDC16_annos_training_val.csv',
    'val_path': 'data/LSMDC/task1/LSMDC16_annos_test.csv',
    'test_path': 'data/LSMDC/task1/LSMDC16_annos_test.csv',
    'vist_path': {
        'train': 'data/VIST/sis/train.story-in-sequence.json',
        'val': 'data/VIST/sis/test.story-in-sequence.json',
        'test': 'data/VIST/sis/test.story-in-sequence.json',
    },
    'fib_path': {
        'train': 'data/LSMDC/fib/LSMDC16_FIB_train.csv',
        'val': 'data/LSMDC/fib/LSMDC16_FIB_val.csv',
    },
    'multichoice_path': {
        'train': 'data/LSMDC/multichoice/LSMDC16_MC_train.csv',
        # 'val': 'data/LSMDC/multichoice/LSMDC16_MC_val.csv',
        'val': 'data/LSMDC/multichoice/LSMDC16_MC_test.csv',
        'test': 'data/LSMDC/multichoice/LSMDC16_MC_test.csv',
    },
    'add_target_to_pretrain': False,
    'train_pretrain_path': None,
    'val_pretrain_path': None,
    'test_pretrain_path': None,
    'use_actynetcap': False,
    'actynetcap_stride': 48,
    'actynetcap_path': {
        'train': 'data/ActyNetCap/actynetcap/train.json',
        'val': 'data/ActyNetCap/actynetcap/val_1.json',
        'test': 'data/ActyNetCap/actynetcap/val_2.json',
    },
    #'keyword_dir': 'keywords_top1000',
    'keyword_name': None,
    'num_workers': 64,

    'sample': False,
    'debug': False,
    'log_cmd': False,
    'log_multi': False, # Display multiple runs in a single board
    'log_path': 'data/log',
    'log_text_every': 2000,
    'log_tag': 'tag', # Brief descriptive tag for logdir readibility
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'hostname': 'localhost',
    'log_keys': [
        'log_tag',
        'sample',
        'model_name',
    ],

    'scripts': {
        'filename': 'publictest_results.json',
        'path': '../../data/LSMDC/task2/LSMDC16_annos_training_blank.csv',
        'relation_file': 'relation.pkl',
        'num_keywords': 1000,
        'logdir': '../../data/log/*',
    },
}

debug_args = {
    'num_workers': 1,
    'sample': True,
    'eval_after': 0,
    'eval_every': 1,
    'fix_gpt_epoch': 1,
    'logger_type': ['cmd'],
    'store_ckpt': False,
}

reinforce_args = {
    'reinforce_epoch': 0,
    'eval_after': 0,
    'eval_every': 1,
    'fix_gpt_epoch': 0,
}


vist_args = {
    'fix_gpt_epoch': 3,
}

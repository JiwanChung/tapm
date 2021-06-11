import logging
from functools import partial
from collections import OrderedDict
import json
from torch import nn
import torch

from transformers import (
    BertForMaskedLM, BertTokenizer,
    GPT2DoubleHeadsModel, GPT2Tokenizer,
    XLNetLMHeadModel, XLNetTokenizer,
    TransfoXLLMHeadModel, TransfoXLTokenizer,
    CTRLLMHeadModel, CTRLTokenizer,
    XLMWithLMHeadModel, XLMTokenizer,
    AutoConfig
)

from exp import ex
from utils import suppress_stdout
from data.tokenizer import build_tokenizer
from model.lstm import PretrainedLSTM,  load_vocab#, PretrainedQRNN


Models = {
    'bert': (BertForMaskedLM, BertTokenizer, 'bert-base-uncased'),
    'bert-cased': (BertForMaskedLM, BertTokenizer, 'bert-base-cased'),
    # 'xlnet': (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
    'gpt2': (GPT2DoubleHeadsModel, GPT2Tokenizer, 'gpt2'),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer, 'xlnet'),
    'xl': (TransfoXLLMHeadModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer, 'ctrl'),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer, 'xlm-mlm-tlm-xnli15-1024'),
}


def get_transformer(model_name):
    # tokenizing and numericalizing
    logging.getLogger('transformers').setLevel(logging.WARNING)
    res = {
        'bert': get_bert,
        'bert_cased': get_bert_cased,
        'gpt2': get_gpt2,
        'gpt2_not_trained': partial(get_gpt2, pretrained=False),
        'gpt2_medium': partial(get_gpt2, transformer_size='medium'),
        'gpt2_large': partial(get_gpt2, transformer_size='large'),
        'gpt2_xl': partial(get_gpt2, transformer_size='xl'),
        'xlnet': get_xlnet,
        'xl': get_xl,
        'ctrl': get_ctrl,
        'lstm': partial(get_lstm, pretrained=True),
        'lstm_not_trained': partial(get_lstm, pretrained=False),
        '''
        'qrnn': partial(get_qrnn, pretrained=True),
        'qrnn_not_trained': partial(get_qrnn, pretrained=False),
        '''
        'xlm': get_xlm,
        'none': get_no_transformer,
    }[model_name.lower()]()
    return res


def get_no_transformer():
    return None, None


def get_lstm(pretrained=True):
    model_class = PretrainedLSTM
    prev_vocab = load_vocab('lstm')
    tokenizer = build_tokenizer(data=None, prev_vocab=prev_vocab)
    net = model_class(tokenizer, pretrained=pretrained)

    return net, tokenizer


'''
def get_qrnn(pretrained=True):
    model_class = PretrainedQRNN
    prev_vocab = load_vocab('lstm')
    tokenizer = build_tokenizer(data=None, prev_vocab=prev_vocab)
    net = model_class(tokenizer, pretrained=pretrained)

    return net, tokenizer
'''


def get_bert():
    model = 'bert'
    model_class, tokenizer, model_name = Models[model]

    tokenizer = tokenizer.from_pretrained(model_name)
    for token in ['mask', 'sep', 'cls', 'pad', 'unk']:
        setattr(tokenizer, f'{token}_id',
                tokenizer.encode(getattr(tokenizer, f'{token}_token'))[0])
    tokenizer.model_name = model

    net = model_class.from_pretrained(model_name)
    net.transformer = net.bert

    return net, tokenizer


def get_bert_cased():
    model = 'bert-cased'
    model_class, tokenizer, model_name = Models[model]

    tokenizer = tokenizer.from_pretrained(model_name)
    for token in ['mask', 'sep', 'cls', 'pad', 'unk']:
        setattr(tokenizer, f'{token}_id',
                tokenizer.convert_tokens_to_ids(getattr(tokenizer, f'{token}_token')))

    with suppress_stdout():
        tokenizer.add_special_tokens({'additional_special_tokens': ['SOMEONE']})
    tokenizer.model_name = model
    config = AutoConfig.from_pretrained(model_name)
    net = model_class.from_pretrained(model_name, config=config)
    # resize embeddings
    if tokenizer.vocab_size != len(tokenizer):
        net.resize_token_embeddings(len(tokenizer))
    net.config = config
    net.transformer = net.bert
    net.lm_head = net.cls

    return net, tokenizer


@ex.capture
def get_gpt2(dropout, change_transformer_dropout, use_vist, use_actynetcap,
             transformer_size='small', pretrained=True):
    model_name = 'gpt2'

    # trying to not add additional tokens to orig vocab
    default_special_tokens = {
        'unk_token': "___",
        #'sep_token': "<|endoftext|>",
        # 'cls_token': "[CLS]",
        'pad_token': "____",
        'sep_token': "_____",
        'seq_sep_token': "______",
        'context_sep_token': "_______",
        'eos_token': "<|endoftext|>",
    }
    special_tokens = {
        'cls_token': "<|endoftext|>",
        # 'type_a': '[TYPE_A]',
        # 'type_b': '[TYPE_B]',
        # 'person': '[SOMEONE]'
    }
    if not (use_vist or use_actynetcap):
        person_tokens = {f'person{i}': f'[PERSON{i}]' for i in range(20)}
        person_tokens = {'blank': '[...]', **person_tokens}
    else:
        person_tokens = {}

    model_class, tokenizer, model_name = Models[model_name]
    size_dict = {
        'small': '',
        'medium': '-medium',
        'large': '-large',
        'xl': '-xl',
    }
    model_name += size_dict[transformer_size]

    with suppress_stdout():
        tokenizer = tokenizer.from_pretrained(model_name, **default_special_tokens)
        # tokenizer.add_tokens(person_tokens.values())
        tokenizer.add_special_tokens({'additional_special_tokens': list(person_tokens.values())})
        # unk is needed to add other special tokens

        for k, v in tokenizer.special_tokens_map.items():
            if k.endswith('_token'):
                setattr(tokenizer, f"{k[:-6]}_id",
                    tokenizer.convert_tokens_to_ids(v))
            else:
                setattr(tokenizer, f"{k}_id",
                    tokenizer.convert_tokens_to_ids(v))
        for k, v in default_special_tokens.items():
            if k.endswith('_token'):
                setattr(tokenizer, f"{k[:-6]}_id",
                    tokenizer.convert_tokens_to_ids(v))
            else:
                setattr(tokenizer, f"{k}_id",
                    tokenizer.convert_tokens_to_ids(v))
    tokenizer.cls_id = tokenizer.encode(special_tokens['cls_token'])[0]
    tokenizer.whitespace = b'\xc4\xa0'.decode()
    tokenizer.model_name = model_name

    config = AutoConfig.from_pretrained(model_name)
    if change_transformer_dropout:
        config.resid_pdrop = dropout
        config.embd_pdrop = dropout
        config.attn_pdrop = dropout

    if pretrained:
        net = model_class.from_pretrained(model_name, config=config)
    else:
        net = model_class(config)
    # resize embeddings
    if tokenizer.vocab_size != len(tokenizer):
        net.resize_token_embeddings(len(tokenizer))
    net.config = config

    return net, tokenizer


@ex.capture
def get_xlnet(transformer_size='small'):
    model_class, tokenizer, model_name = Models['xlnet']
    size_dict = {
        'small': '-base-cased',
        'large': '-large-cased',
    }
    model_name += size_dict[transformer_size]

    default_special_tokens = {
        'unk_token': "<unk>",
        'cls_token': "<s>",
        'pad_token': "<pad>",
        'sep_token': "</s>",
        'seq_sep_token': "<sep>",
        'context_sep_token': "<cls>",
    }

    with suppress_stdout():
        tokenizer = tokenizer.from_pretrained(model_name, **default_special_tokens)

        for k, v in tokenizer.special_tokens_map.items():
            if k.endswith('_token'):
                setattr(tokenizer, f"{k[:-6]}_id",
                    tokenizer.convert_tokens_to_ids(v))
            else:
                setattr(tokenizer, f"{k}_id",
                    tokenizer.convert_tokens_to_ids(v))
        for k, v in default_special_tokens.items():
            if k.endswith('_token'):
                setattr(tokenizer, f"{k[:-6]}_id",
                    tokenizer.convert_tokens_to_ids(v))
            else:
                setattr(tokenizer, f"{k}_id",
                    tokenizer.convert_tokens_to_ids(v))

    tokenizer.model_name = model_name
    config = AutoConfig.from_pretrained(model_name)
    net = model_class.from_pretrained(model_name, config=config)
    net.lm_head = net.lm_loss
    # resize embeddings
    if tokenizer.vocab_size != len(tokenizer):
        net.resize_token_embeddings(len(tokenizer))

    return net, tokenizer


@ex.capture
def get_xl(dropout, change_transformer_dropout, data_path):
    model_name = 'xl'

    # trying to not add additional tokens to orig vocab
    default_special_tokens = {
        'unk_token': "<unk>",
        #'sep_token': "<|endoftext|>",
        # 'cls_token': "[CLS]",
        'eos_token': "<eos>",
    }
    special_tokens = {
        'cls_token': "<cls>",
        'context_sep_token': "<context_sep>",
        'seq_sep_token': "<seq_sep>",
        'sep_token': "<eos>",
        'pad_token': "<pad>",
        # 'type_a': '[TYPE_A]',
        # 'type_b': '[TYPE_B]',
        # 'person': '[SOMEONE]'
    }
    default_special_tokens = {**default_special_tokens,
                              'additional_special_tokens': list(special_tokens.values())}

    model_class, tokenizer, model_name = Models[model_name]

    with suppress_stdout():
        tokenizer = tokenizer.from_pretrained(model_name, **default_special_tokens)
        # unk is needed to add other special tokens
        path = data_path['train'].parent / 'transformer_xl_counter.json'
        if path.is_file():
            print('loading additional tokens')
            with open(path, 'r') as f:
                tokens = json.load(f)
            tokenizer.add_tokens(list(tokens.keys()))
        tokenizer.add_tokens(list(special_tokens.values()))

        for k, v in tokenizer.special_tokens_map.items():
            if k.endswith('_token'):
                setattr(tokenizer, f"{k[:-6]}_id",
                    tokenizer.convert_tokens_to_ids(v))
            else:
                setattr(tokenizer, f"{k}_id",
                    tokenizer.convert_tokens_to_ids(v))
        for k, v in special_tokens.items():
            if k.endswith('_token'):
                setattr(tokenizer, k, v)
                setattr(tokenizer, f"{k[:-6]}_id",
                    tokenizer.convert_tokens_to_ids(v))
            else:
                setattr(tokenizer, f"{k}_id",
                    tokenizer.convert_tokens_to_ids(v))
    tokenizer.cls_id = tokenizer.encode(special_tokens['cls_token'])[0]
    tokenizer.model_name = model_name

    prev_vocab_size = tokenizer.vocab_size
    tokenizer = resize_transformer_xl_vocab(tokenizer)

    config = AutoConfig.from_pretrained(model_name)
    if change_transformer_dropout:
        config.resid_pdrop = dropout
        config.embd_pdrop = dropout
        config.attn_pdrop = dropout
    net = model_class.from_pretrained(model_name, config=config)
    # resize embeddings
    # tokenizer.vocab_size = len(tokenizer)
    if prev_vocab_size != len(tokenizer):
        net.transformer.word_emb = resize_transformer_xl_embedding(
            net.transformer.word_emb,
            len(tokenizer)
        )
        # re initialize output
        if config.sample_softmax > 0:
            net.out_layer = nn.Linear(config.d_model, len(tokenizer))
            net.sampler = net.sampler.__class__(len(tokenizer), config.sample_softmax)
        else:
            net.crit = resize_transformer_xl_out(net.crit, len(tokenizer))
        # net.resize_token_embeddings(len(tokenizer))

    return net, tokenizer


def resize_transformer_xl_vocab(tokenizer):
    added_tokens = [t for i, t in sorted(tokenizer.added_tokens_decoder.items())]
    tokenizer.idx2sym = [*tokenizer.idx2sym, *added_tokens]
    tokenizer.sym2idx = OrderedDict([(v, i) for i, v in enumerate(tokenizer.idx2sym)])
    tokenizer.added_tokens_encoder = {}  # decoding error
    return tokenizer


def resize_transformer_xl_out(embedding, new_size):
    embedding.cutoffs = [*embedding.cutoffs, new_size]
    embedding.cutoff_ends = [*embedding.cutoff_ends, new_size]
    d_emb_i = 256
    l_idx = embedding.cutoff_ends[-2]
    r_idx = embedding.cutoff_ends[-1]
    embedding.out_layers.append(nn.Linear(d_emb_i, r_idx-l_idx))
    embedding.out_projs.append(nn.Parameter(torch.randn(embedding.d_proj, d_emb_i)))

    embedding.n_clusters = len(embedding.cutoffs) - 1
    embedding.head_size = embedding.shortlist_size + embedding.n_clusters

    prev_cluster_weight = embedding.cluster_weight.data
    embedding.cluster_weight = nn.Parameter(torch.zeros(embedding.n_clusters, embedding.d_embed))
    embedding.cluster_weight.data[:prev_cluster_weight.shape[0]] = prev_cluster_weight
    prev_cluster_bias = embedding.cluster_bias.data
    embedding.cluster_bias = nn.Parameter(torch.zeros(embedding.n_clusters))
    embedding.cluster_bias.data[:prev_cluster_bias.shape[0]] = prev_cluster_bias
    embedding.n_token = new_size
    return embedding


def resize_transformer_xl_embedding(embedding, new_size):
    embedding.cutoffs = [*embedding.cutoffs, new_size]
    embedding.cutoff_ends = [*embedding.cutoff_ends, new_size]
    d_emb_i = 256
    l_idx = embedding.cutoff_ends[-2]
    r_idx = embedding.cutoff_ends[-1]
    embedding.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
    embedding.emb_projs.append(nn.Parameter(torch.randn(embedding.d_proj, d_emb_i)))
    return embedding


@ex.capture
def get_ctrl(dropout, change_transformer_dropout, use_vist, use_actynetcap):
    model_name = 'ctrl'

    # trying to not add additional tokens to orig vocab
    default_special_tokens = {
    }
    special_tokens = {
        'cls_token': "--------",
        'unk_token': "<unk>",
        'pad_token': "----",
        'sep_token': "-----",
        'seq_sep_token': "------",
        'context_sep_token': "-------",
        'eos_token': "-----",
    }
    default_special_tokens = {**default_special_tokens,
                              'additional_special_tokens': list(special_tokens.values())}

    model_class, tokenizer, model_name = Models[model_name]

    with suppress_stdout():
        tokenizer = tokenizer.from_pretrained(model_name, **default_special_tokens)
        # unk is needed to add other special tokens

        for k, v in tokenizer.special_tokens_map.items():
            if k.endswith('_token'):
                setattr(tokenizer, f"{k[:-6]}_id",
                    tokenizer.convert_tokens_to_ids(v))
            else:
                setattr(tokenizer, f"{k}_id",
                    tokenizer.convert_tokens_to_ids(v))
        for k, v in special_tokens.items():
            if k.endswith('_token'):
                setattr(tokenizer, k, v)
                setattr(tokenizer, f"{k[:-6]}_id",
                    tokenizer.convert_tokens_to_ids(v))
            else:
                setattr(tokenizer, f"{k}_id",
                    tokenizer.convert_tokens_to_ids(v))
    tokenizer.cls_id = tokenizer.encode(special_tokens['cls_token'])[0]
    tokenizer.model_name = model_name

    config = AutoConfig.from_pretrained(model_name)
    if change_transformer_dropout:
        config.resid_pdrop = dropout
        config.embd_pdrop = dropout
        config.attn_pdrop = dropout
    net = model_class.from_pretrained(model_name, config=config)
    # resize embeddings
    if tokenizer.vocab_size != len(tokenizer):
        net.resize_token_embeddings(len(tokenizer))

    return net, tokenizer


@ex.capture
def get_xlm():
    model_name = 'xlm'

    # trying to not add additional tokens to orig vocab
    default_special_tokens = {
    }
    special_tokens = {
        'cls_token': "<s>",
        'unk_token': "<unk>",
        'pad_token': "<pad>",
        'sep_token': "</s>",
        'seq_sep_token': "<special0>",
        'context_sep_token': "<special1>",
        'eos_token': "</s>",
    }
    default_special_tokens = {**default_special_tokens,
                              'additional_special_tokens': list(special_tokens.values())}

    model_class, tokenizer, model_name = Models[model_name]

    with suppress_stdout():
        tokenizer = tokenizer.from_pretrained(model_name, **default_special_tokens)
        # unk is needed to add other special tokens

        for k, v in tokenizer.special_tokens_map.items():
            if k.endswith('_token'):
                setattr(tokenizer, f"{k[:-6]}_id",
                    tokenizer.convert_tokens_to_ids(v))
            else:
                setattr(tokenizer, f"{k}_id",
                    tokenizer.convert_tokens_to_ids(v))
        for k, v in special_tokens.items():
            if k.endswith('_token'):
                setattr(tokenizer, k, v)
                setattr(tokenizer, f"{k[:-6]}_id",
                    tokenizer.convert_tokens_to_ids(v))
            else:
                setattr(tokenizer, f"{k}_id",
                    tokenizer.convert_tokens_to_ids(v))
    tokenizer.cls_id = tokenizer.encode(special_tokens['cls_token'])[0]
    tokenizer.model_name = model_name

    config = AutoConfig.from_pretrained(model_name)
    config.causal = True
    net = model_class.from_pretrained(model_name, config=config)
    # resize embeddings
    if tokenizer.vocab_size != len(tokenizer):
        net.resize_token_embeddings(len(tokenizer))

    return net, tokenizer

import random

import torch
from pytorch_transformers import (
    BertForMaskedLM, BertTokenizer,
    GPT2DoubleHeadsModel, GPT2Tokenizer
)


# we support only GPT2 at the moment
Models = {
    'bert': (BertForMaskedLM, BertTokenizer, 'bert-base-uncased'),
    # 'xlnet': (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
    'gpt2': (GPT2DoubleHeadsModel, GPT2Tokenizer, 'gpt2'),
}


def get_transformer(model_name):
    # tokenizing and numericalizing
    return {
        'bert': get_bert,
        'gpt2': get_gpt2
    }[model_name.lower()]()


def get_bert():
    model = 'bert'
    model_class, tokenizer, model_name = Models[model]

    tokenizer = tokenizer.from_pretrained(model_name)
    for token in ['mask', 'sep', 'cls', 'pad', 'unk']:
        setattr(tokenizer, f'{token}_id',
                tokenizer.encode(getattr(tokenizer, f'{token}_token'))[0])
    tokenizer.model_name = model

    net = model_class.from_pretrained(model_name)

    return net, tokenizer


def get_gpt2():
    model_name = 'gpt2'
    # trying to not add additional tokens to orig vocab
    default_special_tokens = {
        'unk_token': "___",
        #'sep_token': "<|endoftext|>",
        # 'cls_token': "[CLS]",
        'pad_token': "_____",
        'eos_token': "<|endoftext|>",
    }
    special_tokens = {
        'sos_token': "<|endoftext|>",
        # 'type_a': '[TYPE_A]',
        # 'type_b': '[TYPE_B]',
        # 'person': '[SOMEONE]'
    }
    model_class, tokenizer, model_name = Models[model_name]

    tokenizer = tokenizer.from_pretrained(model_name, **default_special_tokens)
    # unk is needed to add other special tokens
    # tokenizer.add_special_tokens(special_tokens)
    for k, v in tokenizer.special_tokens_map.items():
        if k.endswith('_token'):
            setattr(tokenizer, f"{k[:-6]}_id",
                tokenizer.convert_tokens_to_ids(v))
        else:
            setattr(tokenizer, f"{k}_id",
                tokenizer.convert_tokens_to_ids(v))
    tokenizer.sos_id = tokenizer.encode(special_tokens['sos_token'])[0]
    tokenizer.model_name = model_name

    # get models
    encoder = model_class.from_pretrained(model_name)
    decoder = model_class.from_pretrained(model_name)
    # resize embeddings
    if tokenizer.vocab_size != len(tokenizer):
        encoder.resize_token_embeddings(len(tokenizer))
        decoder.resize_token_embeddings(len(tokenizer))
    # share embeddings
    decoder.transformer.wte.weight = encoder.transformer.wte.weight
    decoder.transformer.wpe.weight = encoder.transformer.wpe.weight

    return {'encoder': encoder, 'decoder': decoder}, tokenizer

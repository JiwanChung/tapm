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
        'gpt2': get_gpt2,
        'none': get_no_transformer,
    }[model_name.lower()]()


def get_no_transformer():
    return None, None


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
        'sep_token': "<|endoftext|>",
    }
    special_tokens = {
        'cls_token': "<|endoftext|>",
        # 'type_a': '[TYPE_A]',
        # 'type_b': '[TYPE_B]',
        # 'person': '[SOMEONE]'
    }
    person_tokens = {f'person{i}': f'[PERSON{i}]' for i in range(20)}
    person_tokens = {'blank': '[...]', **person_tokens}

    model_class, tokenizer, model_name = Models[model_name]

    tokenizer = tokenizer.from_pretrained(model_name, **default_special_tokens)
    # tokenizer.add_tokens(person_tokens.values())
    tokenizer.add_special_tokens(person_tokens)
    # unk is needed to add other special tokens

    for k, v in tokenizer.special_tokens_map.items():
        if k.endswith('_token'):
            setattr(tokenizer, f"{k[:-6]}_id",
                tokenizer.convert_tokens_to_ids(v))
        else:
            setattr(tokenizer, f"{k}_id",
                tokenizer.convert_tokens_to_ids(v))
    tokenizer.cls_id = tokenizer.encode(special_tokens['cls_token'])[0]
    tokenizer.model_name = model_name

    net = model_class.from_pretrained(model_name)
    # resize embeddings
    if tokenizer.vocab_size != len(tokenizer):
        net.resize_token_embeddings(len(tokenizer))

    return net, tokenizer

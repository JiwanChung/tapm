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


def make_batch(model_name, tokenizer, sentences, **kwargs):
    # tokenizing and numericalizing
    return {
        'bert': make_bert_batch,
        'gpt2': make_gpt2_batch
    }[model_name.lower()](tokenizer, sentences, **kwargs)


def make_bert_batch(tokenizer, sentences, random_idx=True):
    sentences = [torch.Tensor([tokenizer.cls_id,
                               *tokenizer.encode(t),
                               tokenizer.sep_id]) for t in sentences]
    targets, _ = pad(sentences, tokenizer.pad_id)
    # mask words
    sentences, mask_ids = mask_words(sentences, tokenizer.mask_id, random_idx)
    if random_idx:
        # tensor B*L
        sentences, lengths = pad(sentences, tokenizer.pad_id)
    else:
        # len B list of tensor L*L
        li = []
        lengths = []
        for sentence in sentences:
            sentence, length = pad(sentence, tokenizer.pad_id)
            li.append(sentence)
            lengths.append(length)
        sentences = li

    return sentences, lengths, mask_ids, targets


def mask_words(tensors, mask_idx, random_idx=True):
    # B(CLS+L+SEP)
    if random_idx:
        # mask random idx
        li = []
        ids = []
        for t in tensors:
            device = t.device
            idx = random.randint(1, t.shape[0] - 1)
            ids.append(idx)
            t[idx] = mask_idx
            li.append(t)
        return li, torch.Tensor(ids).long().to(device)
    else:
        # generate mask for every word
        li = []
        for t in tensors:
            t = t.unsqueeze(0).repeat(t.shape[0], 1)
            eye = torch.eye(t.shape[0]).byte().to(t.device)
            full = torch.full(t.shape, mask_idx, dtype=t.dtype).to(t.device)
            t.masked_scatter_(mask=eye, source=full)
            li.append(t)
        # list of L*L
        return li, None


def make_gpt2_batch(tokenizer, sentences, **kwargs):
    sentences = [tokenizer.encode(t) for t in sentences]

    targets = [torch.Tensor([*t, tokenizer.eos_id]) for t in sentences]
    sentences = [torch.Tensor([tokenizer.sos_id, *t]) for t in sentences]
    sentences, lengths = pad(sentences, tokenizer.pad_id)
    targets, _ = pad(targets, tokenizer.pad_id)

    return sentences, lengths, targets


def pad(x, pad_id=0):
    B = len(x)
    lengths = [t.shape[0] for t in x]
    max_len = max(lengths)
    res = torch.full((B, max_len), pad_id, dtype=torch.long).to(x[0].device)
    for i in range(B):
        res[i, :x[i].shape[0]] = x[i]
    lengths = torch.LongTensor(lengths).to(x[0].device)

    return res, lengths


def remove_pad(x, pad_id=0):
    return x[:(x != pad_id).sum(-1)]


def decode_tensor(tokenizer, x):
    if x.dim() < 1:
        x = x.unsqueeze(0)
    return tokenizer.decode(remove_pad(x, tokenizer.pad_id).cpu().numpy())

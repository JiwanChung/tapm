import torch
from pytorch_transformers import (
    BertModel, GPT2Model, XLNetModel,
    BertTokenizer, GPT2Tokenizer, XLNetTokenizer
)


# we support only GPT2 at the moment
Models = {
    #'bert': (BertModel, BertTokenizer, 'bert-base-uncased'),  we need a language model
    'gpt2': (GPT2Model, GPT2Tokenizer, 'gpt2'),
    'xlnet': (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
}


def get_transformer(model_name):
    # trying to not add additional tokens to orig vocab
    default_special_tokens = {
        'unk_token': "___",
        'sep_token': "<|endoftext|>",
        # 'cls_token': "[CLS]",
        'pad_token': "_____",
    }
    special_tokens = {
        # 'type_a': '[TYPE_A]',
        # 'type_b': '[TYPE_B]',
        # 'person': '[SOMEONE]'
    }
    model_class, tokenizer, model_name = Models[model_name]

    tokenizer = tokenizer.from_pretrained(model_name, **default_special_tokens)
    # unk is needed to add other special tokens
    tokenizer.add_special_tokens(special_tokens)
    for k, v in tokenizer.special_tokens_map.items():
        if k.endswith('_token'):
            setattr(tokenizer, f"{k[:-6]}_id",
                tokenizer.convert_tokens_to_ids(v))
        else:
            setattr(tokenizer, f"{k}_id",
                tokenizer.convert_tokens_to_ids(v))

    # get models
    encoder = model_class.from_pretrained(model_name)
    decoder = model_class.from_pretrained(model_name)
    # resize embeddings
    if tokenizer.vocab_size != len(tokenizer):
        encoder.resize_token_embeddings(len(tokenizer))
        decoder.resize_token_embeddings(len(tokenizer))
    # share embeddings
    decoder.wte.weight = encoder.wte.weight
    decoder.wpe.weight = encoder.wpe.weight

    return {'encoder': encoder, 'decoder': decoder}, tokenizer


def make_batch(tokenizer, sentences):
    # tokenizing and numericalizing
    sentences = [tokenizer.encode(t) for t in sentences]

    targets = [torch.Tensor([*t, tokenizer.sep_id]) for t in sentences]
    sentences = [torch.Tensor([tokenizer.sep_id, *t]) for t in sentences]
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
    return tokenizer.decode(remove_pad(x, tokenizer.pad_id).cpu().numpy())

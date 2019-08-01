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
    default_special_tokens = {
        'unk_token': "[UNK]",
        'sep_token': "[SEP]",
        'cls_token': "[CLS]",
        'pad_token': "[PAD]",
    }
    special_tokens = {
        'type_a': '[TYPE_A]',
        'type_b': '[TYPE_B]',
        'person': '[SOMEONE]'
    }
    model_class, tokenizer, model_name = Models[model_name]

    tokenizer = tokenizer.from_pretrained(model_name, **default_special_tokens)
    # unk is needed to add other special tokens
    tokenizer.add_special_tokens({**special_tokens,
                                  **default_special_tokens})
    tokenizer.encoder = {**tokenizer.encoder, **tokenizer.added_tokens_encoder}
    tokenizer.decoder = {**tokenizer.decoder, **tokenizer.added_tokens_decoder}
    for token in special_tokens.values():
        assert token in tokenizer.encoder, f"Failed to add token {token} to vocab!"
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
    encoder._resize_token_embeddings(tokenizer.vocab_size + 1)
    decoder._resize_token_embeddings(tokenizer.vocab_size + 1)
    # share embeddings
    decoder.wte.weight.data = encoder.wte.weight.data
    decoder.wpe.weight.data = encoder.wpe.weight.data

    return (encoder, decoder), tokenizer


def make_batch(tokenizer, sentences):
    # tokenizing and numericalizing
    sentences = [tokenizer.encode(t) for t in sentences]

    targets = [torch.Tensor([*t, tokenizer.sep_id]) for t in sentences]
    sentences = [torch.Tensor(t) for t in sentences]
    sentences, lengths = pad(sentences, tokenizer.pad_id)
    targets, _ = pad(targets, tokenizer.pad_id)
    # getting length

    return sentences, lengths, targets


def pad(x, pad_id=0):
    B = len(x)
    lengths = [t.shape[0] for t in x]
    max_len = max(lengths)
    res = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i in range(B):
        res[i, :x[i].shape[0]] = x[i]
    lengths = torch.LongTensor(lengths)

    return res, lengths


def remove_pad(x, pad_id=0):
    return x[:(x != pad_id).sum(-1)]

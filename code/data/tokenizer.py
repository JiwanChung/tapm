import json

import torch
import collections
from itertools import chain

from tqdm import tqdm

from torchtext.vocab import pretrained_aliases
from transformers import PreTrainedTokenizer, BasicTokenizer

from exp import ex


@ex.capture
def build_tokenizer(data, data_path, max_vocab, prev_vocab=None):
    json_path = data_path['train'].parent / f'whitespace_vocab.json'
    if json_path.is_file():
        print('loading vocab')
        with open(json_path, 'r') as f:
            prev_vocab = [json.load(f), prev_vocab]
    print('building vocab')
    tokenizer = WhitespaceTokenizer(data, max_vocab, prev_vocab)
    for token in ['mask', 'sep', 'cls', 'pad', 'unk']:
        setattr(tokenizer, f'{token}_id',
                tokenizer.encode(getattr(tokenizer, f'{token}_token'))[0])
    # torch.save(tokenizer, path)

    print(f"vocab num: {len(tokenizer)}")
    return tokenizer


class WhitespaceTokenizer(PreTrainedTokenizer):
    def __init__(self, data, max_vocab, prev_vocab=None):
        self.special_tokens = {
            'mask_token': '[MASK]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'seq_sep_token': '[SEQ_SEP]',
            'context_sep_token': '[CONTEXT_SEP]',
            'pad_token': '[PAD]',
            'unk_token': '[UNK]'
        }
        super(WhitespaceTokenizer, self).__init__(**self.special_tokens)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=True)
        self.max_vocab = max_vocab

        counter = None
        if data is not None:
            counter = self.count(data)
        if isinstance(prev_vocab, list):
            self.prev_vocab = prev_vocab[-1]
            total_prev_vocab = {}
            for v in prev_vocab:
                total_prev_vocab = self.merge_prev_vocabs(total_prev_vocab, v)
            self.vocab = self.counter_to_vocab(counter, total_prev_vocab)
        else:
            self.prev_vocab = prev_vocab
            self.vocab = self.counter_to_vocab(counter, prev_vocab)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

        for k, t in self.special_tokens.items():
            setattr(self, f"{k[:-6]}_id", self.vocab[t])

    def merge_prev_vocabs(self, v1, v2):
        token_diff = set(v1.keys()) - set(v2.keys())  # drop v1 indices
        v_diff = {k: i + len(v2) for i, k in enumerate(token_diff)}
        return {**v_diff, **v2}

    def count(self, data):
        peek = data.values().__iter__().__next__()
        if isinstance(peek, list):
            data = [[i['target'] for i in v] for v in data.values()]
            data = chain(*data)
        else:
            data = [v['target'] for v in data.values()]
        counter = collections.Counter()
        for d in tqdm(data):
            d = self.tokenize(d)
            for word in d:
                counter[word] += 1
        return counter

    def counter_to_vocab(self, counter, prev_vocab=None):
        specials = list(self.special_tokens.values())
        if counter is None:
            vocab = specials
        else:
            if self.max_vocab is None:
                vocab = list(counter.items())
            else:
                vocab = counter.most_common(self.max_vocab - len(specials))
            vocab_sum = sum(num for tok, num in vocab)
            counter_sum = sum(num for tok, num in counter.items())
            print(f"vocab: {vocab_sum}/{counter_sum}({vocab_sum / counter_sum * 100}\%) left")
            vocab = [(tok, 1) for tok in specials] + vocab
            vocab = [v for v, n in vocab]
        if prev_vocab is not None:
            v_diff = set(vocab) - set(prev_vocab.keys())
            total_num = len(v_diff) + len(list(prev_vocab.keys()))
            idx_diff = [i for i in range(total_num) if i not in prev_vocab.values()]
            v_diff = {tok: i for tok, i in zip(list(v_diff), idx_diff)}
            vocab = {**prev_vocab, **v_diff}
        else:
            vocab = {tok: i for i, tok in enumerate(vocab)}
        return vocab

    def _tokenize(self, text):
        return self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens)

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    @property
    def vocab_size(self):
        return len(self.vocab)


@ex.capture
def build_word_embedding(tokenizer, pretrained_embedding):
    if pretrained_embedding is None:
        return None
    elif not pretrained_embedding:
        return None
    embedding = pretrained_aliases[pretrained_embedding]()
    vectors = [tok for ids, tok in tokenizer.ids_to_tokens.items()]
    return embedding.get_vecs_by_tokens(vectors, lower_case_backup=True)

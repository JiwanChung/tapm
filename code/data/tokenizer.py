import torch
import collections
from itertools import chain

from tqdm import tqdm

from pytorch_transformers import PreTrainedTokenizer, BasicTokenizer


def build_tokenizer(args, data):
    path = args.data_path['train'].parent / f'whitespace_vocab_{args.max_vocab}.pickle'
    if not path.is_file():
        print('building vocab')
        tokenizer = WhitespaceTokenizer(args, data)
        for token in ['mask', 'sep', 'cls', 'pad', 'unk']:
            setattr(tokenizer, f'{token}_id',
                    tokenizer.encode(getattr(tokenizer, f'{token}_token'))[0])
        torch.save(tokenizer, path)
    else:
        print('loading vocab')
        tokenizer = torch.load(path)

    print(f"vocab num: {len(tokenizer)}")

    return tokenizer


class WhitespaceTokenizer(PreTrainedTokenizer):
    def __init__(self, args, data):
        self.special_tokens = {
            'mask_token': '[MASK]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'pad_token': '[PAD]',
            'unk_token': '[UNK]'
        }
        super(WhitespaceTokenizer, self).__init__(**self.special_tokens)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=True)

        self.max_vocab = args.get('max_vocab', 20000)

        counter = self.count(data)
        self.vocab = self.counter_to_vocab(counter)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

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

    def counter_to_vocab(self, counter):
        specials = list(self.special_tokens.values())
        if self.max_vocab is None:
            vocab = list(counter.items())
        else:
            vocab = counter.most_common(self.max_vocab - len(specials))
        vocab_sum = sum(num for tok, num in vocab)
        counter_sum = sum(num for tok, num in counter.items())
        print(f"vocab: {vocab_sum}/{counter_sum}({vocab_sum / counter_sum * 100}\%) left")
        vocab = [(tok, 1) for tok in specials] + vocab
        vocab = {tok: i for i, (tok, num) in enumerate(vocab)}
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

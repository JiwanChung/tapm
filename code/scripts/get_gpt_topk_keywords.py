from .common import load_data, run, GetWords
from transformers import (
    GPT2Tokenizer
)


def main():
    run(GetTopkWords(), 'gpt_top')


class GetTopkWords(GetWords):
    def __init__(self):
        super(GetTopkWords, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenize = self.tokenizer._tokenize
        self.filter_tag = lambda x: True

    def __call__(self, sent):
        return super().__call__(sent)

from collections import defaultdict

from utils import suppress_stdout

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


class Metric:
    def __init__(self, args):
        metrics = args.get('metrics', ['meteor', 'bleu', 'rouge', 'cider'])

        self.tokenizer = PTBTokenizer()
        scorers = {
            'bleu': (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            'meteor': (Meteor(),"METEOR"),
            'rouge': (Rouge(), "ROUGE_L"),
            'cider': (Cider(), "CIDEr")   # CIDEr is only applicable to the whole dataset since it
        }
        self.scorers = [v for k, v in scorers.items() if k in metrics]

        self.debug = args.get('debug', False)

    def calculate(self, texts):
        if self.debug:
            return self._calculate(texts)
        else:
            with suppress_stdout():
                return self._calculate(texts)

    def _calculate(self, texts):
        hypo = {k: [{'caption': remove_nonascii(v[0])}] for k, v in texts.items()}
        tgt = {k: [{'caption': remove_nonascii(v[1])}] for k, v in texts.items()}
        with suppress_stdout():
            hypo = self.tokenizer.tokenize(hypo)
            tgt = self.tokenizer.tokenize(tgt)

        data = defaultdict(float)
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(tgt, hypo)
            if type(method) == list:
                for sc, m in zip(score, method):
                    data[m] += sc
            else:
                data[method] += score
        return data

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

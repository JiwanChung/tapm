from collections import defaultdict

from utils import suppress_stdout

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


class Metric:
    ngrams = ['bleu', 'meteor', 'rouge', 'meteor']

    def __init__(self, args):
        metrics = args.get('metrics', ['meteor', 'bleu', 'rouge'])
        self.tokenizer = PTBTokenizer()
        scorers = {
            'bleu': (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            'meteor': (Meteor(),"METEOR"),
            'rouge': (Rouge(), "ROUGE_L"),
            # 'cider': (Cider(), "CIDEr") CIDEr is only applicable to the whole dataset since it
            # uses document-wise tf-idf
        }
        self.scorers = [v for k, v in scorers.items() if k in metrics]

        self.debug = args.get('debug', False)

    def calculate(self, hypo, gts):
        with suppress_stdout(not self.debug):
            data = defaultdict(float)
            if not isinstance(gts, list):
                gts = {str(1): [gts]}
                hypo = {str(1): [hypo]}
            else:
                gts = {str(k): [v] for k, v in enumerate(gts)}
                hypo = {str(k): [v] for k, v in enumerate(hypo)}
            for scorer, method in self.scorers:
                score, scores = scorer.compute_score(gts, hypo)
                if type(method) == list:
                    for sc, m in zip(score, method):
                        data[m] += sc
                else:
                    data[method] += score
        return data

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

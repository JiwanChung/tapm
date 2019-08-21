from collections import defaultdict

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


class Metric:
    ngrams = ['bleu', 'meteor', 'rouge', 'meteor']

    def __init__(self, args):
        metrics = args.get('metrics', ['meteor', 'bleu'])
        self.tokenizer = PTBTokenizer()
        scorers = {
            'bleu': (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            'meteor': (Meteor(),"METEOR"),
            'rouge': (Rouge(), "ROUGE_L"),
            'cider': (Cider(), "CIDEr")
        }
        self.scorers = [v for k, v in scorers.items() if k in metrics]

    def calculate(self, hypo, gts):
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

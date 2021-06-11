from collections import defaultdict

from utils import suppress_stdout, break_tie, remove_nonascii

from exp import ex

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from .vist_tokenizer import VistTokenizer
from .vist_meteor import VistMeteor


class Metric:
    @ex.capture
    def __init__(self, metrics, debug, use_vist=False):
        self.tokenizer = PTBTokenizer()
        # self.vist_tokenizer = VistTokenizer()
        scorers = {
            'bleu': (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            'meteor': (Meteor(),"METEOR"),
            'rouge': (Rouge(), "ROUGE_L"),
            'cider': (Cider(), "CIDEr")   # CIDEr is only applicable to the whole dataset since it
        }
        self.use_vist = use_vist
        if self.use_vist:
            print("using vist meteor options")
            scorers['meteor'] = (VistMeteor(), "METEOR")

        self.scorers = [v for k, v in scorers.items() if k in metrics]

        self.debug = debug

    def calculate(self, texts):
        if self.debug:
            return self._calculate(texts)
        else:
            with suppress_stdout():
                return self._calculate(texts)

    def _calculate(self, texts):
        # story_map: video_id/[story_ids]
        hypo = {k: [{'caption': remove_nonascii(v[0])}] for k, v in texts.items()}
        tgt = {k: [v[1]] if not isinstance(v[1], list) else v[1] for k, v in texts.items()}
        tgt = {k: [{'caption': remove_nonascii(v1)} for v1 in v] for k, v in tgt.items()}

        with suppress_stdout():
            if self.use_vist:
                hypo = {k: [x['caption'] for x in v] for k, v in hypo.items()}
                tgt = {k: [x['caption'] for x in v] for k, v in tgt.items()}
                '''
                hypo = self.vist_tokenizer.tokenize(hypo)
                tgt = self.vist_tokenizer.tokenize(tgt)
                '''
            else:
                hypo = self.tokenizer.tokenize(hypo)
                tgt = self.tokenizer.tokenize(tgt)

        return self.score(hypo, tgt)

    def score(self, hypo, tgt):
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

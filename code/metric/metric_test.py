from collections import defaultdict

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from vist_tokenizer import VistTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from vist_meteor_test import VistMeteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

'''
standalone script for testing
'''


def remove_nonascii(text):
    # return ''.join([i if ord(i) < 128 else ' ' for i in text])
    return text.encode('ascii', 'ignore').decode('ascii')


class Metric:
    def __init__(self, metrics, debug, use_vist=True, no_tokenizer=False):
        self.tokenizer = PTBTokenizer()
        scorers = {
            'bleu': (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            'meteor': (Meteor(),"METEOR"),
            'rouge': (Rouge(), "ROUGE_L"),
            'cider': (Cider(), "CIDEr")   # CIDEr is only applicable to the whole dataset since it
        }
        self.use_vist = use_vist
        self.no_tokenizer = no_tokenizer
        if self.use_vist:
            self.vist_tokenizer = VistTokenizer()
            scorers['meteor'] = (VistMeteor(), "METEOR")

        self.scorers = [v for k, v in scorers.items() if k in metrics]

        self.debug = debug

    def calculate(self, texts, album_map=None):
        return self._calculate(texts, album_map)

    def _calculate(self, texts, story_map=None):
        # story_map: video_id/[story_ids]
        hypo = {k: [{'caption': v[0]}] for k, v in texts.items()}
        tgt = {k: [v[1]] if not isinstance(v[1], list) else v[1] for k, v in texts.items()}
        tgt = {k: [{'caption': v1} for v1 in v] for k, v in tgt.items()}

        if self.no_tokenizer:
            hypo = {k: [x['caption'] for x in v] for k, v in hypo.items()}
            tgt = {k: [x['caption'] for x in v] for k, v in tgt.items()}
        elif self.use_vist:
            # tgt = self.vist_tokenizer.tokenize(tgt)
            tgt = {k: [x['caption'] for x in v] for k, v in tgt.items()}
            hypo = {k: [x['caption'] for x in v] for k, v in hypo.items()}
        else:
            hypo = {k: [{'caption': remove_nonascii(x['caption'])} for x in v] for k, v in hypo.items()}
            tgt = {k: [{'caption': remove_nonascii(x['caption'])} for x in v] for k, v in tgt.items()}
            hypo = self.tokenizer.tokenize(hypo)
            tgt = self.tokenizer.tokenize(tgt)

        return self.score(hypo, tgt)

    def score(self, hypo, tgt):
        data = defaultdict(float)
        for scorer, method in self.scorers:
            print(f"running {method}")
            score, scores = scorer.compute_score(tgt, hypo)
            if type(method) == list:
                for sc, m in zip(score, method):
                    data[m] += sc
            else:
                data[method] += score
        return data

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

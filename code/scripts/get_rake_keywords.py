from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake

from .common import load_data, run, GetWords


def main():
    run(GetRakeWords(), 'rake')


class GetRakeWords(GetWords):
    def __init__(self):
        super(GetRakeWords, self).__init__()

        self.rake = Rake()

    def __call__(self, sent):
        self.rake.extract_keywords_from_text(sent)
        keywords = self.rake.get_ranked_phrases_with_scores()
        res = []
        for v, k in keywords:
            ks = super().__call__(k)
            v = v // len(ks) if len(ks) > 0 else 0
            for w in ks:
                res.append((w, v))
        return list(res)


if __name__ == '__main__':
    main()

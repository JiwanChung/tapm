from pathlib import Path
from tqdm import tqdm
from collections import Counter
import json

import nltk
from nltk.stem import WordNetLemmatizer

from exp import ex


@ex.capture(prefix='scripts')
def run(get_words, custom_file_name, path, num_keywords):
    n = num_keywords
    path = Path(path)
    data = load_data(path)
    process(path, data, get_words, n, custom_file_name)


def process(path, data, get_words, n=1000, file_name=None):
    counter = get_counter(data, get_words)
    print(counter.most_common(n))
    keywords = dict({k: v for k, v in counter.most_common(n)})
    path = path.parent / 'keywords'
    path.mkdir(exist_ok=True)
    path = path / f"keywords_{file_name}_{n}.json"
    print(f"Saving to {path}")
    with open(path, 'w') as f:
        json.dump(keywords, f, indent=4)


def get_counter(data, get_words):
    word_sets = list(map(get_words, tqdm(data.values(), total=len(data))))
    counter = Counter()
    for word_set in word_sets:
        for word in word_set:
            score = 1
            if isinstance(word, tuple) or isinstance(word, list):
                word, score = word
            counter[word] += score
    return counter


def load_data(path):
    data = []
    keys = ['vid', 'vid_start', 'vid_end',
            'cap_start', 'cap_end', 'caption']
    with open(path, 'r') as f:
        for row in f:
            row = row.split('\t')
            row = {k: v for k, v in zip(keys, row)}
            row['caption'] = row['caption'].strip()
            data.append(row)
    data = {i['vid']: i['caption'] for i in data}
    return data


class GetWords(object):
    def __init__(self):
        super(GetWords, self).__init__()

        self.lemmatizer = WordNetLemmatizer()
        self.lemmatize = self.lemmatizer.lemmatize
        self.tokenize = nltk.word_tokenize
        self.tag = lambda x: nltk.pos_tag(x)
        self.filter_tag = lambda tag: (tag.startswith('NN') or tag.startswith('VB'))

    def __call__(self, sent):
        sent = self.tokenize(sent)
        sent = [word for word, tag in self.tag(sent) if self.filter_tag(tag)]
        return list([self.lemmatize(w).lower() for w in sent])

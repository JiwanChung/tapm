from collections import Counter
from pathlib import Path
import nltk
import json
from nltk.stem import WordNetLemmatizer


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


def main(path='../data/LSMDC/task1/LSMDC16_annos_training_someone.csv', n=20000):
    path = Path(path)
    name = path.name
    data = load_data(path)
    get_words = GetWords()
    word_sets = list(map(get_words, data.values()))
    counter = Counter()
    for word_set in word_sets:
        for word in word_set:
            counter[word] += 1
    print(counter.most_common(n))
    keywords = dict({k: v for k, v in counter.most_common(n)})
    path = path.parent / f"keywords_top{n}"
    path.mkdir(exist_ok=True)
    with open(path / f'{name}.json', 'w') as f:
        json.dump(keywords, f, indent=4)


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


if __name__ == "__main__":
    main()

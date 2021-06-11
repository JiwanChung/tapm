import re
from itertools import chain
import stanfordnlp
import json
from tqdm import tqdm
from pathlib import Path

from exp import ex


@ex.capture()
def main(scripts, sample):
    path = scripts['path']
    path = Path(path)

    NAME_LIST = ['Alice', 'Bob', 'Charlie', 'Tom', 'Robert', 'Ben', 'John', 'James', 'Alex', 'Steve', 'Lisa']
    NAME_LIST = [k.lower() for k in NAME_LIST]
    PERSON_TOKEN = 'it'

    blank = '[...]'
    data = {}
    print("loading data")
    with open(path, 'r') as f:
        load_count = 1
        for line in f:
            line = line.split('\t')
            key = line[0]
            text = line[-1].strip()
            i = 0
            orig = text
            while text.find(blank) >= 0:
                text = text.replace(blank, PERSON_TOKEN, 1)
                orig = orig.replace(blank, NAME_LIST[i], 1)
                i += 1
            data[key] = (orig, text, i)  # num_someones
            if sample and load_count >= 100:
                break
            load_count += 1

    nlp = stanfordnlp.Pipeline()

    def sample_run(sample):
        orig, text, num_someones = sample
        doc = nlp(text)
        deps_temp = doc.sentences[0].dependencies

        orig_words = nlp(orig).sentences[0].words
        orig_words_dt = {k.index: k.text for k in orig_words}
        deps = []
        for dep in deps_temp:
            tok1, rel, tok2 = dep
            if int(tok1.index) > 0:  # skip root
                if tok1.index not in orig_words_dt:
                    import ipdb; ipdb.set_trace()  # XXX DEBUG
                tok1.text = orig_words_dt[tok1.index]
            if int(tok2.index) > 0:  # skip root
                if tok2.index not in orig_words_dt:
                    import ipdb; ipdb.set_trace()  # XXX DEBUG
                tok2.text = orig_words_dt[tok2.index]
            deps.append((tok1, rel, tok2))

        res = []

        def find_verb(tokens):
            relss = []
            for token in tokens:
                rels = []
                for relation in deps:
                    token1, relation, token2 = relation
                    if token1.text == token:
                        pos = token1.xpos
                        text = token2.text
                        upos = token2.upos
                        if upos == 'VERB':
                            return (pos, text, upos, relation)
                        rels.append((pos, text, upos, relation))
                    if token2.text == token:
                        pos = token2.xpos
                        text = token1.text
                        upos = token1.upos
                        if upos == 'VERB':
                            return (pos, text, upos, relation)
                        rels.append((pos, text, upos, relation))
                relss.append(rels)
            return relss

        for someone in [NAME_LIST[i] for i in range(num_someones)]:
            tokens = [someone]
            tags = None
            for i in range(len(doc.sentences[0].words)):
                rels = find_verb(tokens)
                if isinstance(rels, tuple):
                    tags = rels
                    break
                rels = chain(*rels)
                tokens = [k[1] for k in rels]
            if tags is None:
                import ipdb; ipdb.set_trace()  # XXX DEBUG
            pos = [k.xpos for k in orig_words if k.text == someone][0]
            res.append({'pos': pos, 'verb': tags[1], 'relation': tags[3]})
        assert num_someones == len(res)

        return res

    output = {}
    for k, v in tqdm(data.items(), total=len(list(data.keys()))):
        output[k] = sample_run(v)

    with open(path.parent / f'verb_{path.stem}.json', 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    main()

import random
import math
import copy
import itertools

import torch
import numpy as np

from utils import jsonl_to_json, remove_duplicate, flatten_list, remove_nonascii


def make_bert_batch(tokenizer, data, **kwargs):
    data = jsonl_to_json(data)
    sentences = data['target']
    sentences = [tokenizer.encode(t) for t in sentences]
    max_limit = kwargs.get('max_sentence_tokens', None)
    if max_limit is not None:
        sentences = list([t[:max_limit] for t in sentences])
    sentences = [torch.Tensor([tokenizer.cls_id,
                               *t,
                               tokenizer.sep_id]) for t in sentences]
    sentences, lengths = pad(sentences, tokenizer.pad_id)
    targets = sentences.clone()

    return {'sentences': sentences,
            'lengths': lengths,
            'targets': targets}


def make_keyword_batch(tokenizer, data, concat=False, keywords=None, lemmatize=None, **kwargs):
    data = jsonl_to_json(data)
    sentences = data['target']
    sentences = [tokenizer.encode(t) for t in sentences]
    max_limit = kwargs.get('max_sentence_tokens', None)
    if max_limit is not None:
        sentences = list([t[:max_limit] for t in sentences])

    ordered_keywords = [[tokenizer.convert_ids_to_tokens(token) for token in sentence] for sentence in sentences]
    ordered_keywords = [[lemmatize(token) for token in sentence] for sentence in ordered_keywords]
    ordered_keywords = [[tokenizer.convert_tokens_to_ids(token) for token in sentence] for sentence in ordered_keywords]
    ordered_keywords = [[(i, token) for i, token in enumerate(sentence) if token in keywords] for sentence in ordered_keywords]
    ordered_keywords = [remove_duplicate(sentence, key=lambda x: x[1]) for sentence in ordered_keywords]
    keyword_ids = [torch.Tensor([i for i, token in sentence]) for sentence in ordered_keywords]
    ordered_keywords = [[token for i, token in sentence] for sentence in ordered_keywords]
    unordered_keywords = [[(token, keywords[token]) for token in sentence] for sentence in ordered_keywords]
    unordered_keywords = [sorted(sentence, key=lambda x: x[1], reverse=False) for sentence in unordered_keywords]
    unordered_keywords = [map(lambda x: x[0], sentence) for sentence in unordered_keywords]
    unordered_keywords = [torch.Tensor([tokenizer.cls_id, *keyword, tokenizer.sep_id, ]) for keyword in unordered_keywords]
    ordered_keywords = [torch.Tensor([tokenizer.cls_id, *keyword, tokenizer.sep_id, ]) for keyword in ordered_keywords]
    targets = [torch.Tensor([*t, tokenizer.sep_id]) for t in sentences]
    sentences = [torch.Tensor([tokenizer.cls_id, *t]) for t in sentences]
    # tensor B*L
    sentences, lengths = pad(sentences, tokenizer.pad_id)
    targets, _ = pad(targets, tokenizer.pad_id)
    ordered_keywords, _ = pad(ordered_keywords, tokenizer.pad_id)
    unordered_keywords, _ = pad(unordered_keywords, tokenizer.pad_id)
    keyword_ids, _ = pad(keyword_ids, tokenizer.pad_id)

    return {'sentences': sentences,
            'lengths': lengths,
            'targets': targets,
            'keywords': unordered_keywords,
            'ordered_keywords': ordered_keywords,
            'keyword_ids': keyword_ids}


def make_mask_model_batch(tokenizer, data, random_idx=True, complementary=False, **kwargs):
    data = jsonl_to_json(data)
    sentences = data['target']
    if type(sentences) == 'list':
        sentences = [x for r in sentences for x in r]
    sentences = [tokenizer.encode(t) for t in sentences]
    max_limit = kwargs.get('max_sentence_tokens', None)
    if max_limit is not None:
        sentences = list([t[:max_limit] for t in sentences])
    sentences = [torch.Tensor([tokenizer.cls_id,
                               *t,
                               tokenizer.sep_id]) for t in sentences]
    targets, _ = pad(sentences, tokenizer.pad_id)
    # mask words
    sentences, mask_ids = mask_words(sentences, tokenizer.mask_id, random_idx, complementary)
    if random_idx:
        # tensor B*L
        sentences, lengths = pad(sentences, tokenizer.pad_id)
    else:
        # len B list of tensor L*L
        li = []
        lengths = []
        for sentence in sentences:
            sentence, length = pad(sentence, tokenizer.pad_id)
            li.append(sentence)
            lengths.append(length)
        sentences = li

    return {'sentences': sentences,
            'lengths': lengths,
            'targets': targets,
            'mask_ids': mask_ids}


def mask_words(tensors, mask_idx, random_idx=True, complementary=False):
    # B(CLS+L+SEP)
    if random_idx:
        # mask random idx
        li = []
        ids = []
        for t in tensors:
            device = t.device
            idx = random.randint(1, t.shape[0] - 1)
            ids.append(idx)
            if complementary:
                val = t[idx].item()
                t.fill_(mask_idx)
                t[idx] = val
            else:
                t[idx] = mask_idx
            li.append(t)
        return li, torch.Tensor(ids).long().to(device)
    else:
        # generate mask for every word
        li = []
        for t in tensors:
            t = t.unsqueeze(0).repeat(t.shape[0], 1)
            eye = torch.eye(t.shape[0]).bool().to(t.device)
            if complementary:
                eye = ~eye
            full = torch.full(t.shape, mask_idx, dtype=t.dtype).to(t.device)
            t.masked_scatter_(mask=eye, source=full)
            li.append(t)
        # list of L*L
        return li, None


def make_autoencoder_batch(tokenizer, data, **kwargs):
    data = jsonl_to_json(data)
    sentences = data['target']
    sentences = [tokenizer.encode(t) for t in sentences]

    targets = [torch.Tensor([*t, tokenizer.eos_id]) for t in sentences]
    sentences = [torch.Tensor([tokenizer.sos_id, *t]) for t in sentences]
    sentences, lengths = pad(sentences, tokenizer.pad_id)
    targets, _ = pad(targets, tokenizer.pad_id)

    return {'sentences': sentences,
            'lengths': lengths,
            'targets': targets}


def make_subset_mask_batch(tokenizer, data, random_idx=True,
                           keyword_min=1, keyword_max_ratio=0.4,
                           **kwargs):
    data = jsonl_to_json(data)
    sentences = data['target']
    sentences = [tokenizer.encode(t) for t in sentences]

    sentences = [torch.Tensor([tokenizer.cls_id, *t, tokenizer.sep_id]) for t in sentences]
    targets = copy.deepcopy(sentences)
    targets, lengths = pad(targets, tokenizer.pad_id)
    if random_idx:
        keyword_ids = []
        for i, sentence in enumerate(sentences):
            length = len(sentence) - 2
            max_length = math.ceil(max(keyword_min, keyword_max_ratio * length))
            keyword_num = random.randrange(keyword_min, max_length + 1)
            ids = list(range(1, length + 1))
            random.shuffle(ids)
            mask_idx = ids[keyword_num:]
            mask_idx = torch.LongTensor(mask_idx).to(sentence.device)
            keyword_idx = ids[:keyword_num]
            keyword_idx = torch.LongTensor(keyword_idx).to(sentence.device)
            sentence[mask_idx] = tokenizer.mask_id
            sentences[i] = sentence
            keyword_ids.append(keyword_idx)
        sentences, lengths = pad(sentences, tokenizer.pad_id)
        keyword_ids, _ = pad(keyword_ids, tokenizer.pad_id)
        return {'sentences': sentences,
                'lengths': lengths,
                'targets': targets,
                'keyword_ids': keyword_ids}
    else:
        # for memory reasons, we should not keep source for every combinations
        '''
        keyword_ids = []
        for i, sentence in enumerate(sentences):
            length = len(sentence) - 2
            max_length = math.ceil(max(keyword_min, keyword_max_ratio * length))
            ids = list(range(1, length + 1))
            combs = []
            for L in range(1, max_length + 1):
                comb_L = []
                for subset in itertools.combinations(ids, L):
                    comb_L.append(subset)
                comb_L = torch.LongTensor(comb_L).to(sentence.device)
                combs.append(comb_L)
            keyword_ids.append(combs)
        return {'sentences': targets,
                'targets': targets,
                'keyword_ids': keyword_ids}
        '''
        # generate masks dynamically
        return {'sentences': targets,
                'lengths': lengths,
                'targets': targets}


class BPECap(object):
    def __init__(self):
        super(BPECap, self).__init__()
        self.small_prefix = b'\xc4\xa1'.decode()
        self.big_prefix = b'\xc4\xa0'.decode()

    def __call__(self, x):
        return x.replace(self.small_prefix, self.big_prefix)


class ConvertToken(object):
    def __init__(self):
        super(ConvertToken, self).__init__()

        self.bpe_capitalize = BPECap()

    def __call__(self, tokenizer, token):
        return tokenizer.convert_tokens_to_ids([self.bpe_capitalize(token)])[0]


def make_feature_lm_batch_with_keywords(tokenizer, data, keywords=None,
                                        word_counter=None, feature_name_map={},
                                        concat_group=False, use_vist=False,
                                        force_ascii=False,
                                        **kwargs):
    # data: list of chunks: list of [item dict]
    data = jsonl_to_json(data)

    group_length = [len(i) for i in data['vid']]
    group_mask = torch.zeros(len(data['vid']), max(group_length)).bool()
    for i, length in enumerate(group_length):
        group_mask[i, :length] = 1

    # keyword_counter = keywords
    if keywords is not None:
        # restore gpt token prefix
        # [:len(keywords)] part is arbitrary, and causes some bugs apparently...
        # keywords = torch.Tensor(list(itertools.chain(*[tokenizer.encode(token) for token in keywords]))[:len(keywords)]).long()
        convert_token = ConvertToken()
        keywords = torch.Tensor([convert_token(tokenizer, token) for token in list(keywords.keys())]).long()

    if 'target' in data:
        batch_sentences = data['target']

        def get_text(sentences):
            '''
            if use_vist:
                sentences = [tokenizer.decode(tokenizer.encode(t)) for t in sentences]
            '''
            if force_ascii:
                sentences = [remove_nonascii(t) for t in sentences]
            sentences = [tokenizer.encode(t, add_special_tokens=False) for t in sentences]
            max_limit = kwargs.get('max_sentence_tokens', None)
            if max_limit is not None:
                sentences = list([t[:max_limit] for t in sentences])  # undo ptb tokenization
            # tensor B*L
            if concat_group:
                sentences = [[*t, tokenizer.seq_sep_id] for t in sentences]
                lengths = torch.LongTensor([len(s) for s in sentences])
                sentences = flatten_list(sentences)
                # [:-1]  # do not remove last seq_sep token
                targets = torch.Tensor([[*sentences, tokenizer.sep_id]])
                sentences = torch.Tensor([[tokenizer.cls_id, *sentences]])
            else:
                targets = [torch.Tensor([*t, tokenizer.sep_id]) for t in sentences]
                targets, _ = pad(targets, tokenizer.pad_id)
                sentences = [torch.Tensor([tokenizer.cls_id, *t]) for t in sentences]
                sentences, lengths = pad(sentences, tokenizer.pad_id)
            '''
            word_subset = torch.zeros(sentences.shape[0], len(tokenizer)).bool().to(sentences.device)
            word_subset = word_subset.scatter(dim=-1, index=sentences, value=1)
            word_subset = [i.squeeze() for i in word_subset.split(1, dim=0)]
            keyword_mask = None
            if keywords is not None:
                keyword_mask = sentences.unsqueeze(-1).expand(-1, -1, keywords.shape[0]) == keywords.view(1, 1, -1)
                keyword_mask = keyword_mask.long().sum(dim=1) > 0  # VN
                keyword_mask = [i.squeeze() for i in keyword_mask.split(1, dim=0)]
            '''
            return sentences, lengths, targets

        sentences, lengths, targets = zip(*[get_text(sentence) for sentence in batch_sentences])
        sentences, batch_lengths = pad(sentences, tokenizer.pad_id)
        targets, _ = pad(targets, tokenizer.pad_id)
        lengths, _ = pad(lengths, 0)

        '''
        word_subsets = pad_tensor(word_subsets, 0)
        word_subsets[:, :, tokenizer.pad_id] = 0
        '''

        ret_batch = {
            'sentences': sentences,
            'batch_lengths': batch_lengths,
            'lengths': lengths,
            'targets': targets,
        }
    else:
        ret_batch = {}

    ret_batch = {
        **ret_batch,
        'vid': data['vid'],
        'group_mask': group_mask,
    }

    if 'album_id' in data:
        ret_batch = {
            **ret_batch,
            'album_id': data['album_id'],
        }
    if 'image_id' in data:
        ret_batch = {
            **ret_batch,
            'image_id': data['image_id'],
        }

    if 'frame' in data:
        ret_batch = {
            **ret_batch,
            'frame': pad_tensor(data['frame'], 0).long(),
        }

    # Process features if applicable
    for k, v in feature_name_map.items():
        if k in data:
            try:
                ret_batch[v] = pad_tensor(data[k], 0)
            except Exception as e:
                print(k)
                print(data['vid'])
                print(e)
                from pdb_multi import set_trace; set_trace()

    return ret_batch


def make_blank_filling_batch(tokenizer, data, feature_name_map={}, **kwargs):
    data = jsonl_to_json(data)
    sentences = data['input']
    targets = data['target']
    sentences = [tokenizer.encode(t) for t in sentences]
    sentences = [torch.Tensor(t) for t in sentences]
    sentences, lengths = pad(sentences, tokenizer.pad_id)
    targets = [tokenizer.encode(t) for t in targets]
    targets = [torch.Tensor(t) for t in targets]
    targets, _ = pad(targets, tokenizer.pad_id)

    blank_ids = sentences == tokenizer.convert_tokens_to_ids(tokenizer.blank)

    ret_batch = {
        'sentences': sentences,
        'lengths': lengths,
        'targets': targets,
        'blank_ids': blank_ids,
        'blank_num': data['blank_num'],
        'vid': data['vid']
    }

    for k,v in feature_name_map.items():
        if k in data:
            ret_batch[v] = pad_tensor(data[k], 0)

    return ret_batch


def pad(x, pad_id=0):
    B = len(x)
    max_size, dtype = get_max_size(x)
    storage = torch.full(max_size, pad_id, dtype=torch.long).to(x[0].device)
    lengths = []
    def add_data(ids, t):
        if hasattr(t, 'shape'):
            if not torch.is_tensor(t):
                t = torch.from_numpy(t)
            t_shape = [slice(None, j) for j in t.shape]
            storage[tuple([*ids, *t_shape])] = t
        else:
            for i in range(len(t)):
                add_data([*ids, i], t[i])
    add_data([], x)
    lengths = torch.LongTensor(lengths).to(x[0].device)

    return storage, lengths


def remove_pad(x, pad_id=0):
    return x[:(x != pad_id).sum(-1)]


def remove_past_idx(x, idx=0):
    idx = (x == idx).nonzero()
    if idx.nelement() > 0:
        idx = idx[0]
    else:
        idx = x.shape[0]
    return x[: idx + 1]


def decode_tensor(tokenizer, x, split_tokens=False, use_vist=False, remove_past_sep=False):
    if x.dim() < 1:
        x = x.unsqueeze(0)
    x = remove_pad(x, tokenizer.pad_id)
    if remove_past_sep:
        x = remove_past_idx(x, tokenizer.sep_id)
    x = list(x.cpu().numpy())

    x_cut = []
    temp = []
    for tok in x:
        if tok == tokenizer.seq_sep_id:
            x_cut.append(temp)
            temp = []
        else:
            temp.append(tok)
    x_cut.append(temp)
    x_cut = [[int(i) for i in x] for x in x_cut]

    if split_tokens:
        return flatten_list([tokenizer.convert_ids_to_tokens(x) for x in x_cut])
    elif use_vist:
        return ' '.join([decode_vist(x, tokenizer) for x in x_cut])
    else:
        return ' '.join(tokenizer.decode(x) for x in x_cut)


def decode_vist(x, tokenizer):
    # decode vist for gpt2 tokenizer
    tokenizer.whitespace = getattr(tokenizer, 'whitespace', b'\xc4\xa0'.decode())
    x = tokenizer.convert_ids_to_tokens(x)
    x = [f" {v[1:]}" if v.startswith(tokenizer.whitespace) else v for v in x]
    return ''.join(x)


def get_max_size(t):
    if hasattr(t, 'shape'):
        if not torch.is_tensor(t):
            t = torch.from_numpy(t)
        return list(t.shape), t.dtype
    else:
        # get max
        t = [get_max_size(i) for i in t]
        dtype = t[0][1]
        t = [i[0] for i in t]
        return [len(t), *list(np.array(t).max(axis=0))], dtype


def pad_tensor(x, val=0):
    max_size, _ = get_max_size(x)
    dtype = torch.float
    storage = torch.full(max_size, val, dtype=dtype)

    def add_data(ids, t):
        if hasattr(t, 'shape'):
            if not torch.is_tensor(t):
                t = torch.from_numpy(t)
            t_shape = [slice(None, j) for j in t.shape]
            t_shape = [*ids, *t_shape]
            storage[t_shape] = t
        else:
            for i in range(len(t)):
                add_data([*ids, i], t[i])

    add_data([], x)

    return storage


def make_fib_batch(tokenizer, data, feature_name_map={}, **kwargs):
    data = jsonl_to_json(data)

    group_length = [len(i) for i in data['vid']]
    group_mask = torch.zeros(len(data['vid']), max(group_length)).bool()
    for i, length in enumerate(group_length):
        group_mask[i, :length] = 1

    def get_text(src, tgt):
        src = tokenizer.encode(src)
        tgt = tokenizer.encode(tgt)
        mask_idx = src.index(tokenizer.mask_id)
        length_diff = len(tgt) - len(src)
        src_extended = src[:mask_idx + 1] + [tokenizer.mask_id] * length_diff + src[mask_idx + 1:]
        return src_extended, tgt

    def process_texts(srcs, tgts):
        srcs, tgts = zip(*[get_text(src, tgt) for src, tgt in zip(srcs, tgts)])
        srcs = [torch.Tensor(t) for t in srcs]
        tgts = [torch.Tensor(t) for t in tgts]
        srcs, _ = pad(srcs, tokenizer.pad_id)
        tgts, lengths = pad(tgts, tokenizer.pad_id)

        return srcs, tgts, lengths

    srcs, tgts, lengths = zip(*[process_texts(src, tgt) for src, tgt in \
                                        zip(data['source'], data['target'])])

    sentences, batch_lengths = pad(srcs, tokenizer.pad_id)
    targets, _ = pad(tgts, tokenizer.pad_id)
    lengths, _ = pad(lengths, 0)

    ret_batch = {
        'sentences': sentences,
        'batch_lengths': batch_lengths,
        'lengths': lengths,
        'targets': targets
    }

    ret_batch = {
        **ret_batch,
        'vid': data['vid'],
        'answer': data['answer'],
        'group_mask': group_mask,
    }

    if 'album_id' in data:
        ret_batch = {
            **ret_batch,
            'album_id': data['album_id'],
        }
    if 'image_id' in data:
        ret_batch = {
            **ret_batch,
            'image_id': data['image_id'],
        }

    if 'frame' in data:
        ret_batch = {
            **ret_batch,
            'frame': pad_tensor(data['frame'], 0).long(),
        }

    # Process features if applicable
    for k, v in feature_name_map.items():
        if k in data:
            try:
                ret_batch[v] = pad_tensor(data[k], 0)
            except Exception as e:
                print(k)
                print(data['vid'])
                print(e)
                from pdb_multi import set_trace; set_trace()

    return ret_batch


def make_multichoice_batch(tokenizer, data, feature_name_map={}, **kwargs):
    data = jsonl_to_json(data)

    group_length = [len(i) for i in data['vid']]
    group_mask = torch.zeros(len(data['vid']), max(group_length)).bool()
    for i, length in enumerate(group_length):
        group_mask[i, :length] = 1

    def get_text(tgt):
        tgt = tokenizer.encode(tgt)
        return tgt

    def process_texts(tgts):
        tgts = [get_text(tgt) for tgt in tgts]
        tgts = [torch.Tensor(t) for t in tgts]
        tgts, lengths = pad(tgts, tokenizer.pad_id)

        return tgts, lengths

    tgts, lengths = zip(*[process_texts(tgt) for tgt in data['target']])

    targets, batch_lengths = pad(tgts, tokenizer.pad_id)
    lengths, _ = pad(lengths, 0)
    sentences = targets

    answer = data['answer']
    try:
        answer = [int(batch_a[0]) for batch_a in answer]  # all choices has same true answer idx
    except:
        print(answer)
    answer = torch.Tensor(answer)
    targets = answer

    ret_batch = {
        'sentences': sentences,
        'batch_lengths': batch_lengths,
        'lengths': lengths,
        'targets': targets
    }

    ret_batch = {
        **ret_batch,
        'vid': data['vid'],
        'answer': data['answer'],
        'group_mask': group_mask,
    }

    if 'album_id' in data:
        ret_batch = {
            **ret_batch,
            'album_id': data['album_id'],
        }
    if 'image_id' in data:
        ret_batch = {
            **ret_batch,
            'image_id': data['image_id'],
        }

    if 'frame' in data:
        ret_batch = {
            **ret_batch,
            'frame': pad_tensor(data['frame'], 0).long(),
        }

    # Process features if applicable
    for k, v in feature_name_map.items():
        if k in data:
            try:
                ret_batch[v] = pad_tensor(data[k], 0)
            except Exception as e:
                print(k)
                print(data['vid'])
                print(e)
                from pdb_multi import set_trace; set_trace()

    return ret_batch

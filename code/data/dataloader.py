import json

from torch.utils import data
from munch import Munch
from nltk.stem import WordNetLemmatizer

from .task_loaders import load_tasks
from .tokenizer import build_tokenizer


class Dataset(data.Dataset):
    def __init__(self, data_path, args):
        super(Dataset, self).__init__()

        self.data, self.global_data = load_tasks(args, data_path)

        self.list_ids = list(self.data.keys())

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, idx):
        idx = self.list_ids[idx]
        return idx, self.data[idx]


class DataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name')
        self.training = kwargs.pop('training', self.name == 'train')
        self.device = kwargs.pop('device')
        self.tokenizer = kwargs.pop('tokenizer')
        self.make_batch = kwargs.pop('batch_func')
        self.max_sentence_tokens = kwargs.pop('max_sentence_tokens')

        self.lemmatizer = WordNetLemmatizer()

        kwargs['collate_fn'] = self.pad_collate
        super(DataLoader, self).__init__(*args, **kwargs)

    def pad_collate(self, data_li):
        ids, sent_li = zip(*data_li)
        res = self.make_batch(self.tokenizer, sent_li, **self.dataset.global_data,
                              max_sentence_tokens=self.max_sentence_tokens,
                              lemmatize=self.lemmatizer.lemmatize)
        res = Munch(res)
        res.id = ids
        return res


def get_datasets(args, paths):
    datasets = {}
    for k, p in paths.items():
        datasets[k] = Dataset(p, args)
    return datasets


def get_dataloaders(args, datasets, batch_func, tokenizer):
    dataloaders = {}
    for k, dataset in datasets.items():
        dataloader = DataLoader(dataset, name=k,
                                batch_size=args.batch_sizes[k],
                                shuffle=True, num_workers=args.num_workers,
                                batch_func=batch_func,
                                tokenizer=tokenizer, device=args.device,
                                max_sentence_tokens=args.max_sentence_tokens)
        dataloaders[k] = dataloader

    return dataloaders

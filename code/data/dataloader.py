import json

from torch.utils import data
from munch import Munch
from nltk.stem import WordNetLemmatizer

from exp import ex
from .task_loaders import load_tasks
from .tokenizer import build_tokenizer


class Dataset(data.Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()

        self.data, self.global_data, \
            self.task, self.path = load_tasks(data_path)

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
        self.feature_name_map = kwargs.pop('feature_name_map')
        self.concat_group = kwargs.pop('concat_group')
        self.use_vist = kwargs.pop('use_vist')
        self.force_ascii = kwargs.pop('force_ascii')

        self.lemmatizer = WordNetLemmatizer()

        kwargs['collate_fn'] = self.pad_collate
        super(DataLoader, self).__init__(*args, **kwargs)

        self.task = self.dataset.task
        self.path = self.dataset.path

    def pad_collate(self, data_li):
        ids, sent_li = zip(*data_li)
        res = self.make_batch(self.tokenizer, sent_li, **self.dataset.global_data,
                              max_sentence_tokens=self.max_sentence_tokens,
                              lemmatize=self.lemmatizer.lemmatize,
                              feature_name_map=self.feature_name_map,
                              concat_group=self.concat_group,
                              use_vist=self.use_vist,
                              force_ascii=self.force_ascii)
        res = Munch(res)
        res.id = ids
        return res


@ex.capture
def get_datasets(paths, pretrain_paths, use_data):
    datasets = {}
    pretrain_datasets = {}
    for k, p in paths.items():
        if k in use_data:
            print("loading {} data".format(k))
            datasets[k] = Dataset(p)
            if k in pretrain_paths:
                print("loading pretraining {} data".format(k))
                ppath = pretrain_paths[k]
                if isinstance(ppath, list):
                    li = []
                    for p in ppath:
                        li.append(Dataset(p))
                    pretrain_datasets[k] = li
                else:
                    pretrain_datasets[k] = Dataset(pretrain_paths[k])
            else:
                pretrain_datasets[k] = datasets[k]

    return {'target': datasets, 'pretrain': pretrain_datasets}


@ex.capture
def get_dataloaders(datasets, batch_func, tokenizer,
                    batch_sizes, num_workers, device,
                    max_sentence_tokens, feature_name_map, concat_group,
                    use_vist, force_ascii):
    dataloaders = {}
    for dataset_type, type_dataset in datasets.items():
        type_dataloaders = {}
        for k, dataset in type_dataset.items():
            shuffle = True if k == 'train' else False

            def get_dataloader(dset):
                nonlocal shuffle
                nonlocal k
                return DataLoader(dset, name=k,
                                    batch_size=batch_sizes[k],
                                    shuffle=shuffle, num_workers=num_workers,
                                    batch_func=batch_func,
                                    tokenizer=tokenizer, device=device,
                                    max_sentence_tokens=max_sentence_tokens,
                                    feature_name_map=feature_name_map,
                                    concat_group=concat_group,
                                    use_vist=use_vist,
                                    force_ascii=force_ascii)

            if isinstance(dataset, list):
                li = []
                for dset in dataset:
                    li.append(get_dataloader(dset))
                type_dataloaders[k] = li
            else:
                type_dataloaders[k] = get_dataloader(dataset)
        dataloaders[dataset_type] = type_dataloaders
    return dataloaders

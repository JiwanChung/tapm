import json

from torch.utils import data

from transformers import make_batch


class Dataset(data.Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()

        load_data = {
            'ActyNetCap': self.load_actynet_cap,
        }[data_path.parts[-2]]
        self.data = load_data(data_path)
        self.list_ids = list(self.data.keys())

    def load_actynet_cap(self, path):
        with open(path, 'r') as f:
            x = json.load(f)
        data = {}
        for k, v in x.items():
            for i, sent in enumerate(v['sentences']):
                data[f"{k}/{i}"] = sent
        return data

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, idx):
        idx = self.list_ids[idx]
        return self.data[idx]


class DataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.device = kwargs.pop('device')
        self.tokenizer = kwargs.pop('tokenizer')
        kwargs['collate_fn'] = self.pad_collate
        super(DataLoader, self).__init__(*args, **kwargs)

    def pad_collate(self, sent_li):
        res = make_batch(self.tokenizer, sent_li)
        return res


def get_dataloaders(args, paths, tokenizer):
    dataloaders = {}
    for k, p in paths.items():
        dataset = Dataset(p)
        dataloader = DataLoader(dataset, batch_size=args.batch_sizes[k],
                                shuffle=True, num_workers=args.num_workers,
                                tokenizer=tokenizer, device=args.device)
        dataloaders[k] = dataloader

    return dataloaders

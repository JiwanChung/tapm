import torch


'''
def onehot(x, max_size):
    hot = torch.FloatTensor(*x.shape, max_size).to(x.device)
    x = x.unsqueeze(-1)
    hot.zero_()
    hot.scatter_(-1, x, 1)
    return hot.detach()
'''

def find_first(t, value=0):
    # t: 1d tensor
    mask = t == value
    mask = mask.nonzero()
    val = mask.sort()[0]
    if val.nelement() > 0:
        return val[0].item()
    else:
        return t.shape[0]


def unsqueeze_expand(src, tgt):
    if len(src.shape) != len(tgt.shape):
        src = src.view(*src.shape, *[1 for i in range(
            len(tgt.shape) - len(src.shape))])

    if src.shape != tgt.shape:
        src = src.expand_as(tgt)

    return src


def move_device(batch, to=-1):
    for key, tensor in batch.items():
        if torch.is_tensor(tensor):
            batch[key] = tensor.to(to)
        elif tensor is None:
            batch[key] = None
        elif isinstance(tensor, dict):
            batch[key] = tensor
        elif isinstance(tensor, list):
            li = []
            for t in tensor:
                if torch.is_tensor(t):
                    li.append(t.to(to))
                else:
                    li.append(t)
            batch[key] = li
    return batch


def remove_cls_sep(x, tokenizer):
    targets = x[:, 1:]  # remove cls
    dec_input = x.clone().detach()
    dec_input.masked_scatter_(x == tokenizer.sep_id,
                                torch.full_like(dec_input, tokenizer.pad_id))
    dec_input = dec_input[:, :-1]  # remove sep

    return dec_input, targets


def onehot(x, total=1000):
    storage = torch.zeros(*x.shape, total).bool().to(x.device)
    storage.scatter_(dim=-1, index=x.unsqueeze(-1), value=1)

    return storage

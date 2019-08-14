import torch


def onehot(x, max_size):
    hot = torch.FloatTensor(*x.shape, max_size).to(x.device)
    x = x.unsqueeze(-1)
    hot.zero_()
    hot.scatter_(-1, x, 1)
    return hot.detach()


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
        else:
            li = []
            for t in tensor:
                if torch.is_tensor(t):
                    li.append(t.to(to))
                else:
                    li.append(t)
            batch[key] = li
    return batch

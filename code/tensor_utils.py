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


def move_device(*tensors, to=-1):
    li = []
    for tensor in tensors:
        if torch.is_tensor(tensor):
            li.append(tensor.to(to))
        elif tensor is None:
            li.append(None)
        else:
            li.append([t.to(to) for t in tensor])
    return li

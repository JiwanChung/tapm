from torch import optim


def get_optimizer(params, **kwargs):
    return optim.Adam(params, **kwargs)

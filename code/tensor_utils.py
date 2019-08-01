def unsqueeze_expand(src, tgt):
    if len(src.shape) != len(tgt.shape):
        src = src.view(*src.shape, *[1 for i in range(
            len(tgt.shape) - len(src.shape))])

    if src.shape != tgt.shape:
        src = src.expand_as(tgt)

    return src

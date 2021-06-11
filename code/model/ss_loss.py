import torch
import torch.nn.functional as F

from exp import ex


def calc_loss(x1, x2, group_mask=None, pool='mean',
                      skip_idx=0):
    x2 = x2.unsqueeze(0).repeat(x2.shape[0], 1, 1)  # BBC
    group_mask_rolled = group_mask.unsqueeze(0).repeat(group_mask.shape[0], 1)  # BB
    group_mask = group_mask.unsqueeze(0).clone()

    # get permutations
    for i in range(x2.shape[0]):
        x2[i] = torch.roll(x2[i], i, 0)
        group_mask_rolled[i] = torch.roll(group_mask_rolled[i], i, 0)

    s = torch.einsum('bc,ibc->ib', x1, x2)  # BB
    mask = group_mask_rolled & group_mask
    loss = get_loss(s, mask, pool)
    with torch.no_grad():
        acc = get_accuracy(s, mask)

    return loss, acc


def calc_l2_loss(x1, x2, group_mask=None, pool='mean',
                      skip_idx=0):
    s = torch.einsum('bc,bc->b', x1, x2)  # B
    s = s * group_mask.type_as(s)  # masking out
    if group_mask.float().sum() < 1:
        loss = 0
    else:
        loss = ((1 - s) ** 2).sum() / group_mask.float().sum()  # [-1, 1]
    acc = 0

    return loss, acc


def calc_loss_group(x1, x2, group_mask=None, pool='mean',
                            skip_idx=0):
    # BGC, BG
    x2 = x2.unsqueeze(1).repeat(1, x2.shape[1], 1, 1)  # BGGC
    group_mask_rolled = group_mask.unsqueeze(1).repeat(1, group_mask.shape[1], 1)  # BGG
    group_mask = group_mask.unsqueeze(1).clone()

    # get permutations
    for i in range(x2.shape[1]):
        x2[:, i] = torch.roll(x2[:, i], i, 1)
        group_mask_rolled[:, i] = torch.roll(group_mask_rolled[:, i].byte(), i, 1).bool()

    s = torch.einsum('bgc,bigc->big', x1, x2)  # BGG
    mask = group_mask_rolled & group_mask
    if skip_idx != 0:
        if mask.shape[1] > skip_idx:
            mask[:, skip_idx] = 0
    loss = get_batch_loss(s, mask, pool)
    with torch.no_grad():
        acc = get_batch_accuracy(s, mask)

    return loss, acc


def get_accuracy(s, mask):
    # i b
    res = []
    for i in range(s.shape[-1]):
        s_i = s[:, i]
        m_i = mask[:, i]
        s_i = s_i.masked_select(m_i)
        if s_i.sum() != 0:
            idx = s_i.argmax(dim=0)
            res.append(int((idx == 0).any()))
    return sum(res) / len(res)


def get_batch_accuracy(s, mask):
    # BGG
    mask = mask.float()
    s = s * mask
    acc = s.argmax(dim=1) == 0  # BG
    nonzero = mask.sum(dim=1) >= 1  # BG
    acc = acc.masked_select(nonzero)
    acc = acc.float().mean()
    return acc.item()


@ex.capture
def get_loss(s, mask, pool, ss_loss_type):
    # i b (i==0 -> true), i b

    def pool_loss(x):
        if pool == 'mean':
            x = x.mean()
        elif pool == 'max':
            x, _ = x.max(dim=0)
            x = x.mean()
        return x

    s_true = s[0].unsqueeze(0)
    s_false = s[1:]
    f = {
        'ranking': get_ranking_loss,
        'bce': get_bce_loss,
        'l2': get_l2_loss
    }[ss_loss_type.lower()]
    loss = f(s_true, s_false)
    num_element = loss.nelement()
    if num_element == 0:
        return None
    if mask is not None:
        mask = mask[1:]
        res = []
        for i in range(mask.shape[1]):
            mask_i = mask[:, i]
            loss_i = loss[:, i].masked_select(mask_i)
            num_element_i = mask_i.float().sum()
            if num_element_i > 0:
                res.append(pool_loss(loss_i))
        if len(res) > 0:
            loss = torch.stack(res, dim=-1)
            loss = loss.mean()
        else:
            return None
    else:
        loss = pool_loss(loss)
    return loss


@ex.capture
def get_batch_loss(s, mask, pool, ss_loss_type):
    def pool_loss(x):
        if pool == 'mean':
            x = x.mean()
        elif pool == 'max':
            x, _ = x.max(dim=0)
            x = x.mean()
        return x

    s_true = s[:, 0].unsqueeze(1)
    s_false = s[:, 1:]
    f = {
        'ranking': get_ranking_loss,
        'bce': get_bce_loss,
        'l2': get_l2_loss
    }[ss_loss_type.lower()]
    loss = f(s_true, s_false)
    num_element = loss.nelement()
    if num_element == 0:
        return None
    if mask is not None:
        eps = 1e-09
        mask = mask[:, 1:]
        mask = mask.float()
        loss = loss * mask
        loss = loss.sum(dim=-1) / (mask.sum(dim=-1) + eps)
    loss = pool_loss(loss)
    return loss


def get_ranking_loss(s_true, s_false, margin=1):
    loss = torch.max(torch.zeros(1).to(s_true.device), margin + s_false - s_true)  # (i-1) b
    return loss


def get_bce_loss(s_true, s_false):
    tgt = torch.zeros(*s_true.shape).to(device=s_true.device)
    loss_true = F.binary_cross_entropy_with_logits(s_true, tgt, reduction='none')
    tgt = torch.ones(*s_false.shape).to(device=s_false.device)
    loss_false = F.binary_cross_entropy_with_logits(s_false, tgt, reduction='none')
    return loss_true + loss_false


def get_l2_loss(s_true, s_false):
    loss = F.mse_loss(s_true, torch.zeros_like(s_true), reduction='none')
    return loss.expand(-1, s_false.shape[1], -1)

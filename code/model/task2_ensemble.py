import torch
from torch import nn
import pickle

from utils import mean
from .task2_baseline import Task2Baseline2


class Task2Ensemble(Task2Baseline2):
    def __init__(self, args, transformer, tokenizer):
        super(Task2Ensemble, self).__init__(args, transformer, tokenizer)

        del self.net
        del self.reduce_dim

        self.ensemble_inputs = self.load_ensemble_inputs(args)
        for k in self.ensemble_inputs.keys():
            module = nn.Linear(1, 1, bias=False)
            noise = (torch.randn(1) * 1e-1).item()
            module.weight.data.fill_(1 + noise)
            setattr(self, f"linear_{k}", module)

    def load_ensemble_inputs(self, args):
        paths = args.ensemble_path
        res = {}
        for k, path in paths.items():
            with open(path, 'rb') as f:
                res[k] = pickle.load(f)
        return res

    def get_ensembles(self, batch):
        vids = [vid[0] for vid in batch.vid]
        res = []
        for vid in vids:
            x = {k: data[vid] if vid in data else None for k, data in self.ensemble_inputs.items()}
            if None in list(x.values()):
                x = None
            if x is not None and x['text'] is not None:
                if len(set([v.shape for v in x.values()])) > 1:
                    print(f'ensemble shape wrong {vid}')
                    x = None
            res.append(x)
        return res

    def ensemble(self, hypo, device=0):
        outputs = []
        for h in hypo:
            res = []
            if h is not None:
                for k, v in h.items():
                    if v is not None:
                        v = torch.from_numpy(v).to(device).float()
                        v = v.unsqueeze(-1)
                        v = getattr(self, f"linear_{k}")(v)
                        v = v.squeeze(-1)
                        res.append(v)
            res = mean(res)
            outputs.append(res)
        return outputs

    def forward(self, batch, **kwargs):
        blank_ids = batch.blank_ids
        target = self.make_target(batch)

        hypo = self.get_ensembles(batch)
        hypo = self.ensemble(hypo, device=blank_ids.device)

        hypo, target = zip(*[(h, t) for h, t in zip(hypo, target) if h is not None])
        for i in range(len(hypo)):
            if hypo[i].shape[0] != target[i].shape[0]:
                print('blank num size wrong')
                import ipdb; ipdb.set_trace()  # XXX DEBUG

        reg_loss = self.get_loss(hypo, target)

        with torch.no_grad():
            hypo_cont = [h.clone().detach() if torch.is_tensor(h) else h for h in hypo]
            hypo = self.get_prob(hypo)
            acc, same_acc, diff_acc = self.get_acc(hypo, target)
            relation = self.get_relation(hypo, target)
            stats = {'blank_loss': reg_loss.item(),
                    'blank_num': blank_ids.float().sum(dim=-1).mean().item(),
                    'blank_acc': None if acc is None else acc.float().mean().item(),
                    'blank_same_acc': None if same_acc is None else same_acc.float().mean().item(),
                    'blank_diff_acc': None if diff_acc is None else diff_acc.float().mean().item()}

        return None, batch.targets, reg_loss, stats, \
            {'text': relation, 'hypo': [h.detach().cpu().numpy() if torch.is_tensor(h) else h for h in hypo_cont]}

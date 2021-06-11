from collections import defaultdict
from pathlib import Path
import pickle

from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from einops import rearrange

from exp import ex

from tensor_utils import move_device
from run_transformer import transformer_embed


@ex.capture
def _distance(model, loss_fn, optimizer, tokenizer, dataloaders, logger,
                metrics, batch_sizes, device, reg_coeff, eval_metric, model_name,
          root,
                epoch=-1, subset=None, key=None, eval_generate=True,
                eval_set=False, sample=False, concat_group=False,
                use_vist=False):
    if key is None:
        key = 'val'
    epoch_stats = defaultdict(float)
    model.eval()
    dataloader = dataloaders[key]
    if subset is not None:
        subset = (len(dataloader) * subset) // batch_sizes[key]
        subset = max(1, subset)
        total_length = subset

    path = root / 'data' / 'vis'
    path.mkdir(exist_ok=True)

    print("starting extraction")
    dists = 0
    counts = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=subset):
            batch = move_device(batch,
                                to=device)
            vid, text = extract_reps(model, batch)
            dist, count = calc_dist(vid, text)
            dists += dist
            counts += count

    print(f"Mean similarity: {dists / counts} (count: {counts})")
    print("extraction done!")


def extract_reps(model, batch):
    features, features_merged, keywords, G = model.prepare_group(batch)
    hypo = batch.sentences
    hypo = rearrange(hypo.contiguous(), 'b g l -> (b g) l')
    h, inputs = transformer_embed(model.net.transformer, hypo,
                                    skip_ids=[model.tokenizer.pad_id, model.tokenizer.sep_id],
                                    infer=False)
    return features_merged, h


def calc_dist(vid, text):
    vid = torch.cat(list(vid.values()), dim=1)
    vid = vid.mean(dim=1)
    text = text.mean(dim=1)
    res = F.cosine_similarity(vid, text, dim=1)
    # res = torch.einsum('bc,bc->b', vid, text)
    return res.sum().item(), res.shape[0]

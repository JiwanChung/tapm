from collections import defaultdict
from pathlib import Path
import pickle

from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import torch
from einops import rearrange
# from tsnecuda import TSNE

from exp import ex

from tensor_utils import move_device
from run_transformer import transformer_embed


@ex.capture
def _tsne(model, loss_fn, optimizer, tokenizer, dataloaders, logger,
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
    tsne = []
    tsne_group = []
    tsne_large = []

    print("starting extraction")
    with torch.no_grad():
        for batch in tqdm(dataloader, total=subset):
            batch = move_device(batch,
                                to=device)
            vid, text = extract_reps(model, batch)
            res, group, large_pool = get_tsne(vid, text, max_samples=None)
            tsne = [*tsne, *res]
            tsne_group = [*tsne_group, *group]
            tsne_large.append(large_pool)

    with open(path / f"{model_name}_single.pkl", 'wb') as f:
        pickle.dump(tsne, f)
    with open(path / f"{model_name}_group.pkl", 'wb') as f:
        pickle.dump(tsne_group, f)
    with open(path / f"{model_name}_large.pkl", 'wb') as f:
        pickle.dump(tsne_large, f)
    print("extraction done!")


def extract_reps(model, batch):
    features, features_merged, keywords, G = model.prepare_group(batch)
    hypo = batch.sentences
    hypo = rearrange(hypo.contiguous(), 'b g l -> (b g) l')
    h, inputs = transformer_embed(model.net.transformer, hypo,
                                    skip_ids=[model.tokenizer.pad_id, model.tokenizer.sep_id],
                                    infer=False)
    return features_merged, h


def get_tsne(vid, text, max_samples=None):
    # (b g) l c
    vid = torch.cat(list(vid.values()), dim=1)
    vid_length = vid.shape[1]
    data = torch.cat((vid, text), dim=1)
    pca = PCA(n_components=10)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    res = []
    group_pool = []
    large_pools = []
    if max_samples is None:
        max_samples = data.shape[0] // 5
    # assert max_samples <= data.shape[0], 'max_sample too large'
    max_samples = min(max_samples, data.shape[0] // 5)
    for i in tqdm(range(max_samples)):
        group = []
        '''
        for j in tqdm(range(5)):
            sample = data[i * 5 + j]
            sample = sample.cpu()
            sample = sample.numpy()
            sample = pca.fit_transform(sample)
            sample = tsne.fit_transform(sample)
            # tsne = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(sample)
            group.append((sample[:vid_length], sample[vid_length:]))
        res.append(group)
        '''
        group_data = data[i * 5: i * 5 + 5]
        g_vid = group_data[:, :vid_length].mean(dim=1)  # 5C
        g_text = group_data[:, vid_length:].mean(dim=1)
        g_data = torch.cat((g_vid, g_text), dim=0).cpu().numpy()
        g_data = pca.fit_transform(g_data)
        g_data = tsne.fit_transform(g_data)
        length = g_vid.shape[0]
        group_pool.append([g_data[:length], g_data[length:]])

    '''
    sample_size = 20
    for i in tqdm(range(data.shape[0] // sample_size)):
        l_vid = data[sample_size * i : sample_size * (i+1), :vid_length].mean(dim=1)  # 5C
        l_text = data[sample_size * i : sample_size * (i+1), vid_length:].mean(dim=1)
        l_data = torch.cat((l_vid, l_text), dim=0).cpu().numpy()
        l_data = pca.fit_transform(l_data)
        l_data = tsne.fit_transform(l_data)
        length = l_vid.shape[0]
        large_pool = [l_data[:length], l_data[length:]]
        large_pools.append(large_pool)
    '''
    return res, group_pool, large_pools


@ex.capture
def _silhouette(model, loss_fn, optimizer, tokenizer, dataloaders, logger,
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

    print("starting extraction")
    vids = []
    texts = []
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=subset):
            batch = move_device(batch,
                                to=device)
            vid, text = extract_reps(model, batch)
            vid = torch.cat(list(vid.values()), dim=1)
            vid, text = get_random_feat(vid), get_random_feat(text)
            vids.append(vid)
            texts.append(text)
            count += vid.shape[0]
            if count >= 100:
                break
        vids = torch.cat(vids, dim=0)[:100]
        texts = torch.cat(texts, dim=0)[:100]
        data = torch.cat((vids, texts), dim=0).cpu().numpy()
        idx = torch.cat((torch.zeros((100,)), torch.ones((100,))), dim=0).cpu().numpy()
        print(data.shape, idx.shape)
        score = silhouette_score(data, idx, metric='cosine')
    print(f"silhouette_score: {score}")


def get_random_feat(feat):
    # (BG) LC
    feat = rearrange(feat, '(b g) l c -> b (g l) c', g=5)
    idx = torch.randint(0, feat.shape[1], (feat.shape[0],)).to(feat.device)
    feat = feat.gather(1, idx.view(-1, 1, 1).expand(feat.shape[0], 1, feat.shape[-1]))
    return feat.squeeze(1)

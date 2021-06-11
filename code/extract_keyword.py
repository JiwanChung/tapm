import json
import torch
import numpy as np
from tqdm import tqdm

from collections import defaultdict

from exp import ex
from tensor_utils import move_device
from path import get_dirname_from_args, get_keyword_path
from data.batcher import decode_tensor, remove_pad


def extract_keyword(model, tokenizer, dataloader,
                    extraction_threshold, device,
                    extraction_min_words):
    model.eval()
    threshold = extraction_threshold  # prediction loss for mask_model
    res = {}
    ratios = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            data_ids = batch['id']
            batch = move_device(batch, to=device)
            B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
            targets = batch['targets']
            loss, scores, ids = model(batch)

            for i in range(B):
                min_words = min(extraction_min_words, ids[i].shape[0])
                keywords = [(key, score)
                            for j, (key, score) in enumerate(zip(ids[i], scores[i]))
                            if score < threshold or j < min_words]
                keywords, score = zip(*keywords)
                keywords, score = torch.Tensor(list(keywords)).to(ids[i].device), \
                    torch.Tensor(list(score)).to(ids[i].device).cpu().numpy().tolist()
                target_len = remove_pad(targets[i], tokenizer.pad_id).shape[0] - 2  # remove cls, sep
                keywords_len = remove_pad(keywords, tokenizer.pad_id).shape[0]
                keywords = decode_tensor(tokenizer, keywords, split_tokens=True)
                target = decode_tensor(tokenizer, targets[i])
                ratios.append(keywords_len / target_len)
                res[data_ids[int(i/5)]] = {'keyword': keywords, 'score': score}

    ratios = np.array(ratios)
    model.train()

    return res, ratios


@ex.capture
def extract_and_save(key, model, tokenizer, dataloaders, data_path):
    path = None
    if key in data_path:
        print(f"extracting keyword for {key}")

        res, ratios = extract_keyword(model, tokenizer, dataloaders[key])
        ratio_percentiles = [10, 20, 50, 80, 90]
        ratio_percentiles = {i: np.percentile(ratios, i) for i in ratio_percentiles}
        print(f"keyword ratio percentiles: {ratio_percentiles}")
        path = get_keyword_path(data_path, key)
        print(f"saving keyword to {path}")
        with open(path, 'w') as f:
            json.dump(res, f, indent=4)

    return path


def extract_and_save_all(model, tokenizer, dataloaders):
    extract_and_save('train', model, tokenizer, dataloaders)
    extract_and_save('val', model, tokenizer, dataloaders)
    extract_and_save('test', model, tokenizer, dataloaders)

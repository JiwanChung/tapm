import json
import torch
import numpy as np

from collections import defaultdict

from tensor_utils import move_device
from utils import get_dirname_from_args
from data.batcher import decode_tensor, remove_pad


def extract_keyword(args, model, tokenizer, dataloader):
    model.eval()
    threshold = args.extraction_threshold  # prediction probability for mask_model
    res = {}
    ratios = []
    with torch.no_grad():
        for batch in dataloader:
            data_ids = batch[0]
            batch = batch[1:]
            batch = move_device(*batch, to=args.device)
            B = batch[0].shape[0] if torch.is_tensor(batch[0]) else len(batch[0])
            targets = batch[-1]
            loss, scores, ids = model(*batch)

            for i in range(B):
                keywords = [(key, score)
                            for j, (key, score) in enumerate(zip(ids[i], scores[i]))
                            if score < threshold or j < args.extraction_min_words]
                keywords, score = zip(*keywords)
                keywords, score = torch.Tensor(list(keywords)).to(ids[i].device), \
                    torch.Tensor(list(score)).to(ids[i].device).cpu().numpy().tolist()
                target_len = remove_pad(targets[i], tokenizer.pad_id).shape[0] - 2  # remove cls, sep
                keywords_len = remove_pad(keywords, tokenizer.pad_id).shape[0]
                keywords = decode_tensor(tokenizer, keywords, split_tokens=True)
                target = decode_tensor(tokenizer, targets[i])
                ratios.append(keywords_len / target_len)
                res[data_ids[i]] = {'keyword': keywords, 'score': score}

    ratios = np.array(ratios)
    model.train()

    return res, ratios


def extract_and_save(key, args, model, tokenizer, dataloaders):
    path = None
    if key in args.data_path:
        print(f"extracting keyword for {key}")
        path = args.data_path[key].parent / \
            f'keywords_{get_dirname_from_args(args)}'
        path.mkdir(parents=True, exist_ok=True)
        path = path / args.data_path[key].name
        res, ratios = extract_keyword(args, model, tokenizer, dataloaders[key])
        ratio_percentiles = [10, 20, 50, 80, 90]
        ratio_percentiles = {i: np.percentile(ratios, i) for i in ratio_percentiles}
        print(f"keyword ratio percentiles: {ratio_percentiles}")
        print("saving keyword")
        with open(path, 'w') as f:
            json.dump(res, f, indent=4)

    return path


def extract_and_save_all(args, model, tokenizer, dataloaders):
    extract_and_save('train', args, model, tokenizer, dataloaders)
    extract_and_save('val', args, model, tokenizer, dataloaders)
    extract_and_save('test', args, model, tokenizer, dataloaders)

import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import make_keyword_batch, remove_pad, pad
from loss.base import Loss

from .transformer_model import TransformerModel


class InsertKeywordLM(TransformerModel):
    transformer_name = 'bert'

    @classmethod
    def get_args(cls, args):
        args = super().get_args(args)
        args.eval_generate = True
        return args

    def __init__(self, args, transformer, tokenizer):
        super(InsertKeywordLM, self).__init__()

        self.V = len(tokenizer)
        self.dim = 768
        self.num_layers = 2
        self.use_keyword = args.use_keyword

        self.tokenizer = tokenizer

        self.wte = nn.Embedding(self.V, self.dim)
        self.preprocess = PreProcessor(self.dim)
        self.decoder = nn.GRU(self.dim, self.dim, self.num_layers,
                              bidirectional=False,
                              batch_first=True)

    @staticmethod
    def make_batch(*args, **kwargs):
        return make_keyword_batch(*args, **kwargs, concat=False)

    def generate(self, insertions, keyword_idx, target):
        # B(n+1)C

        # teacher forcing ver.
        init_idx = torch.cat((torch.LongTensor(0).to(keyword_idx.device),
                              keyword_idx[1:] + 1), dim=0)
        end_idx = torch.cat((keyword_idx[1:] + 1,
                             torch.LongTensor(keyword_idx.shape[0]).to(keyword_idx.device)),
                            dim=0)
        gts = []
        insertion_target = []
        for i in range(init_idx.shape[0]):
            gt = target[init_idx[i]: end_idx[i]]
            # TODO: add_specials
            gt = torch.cat((
                torch.LongTensor([self.tokenizer.insert_sos_id]).to(insertions.device),
                gt,
                torch.LongTensor([self.tokenizer.insert_eos_id]).to(insertions.device),
            ), dim=0)
            gts.append(gt[1:])
            insertion_target.append(gt[:-1])
        gts, _ = pad(gts, self.tokenizer.pad_id)  # NL
        gts = gts.to(insertions.device)
        insertion_target, _ = pad(insertion_target, self.tokenizer.pad_id)  # NL
        insertion_target = insertion_target(insertions.device)
        gts = self.wte(gts)  # NLC
        o, h = self.decoder(gts, insertions)

        return o, insertion_target, torch.stack(init_idx, end_idx, dim=-1)

    def postprocess(self, insertions, keyword, keyword_idx, insertion_ids, shape):
        res = torch.zeros(*shape, insertions.shape[-1]).float().to(insertions.device)
        for i in range(insertions.shape[0]):
            idx = insertion_ids[i]
            res[idx[0]: idx[1]] = insertions[i]
        for i in range(keyword.shape[0]):
            idx = keyword_idx[i]
            res[idx] = keyword[i]
        return res

    def insertion_loss(self, insertions, insertion_target):
        logit = self.out(insertions)
        loss = Loss(self.tokenizer.pad_id)
        return loss(insertions, insertion_target)

    def out(self, x):
        return torch.matmul(x, self.wte.weight.t())

    def forward(self, batch, **kwargs):
        keywords = batch.keywords
        ordered_keywords = batch.ordered_keywords
        sentences = batch.sentences
        targets = batch.targets
        keyword_ids = batch.keyword_ids

        reorder_losses = []
        insertion_losses = []
        logits = []
        target_list = []
        for sentence, target, keyword, ordered_keyword, keyword_idx in \
                zip(sentences, targets, keywords, ordered_keywords, keyword_ids):
            keyword = remove_pad(keyword, self.tokenizer.pad_id)
            ordered_keyword = remove_pad(ordered_keyword, self.tokenizer.pad_id)
            target = remove_pad(target, self.tokenizer.pad_id)
            keyword = ordered_keyword  # debug
            reorder_loss = 0
            keyword = self.wte(keyword)
            '''
            keyword = self.reorder(keyword)
            reorder_loss = self.reorder_loss(keyword, ordered_keyword)
            '''
            insertions = self.preprocess(keyword.unsqueeze(0)).squeeze(0)  # n+2 -> n+1
            insertions, insertion_target, insertion_ids = self.generate(insertions, keyword_idx, target)  # (num insertions)LC
            insertion_loss = self.insertion_loss(insertions, insertion_target)
            insertions = self.postprocess(insertions, keyword, keyword_idx, insertion_ids, target.shape - 2)  # remove cls, sep
            logit = self.out(insertions)

            reorder_losses.append(reorder_loss)
            insertion_losses.append(insertion_loss)
            logits.append(logit)
            target_list.append(target)

        reorder_loss = sum(reorder_losses) / len(reorder_loss)
        insertion_loss = sum(insertion_losses) / len(insertion_loss)

        return logits, target_list, reorder_loss + insertion_loss, {}, None


class PreProcessor(nn.Module):
    def __init__(self, dim):
        super(PreProcessor, self).__init__()

        self.dim = dim

        self.blocks = nn.Sequential(
            Block(self.dim),
            Block(self.dim),
        )
        self.reduce_cnn = nn.Conv1d(self.dim, self.dim,
                                    kernel_size=2,
                                    padding=0)

    def forward(self, keyword):
        keyword = self.blocks(keyword)
        keyword = keyword.transpose(1, 2)
        keyword = self.reduce_cnn(keyword)
        keyword = keyword.transpose(1, 2)
        return keyword


class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()

        self.dim = dim

        self.layer_norm = nn.LayerNorm(self.dim)
        self.conv = nn.Conv1d(self.dim, self.dim,
                              kernel_size=3,
                              padding=1)

    def forward(self, x):
        x_orig = x
        x = F.relu(x)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x + x_orig

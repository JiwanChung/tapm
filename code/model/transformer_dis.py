from collections import OrderedDict
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F

from .hybrid_dis import HybridDis
from debug_utils import timeit


class TransformerDis(HybridDis):
    transformer_name = 'gpt2'  # assign transformer_name = 'bert' to use BPE

    def __init__(self, args, transformer, tokenizer):
        super(TransformerDis, self).__init__(args, transformer, tokenizer)

        self.dropout_ratio = args.get('dropout', 0.5)

        self.net = transformer
        self.net.train()
        self.gpt_dim = self.net.transformer.config.n_embd

        for feature in self.feature_names:
            setattr(self, feature, FeatureEncoder(getattr(self, f"{feature}_dim"), self.gpt_dim))

        self.reduce_cat = nn.Linear(self.gpt_dim + self.keyword_num, self.gpt_dim)
        self.reduce_c = nn.Linear(self.gpt_dim, self.dim)

        self.dropout = nn.Dropout(self.dropout_ratio)

    def run_transformer(self, hypo, features, keyword):
        h, past, head_mask = transformer_embed(self.net.transformer, hypo)
        h = torch.cat((h, # features.unsqueeze(1).expand(-1, h.shape[1], -1),
                       keyword.unsqueeze(1).expand(-1, h.shape[1], -1)), dim=-1)
        h = self.reduce_cat(h)
        cls_embd = self.net.transformer.wte(torch.LongTensor([self.tokenizer.cls_id]).to(h.device))
        sep_embd = self.net.transformer.wte(torch.LongTensor([self.tokenizer.sep_id]).to(h.device))
        B, L, C = h.shape
        cls_embd = cls_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        sep_embd = sep_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        context = torch.cat((cls_embd, *chain(*[(feature, sep_embd) for feature in features.values()])), dim=1)
        h = torch.cat((context, h), dim=1)

        o = transformer_run_cells(self.net.transformer, h, past=past, head_mask=head_mask)[0]
        o = o[:, context.shape[1]:]
        o = self.dropout(o)
        c = o.mean(dim=1)
        c = self.reduce_c(c)
        logits = self.net.lm_head(o)
        return logits, c

    def run_token(self, features, hypo, h, c, keyword):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        for feature in self.feature_names:
            features[feature] = getattr(self, feature)(features[feature], None)
        logits, h = self.generate_token(hypo, features, c, h, keyword)
        return h, c, logits

    def run_train(self, hypo, features, keyword):
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        for feature in self.feature_names:
            features[feature] = getattr(self, feature)(features[feature], None)
        return self.run_transformer(hypo, features, keyword)

    def generate_token(self, hypo, features, c, h, keyword):
        logits, h = self.run_transformer(hypo, features, keyword)
        return logits, h

    def run_video(self, features, c, v, L, sentences=None, sampler=None,
                  keyword=None, reduce_hypo=True):
        video = features['video']
        B = video.shape[0]
        empty = torch.full((B, self.vocab_size), float('-inf')).to(video.device)
        sent = []
        eos_flags = torch.LongTensor([0] * B).byte().to(video.device)
        if c is None:
            c = self.rnn.init_c(B, self.context_dim, device=video.device) if hasattr(self, 'rnn') else None
        s0 = sentences[:, v, 0] if sentences is not None \
            else torch.Tensor([self.tokenizer.cls_id]).long().to(video.device).expand(B)
        s = s0
        hypo = s0.unsqueeze(-1)

        if sentences is not None:  # training
            sent, h = self.run_train(sentences[:, v], features, keyword)
        else:
            for w in range(L):
                if eos_flags.all():
                    logits = empty.clone()
                else:
                    h = None
                    h, c, logits = self.run_token(features, hypo, h, c, keyword=keyword)
                    s, probs = sampler(logits)
                    eos_flags = eos_flags | (logits[:, -1].argmax(dim=-1) == self.tokenizer.pad_id)
                hypo = torch.cat((hypo, s.unsqueeze(-1)), dim=1)
                sent.append(logits)

            hypo = hypo[:, 1:]
            if reduce_hypo:
                hypo = hypo[probs.argmax(dim=-1)]
        c = self.context_encoder(h)
        if not self.use_context:
            c = torch.full_like(c.detach(), 0)
            c.requires_grad_(False)
        return c, sent, hypo


def transformer_embed(self, input_ids, position_ids=None, token_type_ids=None, past=None, head_mask=None):
    if past is None:
        past_length = 0
        past = [None] * len(self.h)
    else:
        past_length = past[0][0].size(-2)
    if position_ids is None:
        position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * self.config.n_layer

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_ids.size(-1))
    position_ids = position_ids.view(-1, position_ids.size(-1))

    inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        token_type_embeds = self.wte(token_type_ids)
    else:
        token_type_embeds = 0
    hidden_states = inputs_embeds + position_embeds + token_type_embeds
    return hidden_states, past, head_mask


def transformer_run_cells(self, hidden_states, position_ids=None,
                          token_type_ids=None, past=None, head_mask=None):
    hidden_states = self.drop(hidden_states)

    input_shape = hidden_states.shape[:-1]
    output_shape = input_shape + (hidden_states.size(-1),)

    presents = ()
    all_attentions = []
    all_hidden_states = ()
    for i, (block, layer_past) in enumerate(zip(self.h, past)):
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

        outputs = block(hidden_states, layer_past, head_mask[i])
        hidden_states, present = outputs[:2]
        presents = presents + (present,)

        if self.output_attentions:
            all_attentions.append(outputs[2])

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(*output_shape)
    # Add last hidden state
    if self.output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states, presents)
    if self.output_hidden_states:
        outputs = outputs + (all_hidden_states,)
    if self.output_attentions:
        # let the number of heads free (-1) so we can extract attention even after head pruning
        attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
        all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
        outputs = outputs + (all_attentions,)
    return outputs  # last hidden state, presents, (all hidden_states), (attentions)


class FeatureEncoder(nn.Module):
    def __init__(self, video_dim, dim):
        super(FeatureEncoder, self).__init__()

        self.linear = nn.Linear(video_dim, dim)

    def forward(self, feature, h):
        # BLC
        feature = self.linear(feature)
        return feature

import torch
from torch import nn
import torch.nn.functional as F

from transformers import remove_pad, pad


class Decoder(nn.Module):
    def __init__(self, args, transformer, tokenizer):
        super(Decoder, self).__init__()

        self.transformer = transformer
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad_id
        self.type_ids = (self.tokenizer.convert_tokens_to_ids(self.tokenizer.type_a),
                         self.tokenizer.convert_tokens_to_ids(self.tokenizer.type_b))
        transformer_embed_dim = self.transformer.wte.weight.shape[1]
        self.out = nn.Linear(transformer_embed_dim,
                             tokenizer.vocab_size)

    def forward(self, sentence, keywords, keyword_lengths, scores):
        inputs, lengths = self.concat_input(keywords, sentence)
        tids = self.make_token_type_ids(inputs, keyword_lengths, self.type_ids)
        # mask = self.make_attention_mask(inputs, self.pad_id)
        h, past, head_mask = transformer_embed(self.transformer, inputs,
                              token_type_ids=tids)
        h = self.multiply_score(h, scores)
        logits, _ = transformer_run_cells(self.transformer, h, inputs,
                                          token_type_ids=tids,
                                          past=past, head_mask=head_mask)
        logits = self.cut_output(keyword_lengths, lengths, logits)
        logits = self.out(logits)
        return logits  # B (max_len + [SEP]) V

    @staticmethod
    def make_token_type_ids(inputs, keyword_lengths, type_ids):
        ids = inputs.clone().detach()
        ids.fill_(type_ids[1])
        B = ids.shape[0]
        for i in range(B):
            ids[i, 1: keyword_lengths[i] + 1] = type_ids[0]
        return ids

    @staticmethod
    def multiply_score(h, scores):
        B = h.shape[0]
        for i in range(B):
            h[i, 1: scores[i].shape[0] + 1] *= scores[i].unsqueeze(-1)

        return h

    def concat_input(self, keywords, sentence):
        # BL
        B = keywords.shape[0]
        res = []
        for i in range(B):
            res.append(torch.Tensor([self.tokenizer.cls_id,
                        *remove_pad(keywords[i], self.pad_id).cpu().numpy(),
                        self.tokenizer.sep_id,
                        *remove_pad(sentence[i], self.pad_id).cpu().numpy(),
                        self.tokenizer.sep_id]).long())
        res, lengths = pad(res, self.pad_id)

        return res, lengths

    def cut_output(self, keyword_lengths, lengths, x):
        # CLS keyword SEP sent SEP pad -> sent SEP (the target is shifted right)
        B = keyword_lengths.shape[0]
        res = []
        for i in range(B):
            res.append(x[i, keyword_lengths[i] + 1: lengths[i] - 1])
        # pad zeros
        max_len = max([i.shape[0] for i in res])
        shape = list(x.shape)
        shape[1] = max_len
        x = torch.zeros(*shape, dtype=x.dtype)
        for i in range(B):
            x[i, :res[i].shape[0]] = res[i]
        return x


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


def transformer_run_cells(self, hidden_states, input_ids, position_ids=None,
                          token_type_ids=None, past=None, head_mask=None):
    hidden_states = self.drop(hidden_states)

    input_shape = input_ids.size()
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

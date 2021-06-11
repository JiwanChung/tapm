import torch
import torch.nn.functional as F

from transformers import (
    XLNetModel, GPT2Model, XLMModel, BertModel  # , TransfoXLModel
)
# from transformers.modeling_transfo_xl_utilities import sample_logits
from model.lstm import RNNModel


# temporarily deprecating transfo_xl due to source code refactoring


def transformer_embed(self, *args, **kwargs):
    if isinstance(self, XLNetModel):
        return xlnet_embed(self, *args, **kwargs)
    elif isinstance(self, GPT2Model):
        return gpt2_embed(self, *args, **kwargs)
    # elif isinstance(self, TransfoXLModel):
    #    return transfo_xl_embed(self, *args, **kwargs)
    elif isinstance(self, XLMModel):
        return xlm_embed(self, *args, **kwargs)
    elif isinstance(self, RNNModel):
        return base_embed(self, *args, **kwargs)
    elif isinstance(self, BertModel):
        return bert_embed(self, *args, **kwargs)
    else:
        return base_embed(self, *args, **kwargs)


def transformer_run_cells(self, *args, **kwargs):
    if isinstance(self, XLNetModel):
        return xlnet_run_cells(self, *args, **kwargs)
    elif isinstance(self, GPT2Model):
        return gpt2_run_cells(self, *args, **kwargs)
    # elif isinstance(self, TransfoXLModel):
    #    return transfo_xl_run_cells(self, *args, **kwargs)
    elif isinstance(self, RNNModel):
        return lstm_run_cells(self, *args, **kwargs)
    elif isinstance(self, XLMModel):
        return xlm_run_cells(self, *args, **kwargs)
    elif isinstance(self, BertModel):
        return bert_run_cells(self, *args, **kwargs)
    else:
        return base_run_cells(self, *args, **kwargs)


def transformer_lm_head(self, *args, **kwargs):
    # if isinstance(self.transformer, TransfoXLModel):
    #    return transfo_xl_lm_head(self, *args, **kwargs)
    if isinstance(self.transformer, XLMModel):
        return xlm_lm_head(self, *args, **kwargs)
    else:
        return base_lm_head(self, *args, **kwargs)


def base_embed(self, input_ids, **kwargs):
    return self.wte(input_ids), {}


def base_run_cells(self, context, hidden_states):
    hidden_states = torch.cat((context, hidden_states), dim=1)
    outputs = self.forward(inputs_embeds=hidden_states)
    o = outputs[0]
    context_embedded = o[:, :context.shape[1]]
    o = o[:, context.shape[1]:]

    return o, context_embedded


def base_lm_head(self, o):
    return self.lm_head(o)


def bert_embed(self, input_ids, **kwargs):
    return self.embeddings(input_ids), {}


def bert_run_cells(self, context, hidden_states, hypo, pad_id=0, **kwargs):
    hidden_states = torch.cat((context, hidden_states), dim=1)
    attention_mask = (hypo != pad_id).long()
    context_mask = torch.ones(context.shape[:-1], device=attention_mask.device).long()
    attention_mask = torch.cat((context_mask, attention_mask), dim=-1)
    outputs = self.forward(inputs_embeds=hidden_states, attention_mask=attention_mask)
    o = outputs[0]
    context_embedded = o[:, :context.shape[1]]
    o = o[:, context.shape[1]:]

    return o, context_embedded


def lstm_run_cells(self, context, hidden_states, **kwargs):
    hidden_states = torch.cat((context, hidden_states), dim=1)
    o = self.run_cell(hidden_states)
    context_embedded = o[:, :context.shape[1]]
    o = o[:, context.shape[1]:]

    return o, context_embedded


'''
def transfo_xl_embed(self, input_ids, **kwargs):
    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = [None] * self.config.n_layer

    input_ids = input_ids.view(-1, input_ids.size(-1))

    inputs_embeds = self.word_emb(input_ids)
    return inputs_embeds, {}


def transfo_xl_lm_head(self, o):
    B, L = o.shape[:2]
    if self.sample_softmax > 0 and self.training:
        assert self.config.tie_weight
        logit = sample_logits(self.transformer.word_emb, self.out_layer.bias,
                              None, o, self.sampler)
        softmax_output = -F.log_softmax(logit, -1)[:, :, 0]
    else:
        softmax_output = self.crit(o.contiguous().view(-1, o.shape[-1]), None)
        softmax_output = softmax_output.view(B, L, -1)
    softmax_output.if_log_softmax = True
    return softmax_output
'''


def xlm_embed(self, input_ids, **kwargs):
    return self.wte(input_ids), {'input_ids': input_ids}


def xlm_lm_head(self, o):
    o = self.pred_layer(o)
    o = o[0]
    if self.pred_layer.asm:
        o.if_log_softmax = True
    return o


def xlm_run_cells(self, context, hidden_states, input_ids=None, **kwargs):
    hidden_states = torch.cat((context, hidden_states), dim=1)
    B, CL = context.shape[:2]
    # attention_mask = input_ids != self.pad_index
    lengths = (input_ids != self.pad_index).long().sum(dim=-1) + CL
    outputs = self.forward(inputs_embeds=hidden_states, lengths=lengths)
    o = outputs[0]
    context_embedded = o[:, :context.shape[1]]
    o = o[:, context.shape[1]:]

    return o, context_embedded


def transfo_xl_run_cells(self, context, hidden_states, position_ids=None,
                          token_type_ids=None, past=None, head_mask=None,
                         **kwargs):

    hidden_states = torch.cat((context, hidden_states), dim=1)
    inputs = {
        'inputs_embeds': hidden_states,
        # 'head_mask': head_mask
    }
    outputs = self.forward(**inputs)

    # last hidden state, presents, (all hidden_states), (attentions)
    o = outputs[0]
    context_embedded = o[:, :context.shape[1]]
    o = o[:, context.shape[1]:]

    return o, context_embedded


def xlnet_embed(self, input_ids, skip_ids=[], infer=False, **kwargs):
    device = input_ids.device
    B = input_ids.shape[0]
    if infer:
        # adding predict token for xlnet
        pred_token = torch.zeros((1, 1), dtype=torch.long, device=device).repeat(B, 1)
        input_ids = torch.cat((input_ids, pred_token), dim=1)
        inputs = {}
    else:
        # BL
        mask = skip_ids[0] != input_ids
        for i in range(1, len(skip_ids)):
            mask = mask & (skip_ids[i] != input_ids)
        lengths = mask.cumsum(dim=-1).max(dim=-1)[0]  # get lengths
        pred_ids = []
        for b in range(B):
            length = 1 if lengths[b] == 0 else lengths[b]
            pred_ids.append(torch.randint(0, length, (1,)).to(input_ids.device))
        pred_ids = torch.cat(pred_ids, dim=0)
        pred_token = torch.zeros((1, 1), dtype=torch.long, device=device).repeat(B, 1)
        input_ids.scatter_(-1, pred_ids.unsqueeze(-1), pred_token)
        inputs = {'pred_ids': pred_ids}
    return self.word_embedding(input_ids), inputs


def xlnet_run_cells(self, context, h, pred_ids=None, **kwargs):
    h = torch.cat((context, h), dim=1)
    B, L = h.shape[:2]
    device = h.device
    perm_mask = torch.zeros((1, L, L), dtype=torch.float, device=device)
    perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
    # self.attn_type = 'uni'  # single directional
    target_mapping = torch.ones((B, L, L), dtype=torch.float, device=device)
    # target_mapping[0, 0, -1] = 1.0  # predict last token
    if pred_ids is None:
        pred_ids = torch.full((B,), L - 1, dtype=torch.long).to(device)
    pred_ids_c = pred_ids + context.shape[1]
    target_mapping.scatter_(-1, pred_ids_c.view(-1, 1, 1).repeat(1, L, 1), 1)
    token_type_ids = torch.zeros((B, L), dtype=torch.float, device=device)
    token_type_ids[:, context.shape[1]:] = 1
    inputs = {'inputs_embeds': h.transpose(0, 1),
              'target_mapping': target_mapping,  # LB1
              'perm_mask': perm_mask,
              'token_type_ids': token_type_ids}
    o = self(**inputs)[0]
    context_embedded = o[:, :context.shape[1]]
    o = o[:, context.shape[1]:]
    # o = o[:, 1:]  # match length difference caused by the prediction token
    # XLNet is a same token prediction task model
    pred_mask = F.one_hot(pred_ids, num_classes=o.shape[1]).bool().unsqueeze(-1)
    o_cut = o.detach() * (~pred_mask).float() + o * pred_mask.float()  # cut grad

    return o_cut, context_embedded


def gpt2_embed(self, input_ids, **kwargs):
    past_length = 0
    past = [None] * len(self.h)
    position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = [None] * self.config.n_layer

    input_ids = input_ids.view(-1, input_ids.size(-1))
    position_ids = position_ids.view(-1, position_ids.size(-1))

    inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    token_type_embeds = 0
    hidden_states = inputs_embeds + position_embeds + token_type_embeds
    return hidden_states, {'past': past, 'head_mask': head_mask}


def gpt2_run_cells(self, context, hidden_states, position_ids=None,
                   token_type_ids=None, past=None,
                   attention_mask=None, head_mask=None,
                   output_hidden_states=False,
                   output_attentions=False, **kwargs):

    hidden_states = torch.cat((context, hidden_states), dim=1)
    hidden_states = self.drop(hidden_states)

    input_shape = hidden_states.shape[:-1]
    output_shape = input_shape + (hidden_states.size(-1),)

    presents = ()
    all_attentions = []
    all_hidden_states = ()
    for i, (block, layer_past) in enumerate(zip(self.h, past)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

        outputs = block(hidden_states, layer_past=layer_past,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i])
        hidden_states, present = outputs[:2]
        presents = presents + (present,)

        if output_attentions:
            all_attentions.append(outputs[2])

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(*output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states, presents)
    if output_hidden_states:
        outputs = outputs + (all_hidden_states,)
    if output_attentions:
        # let the number of heads free (-1) so we can extract attention even after head pruning
        attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
        all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
        outputs = outputs + (all_attentions,)

    # last hidden state, presents, (all hidden_states), (attentions)
    o = outputs[0]
    context_embedded = o[:, :context.shape[1]]
    o = o[:, context.shape[1]:]

    return o, context_embedded

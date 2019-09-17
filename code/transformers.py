import random

import torch
from pytorch_transformers import (
    BertForMaskedLM, BertTokenizer,
    GPT2DoubleHeadsModel, GPT2Tokenizer
)


# we support only GPT2 at the moment
Models = {
    'bert': (BertForMaskedLM, BertTokenizer, 'bert-base-uncased'),
    # 'xlnet': (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
    'gpt2': (GPT2DoubleHeadsModel, GPT2Tokenizer, 'gpt2'),
}


def get_transformer(model_name):
    # tokenizing and numericalizing
    return {
        'bert': get_bert,
        'gpt2': get_gpt2,
        'none': get_no_transformer,
    }[model_name.lower()]()


def get_no_transformer():
    return None, None


def get_bert():
    model = 'bert'
    model_class, tokenizer, model_name = Models[model]

    tokenizer = tokenizer.from_pretrained(model_name)
    for token in ['mask', 'sep', 'cls', 'pad', 'unk']:
        setattr(tokenizer, f'{token}_id',
                tokenizer.encode(getattr(tokenizer, f'{token}_token'))[0])
    tokenizer.model_name = model

    net = model_class.from_pretrained(model_name)

    return net, tokenizer


def get_gpt2():
    model_name = 'gpt2'
    # trying to not add additional tokens to orig vocab
    default_special_tokens = {
        'unk_token': "___",
        #'sep_token': "<|endoftext|>",
        # 'cls_token': "[CLS]",
        'pad_token': "_____",
        'sep_token': "<|endoftext|>",
    }
    special_tokens = {
        'cls_token': "<|endoftext|>",
        # 'type_a': '[TYPE_A]',
        # 'type_b': '[TYPE_B]',
        # 'person': '[SOMEONE]'
    }
    person_tokens = {f'person{i}': f'[PERSON{i}]' for i in range(20)}
    person_tokens = {'blank': '[...]', **person_tokens}

    model_class, tokenizer, model_name = Models[model_name]

    tokenizer = tokenizer.from_pretrained(model_name, **default_special_tokens)
    # tokenizer.add_tokens(person_tokens.values())
    tokenizer.add_special_tokens(person_tokens)
    # unk is needed to add other special tokens

    for k, v in tokenizer.special_tokens_map.items():
        if k.endswith('_token'):
            setattr(tokenizer, f"{k[:-6]}_id",
                tokenizer.convert_tokens_to_ids(v))
        else:
            setattr(tokenizer, f"{k}_id",
                tokenizer.convert_tokens_to_ids(v))
    tokenizer.cls_id = tokenizer.encode(special_tokens['cls_token'])[0]
    tokenizer.model_name = model_name

    net = model_class.from_pretrained(model_name)
    # resize embeddings
    if tokenizer.vocab_size != len(tokenizer):
        net.resize_token_embeddings(len(tokenizer))

    return net, tokenizer


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



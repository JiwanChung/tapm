import torch

from exp import ex
from run_transformer import transformer_embed, transformer_run_cells
from .no_gt import ConcatNoGtSos


class ConcatMask(ConcatNoGtSos):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, fix_gpt_epoch):
        super().__init__(transformer, tokenizer, dropout_before, fix_gpt_epoch)

        n_ctx = self.net.transformer.config.n_ctx
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )

    def prepare_transformer_input(self, hypo, context, lengths, group_mask):
        # BGL, B(GL)C, BGK
        B = hypo.shape[0]
        h, res  = transformer_embed(self.net.transformer, hypo)
        past = res['past']
        head_mask = res['head_mask']
        # B (G * L + (G - 1))
        context_sep_embd = self.net.transformer.wte(torch.LongTensor([self.tokenizer.context_sep_id]).to(h.device))
        context_sep_embd = context_sep_embd.view(1, 1, -1).contiguous().expand(B, 1, -1)
        h = torch.cat((context_sep_embd, h), dim=1)

        context_len = context.shape[1]
        total_len = h.shape[1] + context_len
        attention_mask = self.build_attention_mask(hypo, lengths, context_len, total_len)
        head_mask = {'head_mask': head_mask, 'attention_mask': attention_mask}

        return h, past, head_mask

    def run_transformer(self, h, past, head_mask, context):
        attention_mask = head_mask['attention_mask']
        head_mask = head_mask['head_mask']
        if self.dropout_before:
            h = self.dropout(h)
        o, context_embedded = transformer_run_cells(self.net.transformer, context, h, past=past,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask)
        # c = o[:, :context.shape[1]]
        # o = o[:, context.shape[1]:]
        if not self.dropout_before:
            o = self.dropout(o)
        return o, context_embedded

    def build_attention_mask(self, hypo, lengths, context_len, total_len):
        # output: [batch_size, num_heads, from_seq_length, to_seq_length]
        # lengths: BG
        B = hypo.shape[0]
        if lengths is None:
            # inference
            # build lengths
            if hypo.shape[1] > 0:
                hypo_shifted = hypo[:, 1:]
                seq_seps = (hypo_shifted == self.tokenizer.seq_sep_id)  # BL skip sos
                num_sents = seq_seps.sum(dim=-1).squeeze(-1)  # B
                max_num_sents = num_sents.max().item()
                max_num_sents = max_num_sents + 1 if max_num_sents < 5 else max_num_sents
                pseudo_lengths = torch.zeros(B, max_num_sents).to(hypo.device).long()
                max_len = torch.Tensor([hypo_shifted.shape[1]]).to(hypo.device).to(hypo.dtype)
                for b in range(B):
                    seq_sep = seq_seps[b].nonzero().squeeze(-1)
                    if seq_sep.shape[0] < 5:
                        seq_sep = torch.cat((seq_sep, max_len), dim=0)
                    pseudo_lengths[b, :seq_sep.shape[0]] = seq_sep
                # absence of last seq_sep

                # cumsubtraction
                p = pseudo_lengths - pseudo_lengths.roll(1, 1)
                p[:, 0] = pseudo_lengths[:, 0]
                p = p.clamp(min=0)  # in case of adding the last max_len
                pseudo_lengths = p
                lengths = pseudo_lengths
            else:
                lengths = None

        if lengths is not None:
            skips = 2  # context_sep, cls
            max_size = context_len + skips + lengths.sum(dim=-1).max().item()  # context_sep, cls
            # assert max_size == total_len, f"recomputed size: {max_size}, representation size: {total_len}"
            if total_len > max_size:
                diff = (total_len - context_len) - lengths.sum(dim=-1).max().item()
                skips = diff
                print(f"recomputed size: {max_size}, representation size: {total_len}")
                max_size = total_len

            mask = self.bias[:, :, :max_size, :max_size].bool()
            mask = mask.expand(B, -1, -1, -1)

            segments = torch.cat((torch.zeros(B, 1).to(lengths.dtype).to(lengths.device), lengths), dim=1)
            segments = torch.stack((segments, segments.roll(shifts=-1, dims=-1)), dim=2)
            segments = segments[:, :-1]  # BG2
            segments = segments.cumsum(dim=1)
            segments = segments + skips + context_len  # shifting

            attention_mask = torch.ones_like(mask)
            for b in range(B):
                for i in range(segments.shape[1]):
                    for j in range(i+1, segments.shape[1]):
                        attention_mask[b, 0, segments[b,i,0]:segments[b,i,1], segments[b,j,0]:segments[b,j,1]] = 0

            attention_mask = mask & attention_mask
            attention_mask = attention_mask.byte()
        else:
            max_size = total_len
            mask = self.bias[:, :, :max_size, :max_size].bool()
            mask = mask.expand(B, -1, -1, -1)
            attention_mask = mask

        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0
        return attention_mask

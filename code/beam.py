# coding=utf-8
# Copyright (c) 2019 Yang Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
A general wrapper around models with LM heads to generate sequences
using beam search.
"""
import torch
from torch import nn

from exp import ex
from utils import chunks


class TransformerBeamSearch(nn.Module):
    @ex.capture
    def __init__(
        self,
        model,
        batch_size,
        num_samples,
        max_target_len,
        min_length=4,
        alpha=0,
        block_repeating_trigram=True,
    ):
        """
        Attributes:
            mask_word_id: token id that corresponds to the mask
        """
        super(TransformerBeamSearch, self).__init__()
        self.model = model
        self.tokenizer = model.tokenizer
        tokenizer = self.tokenizer

        self.start_token_id = tokenizer.cls_id
        self.end_token_id = tokenizer.eos_id
        self.pad_token_id = tokenizer.pad_id

        self.beam_size = num_samples
        self.min_length = min_length
        self.max_length = max_target_len

        self.block_repeating_trigram = block_repeating_trigram
        self.apply_length_penalty = False if alpha == 0 else True
        self.alpha = alpha

    def init_states(self, batch_size, device):
        # State of the beam
        self.hypotheses = [[] for _ in range(batch_size)]
        self.batch_offset = torch.arange(batch_size, dtype=torch.long).to(device)
        self.beam_offset = torch.arange(
            0, batch_size * self.beam_size, step=self.beam_size, dtype=torch.long
        ).to(device)
        self.growing_beam = torch.full(
            (batch_size * self.beam_size, 1), self.start_token_id, dtype=torch.long
        ).to(device)
        self.topk_log_probabilities = torch.tensor(
            [0.0] + [float("-inf")] * (self.beam_size - 1), dtype=torch.float
        ).repeat(batch_size).to(device)
        self.results = {
            "predictions": [[] for _ in range(batch_size)],
            "scores": [[] for _ in range(batch_size)],
        }
        self._step = 0
        self.is_done = False

    def step(self, log_probabilities):
        """ Grows the beam by one step. """
        self._step += 1

        # The batch size changes as some beams finish so we define _B
        vocab_size = log_probabilities.size(-1)
        _B = log_probabilities.size(0) // self.beam_size

        # Multiply each beam probability with the probability of the
        # next token (conditioned on the words in the beam).
        log_probabilities += self.topk_log_probabilities.view(-1, 1)

        log_probabilities = self.enforce_min_length(log_probabilities)
        if self.block_repeating_trigram:
            log_probabilities = self.remove_repeating_trigrams(log_probabilities, _B)

        # Find the `beam_size` (previous_beam + token) combinations with
        # the highest score
        topk_log_probabilities, topk_ids = torch.topk(
            log_probabilities.view(_B, self.beam_size * vocab_size),
            self.beam_size,
            dim=1,
        )

        # Apply the length penalty. The +1 accounts for the [EOS] token
        # that will be added if the beam ends.
        topk_scores = topk_log_probabilities / self.length_penalty()

        # Retrieve the corresponding respective beam and token id
        # topk_token_ids[i] will be added to topk_beam_ids[i]
        topk_beam_ids = topk_ids.div(vocab_size)
        topk_token_ids = topk_ids.fmod(vocab_size)

        # Retrieve the row index of the surviving beams in the original
        # view of the log_probabilities tensor
        surviving_beams_rows = (topk_beam_ids + self.beam_offset[:_B].view(-1, 1)).view(
            -1
        )

        # Append the last predictions
        self.growing_beam = torch.cat(
            [
                self.growing_beam.index_select(0, surviving_beams_rows),
                topk_token_ids.view(-1, 1),
            ],
            1,
        )

        # Check if any of the beam searches has ended during this
        # growth step. Also if top beam (most probable) has ended
        # for one element of the batch.
        is_finished = topk_token_ids.eq(self.end_token_id)
        is_finished = self.enforce_max_length(is_finished)
        is_top_beam_finished = is_finished[:, 0].eq(1)

        # Save the finished searches
        if is_finished.any():
            predictions = self.growing_beam.view(
                -1, self.beam_size, self.growing_beam.size(1)
            )
            for i in range(is_finished.size(0)):
                if is_top_beam_finished[i]:
                    is_finished[i].fill_(1)
                finished_hyp = is_finished[i].nonzero().view(-1)

                # Store finished hypotheses for this batch.
                b = self.batch_offset[i]
                for j in finished_hyp:
                    self.hypotheses[b].append((topk_scores[i, j], predictions[i, j, :]))

                # If the batch reached the end, save the best hypotheses
                # in terms of length-penalized score.
                if is_top_beam_finished[i]:
                    best_hyp = sorted(
                        self.hypotheses[b], key=lambda x: x[0], reverse=True
                    )
                    best_score, best_prediction = best_hyp[0]
                    self.results["scores"][b].append(best_score)
                    self.results["predictions"][b].append(best_prediction)

            non_finished = is_top_beam_finished.eq(0).nonzero().view(-1)
            if len(non_finished) == 0:
                self.is_done = True

            # Remove finished batches for the next step.
            topk_log_probabilities = topk_log_probabilities.index_select(
                0, non_finished
            )
            self.batch_offset = self.batch_offset.index_select(0, non_finished)
            self.growing_beam = predictions.index_select(0, non_finished).view(
                -1, self.growing_beam.size(-1)
            )

            # surviving_beams_rows = surviving_beams_rows.index_select(0, non_finished)

        return surviving_beams_rows

    def tile(self, x, i):
        x = x.unsqueeze(0).repeat(i, *[1 for _ in range(len(x.shape))])
        x = x.view(-1, *x.shape[2:]).contiguous()
        return x

    def forward(self, batch, **kwargs):
        # forward pass on the encoder
        features, features_merged, keywords, G = self.model.prepare_group(batch)
        # grow the beam by generating sequences in an autoregressive way
        self.start_token_id = self.model.tokenizer.cls_id
        sample_feature = features_merged[list(features.keys())[0]]
        device = sample_feature.device
        self.batch_size = sample_feature.shape[0]
        B = self.batch_size
        self.init_states(B, device)
        self.growing_beam = torch.full(
            (B * self.beam_size, 1), self.start_token_id, dtype=torch.long
        ).to(device)
        for k, v in features_merged.items():
            features_merged[k] = self.tile(v, self.beam_size)
        keywords = self.tile(keywords, self.beam_size)
        for step in range(self.max_length):
            decoder_input = self.growing_beam
            logit, _, _ = self.model.run_token(
                                    decoder_input,
                                    features_merged, keywords)
            logit = logit[:, -1]

            log_probabilities = torch.nn.functional.log_softmax(logit, dim=-1)
            surviving_beams_rows = self.step(log_probabilities)
            if self.is_done:
                break

            for k, v in features_merged.items():
                features_merged[k] = v.index_select(0, surviving_beams_rows)
            keywords = keywords.index_select(0, surviving_beams_rows)

        hypo = self.results['predictions']
        hypo = [self.remove_after_sep(h[0][1:]) for h in hypo]  # remove sos, eos
        hypo = list(chunks(hypo, 5))

        return hypo

    def remove_after_sep(self, x):
        idx = (x == self.tokenizer.sep_id).nonzero()
        if idx.nelement() > 0:
            idx = idx[0].item()
            return x[:idx + 1]
        else:
            return x

    def remove_repeating_trigrams(self, log_probabilities, _B):
        if(self._step + 1 > 3):
            for i in range(_B * self.beam_size):
                tokens = [t for t in self.growing_beam[i]]
                trigrams = [(tokens[i - 1], tokens[i], tokens[i + 1]) for i in range(1, len(tokens) - 1)]
                if len(trigrams) > 0:
                    last_trigram = tuple(trigrams[-1])
                    if last_trigram in trigrams[:-1]:
                        log_probabilities[i] = -1e20
        return log_probabilities

    def enforce_min_length(self, log_probabilities):
        if self._step < self.min_length:
            log_probabilities[:, self.end_token_id] = -1e20
        return log_probabilities

    def enforce_max_length(self, is_finished):
        if self._step + 1 == self.max_length:
            is_finished.fill_(1)
        return is_finished

    def length_penalty(self):
        return ((5.0 + (self._step + 1)) / 6.0) ** self.alpha


def tile(x, count, dim=0):
    """
    Tiles `x` along dimension `dim` `count` times.
    Example:
        >> ex = torch.tensor([1,2],[3,4])
        >> tile(ex, 2, 0)
        torch.Tensor([[1,2],[1,2],[3,4],[3,4]])
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

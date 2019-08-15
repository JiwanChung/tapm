import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import make_autoencoder_batch, make_bert_batch

from .transformer_model import TransformerModel
from .encoder import Encoder
from .decoder import Decoder


class Autoencoder(TransformerModel):
    transformer_name = 'gpt2'

    def __init__(self, args, transformers, tokenizer):
        super(Autoencoder, self).__init__()

        self.use_keyword = args.use_keyword

        if self.use_keyword:
            self.encoder = Encoder(args, transformers['encoder'], tokenizer)
        self.decoder = Decoder(args, transformers['decoder'], tokenizer)

    def make_batch(self, *args, **kwargs):
        return make_autoencoder_batch(*args, **kwargs)

    def forward(self, batch, **kwargs):
        sentence = batch.sentences
        targets = batch.targets
        lengths = batch.lengths
        if self.use_keyword:
            keywords, keyword_lengths, scores, reg_loss = \
                self.encoder(sentence, lengths)
        else:
            keywords, keyword_lengths, scores, reg_loss = None, None, None, None
        logits = self.decoder(sentence, lengths,
                              keywords, keyword_lengths, scores)
        return logits, targets, reg_loss, {'prob': scores.mean().item()}, keywords


class MultiLabelAutoencoder(TransformerModel):
    transformer_name = 'bert'

    def __init__(self, args, transformer, tokenizer):
        super(MultiLabelAutoencoder, self).__init__()

        self.net = transformer
        self.net.train()

        self.tokenizer = tokenizer

        bert_dim = self.net.bert.config.hidden_size
        self.reduce_dim = nn.Linear(bert_dim * 2, bert_dim)

    def make_batch(self, *args, **kwargs):
        return make_bert_batch(*args, **kwargs)

    def pool(self, x, dim=-1):
        return x.mean(dim=dim)

    def extend_attention_mask(self, input_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.net.bert.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def get_head_mask(self):
        head_mask = [None] * self.net.bert.config.num_hidden_layers
        return head_mask

    def forward(self, batch, **kwargs):
        sentences = batch.sentences
        targets = batch.targets

        attention_mask = sentences != self.tokenizer.pad_id
        extended_attention_mask = self.extend_attention_mask(sentences, attention_mask)
        encoder_out = self.net.bert(sentences, attention_mask=attention_mask)[0]
        encoder_out = self.pool(encoder_out, dim=1)  # BC
        encoder_out = self.net.cls(encoder_out)
        keyword_prob = F.softmax(encoder_out, dim=-1)
        keyword_att = torch.matmul(keyword_prob, self.net.bert.embeddings.word_embeddings.weight)

        empty = sentences.clone()
        empty = empty.fill_(0)
        decoder_in = self.net.bert.embeddings(empty)
        decoder_in = torch.cat((decoder_in, keyword_att.unsqueeze(1).expand(*decoder_in.shape)), dim=-1)
        decoder_in = self.reduce_dim(decoder_in)
        head_mask = self.get_head_mask()
        decoder_out = self.net.bert.encoder(decoder_in, extended_attention_mask,
                                            head_mask=head_mask)[0]

        logits = self.net.cls(decoder_out)

        with torch.no_grad():
            stats = {'keyword_lhalf': ((keyword_prob ** 2).sum(dim=-1) ** 0.5).mean().item()}
            _, keywords = keyword_prob.sort(dim=-1, descending=True)  # BV
            keywords = keywords[:, :10]

        return logits, targets, None, stats, keywords

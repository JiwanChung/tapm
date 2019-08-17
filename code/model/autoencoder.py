import torch
from torch import nn
import torch.nn.functional as F

from data.batcher import make_autoencoder_batch, make_bert_batch

from tensor_utils import remove_cls_sep
from .transformer_model import TransformerModel
from .encoder import Encoder
from .decoder import Decoder
from .sparsemax import Sparsemax
from .modules import LSTMDecoder, l_n_norm


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

        self.threshold_keyword = args.get('threshold_keyword', False)
        self.extraction_min_words = args.extraction_min_words
        self.keyword_ratio = args.keyword_ratio
        self.use_reg_loss = args.get('use_reg_loss', False)

        self.net = transformer
        self.net.train()

        self.tokenizer = tokenizer

        bert_dim = self.net.bert.config.hidden_size
        # self.reduce_dim = nn.Linear(bert_dim * 2, bert_dim)
        self.keyword_linear = nn.Linear(bert_dim, bert_dim)

        self.use_bert = False
        if not self.use_bert:
            self.decoder = LSTMDecoder(self.net.bert.embeddings.word_embeddings)

        self.keyword_activation = args.get('keyword_activation', 'sparsemax')
        self.keyword_activation = {
            'softmax': self.softmax,
            'sparsemax': self.sparsemax,
            'sigmoid': self.sigmoid,
            'saturating_sigmoid': self.saturating_sigmoid,
        }[self.keyword_activation.lower()]
        self.loss_norm_n = args.get('loss_norm_n', 1)
        self.reduce_dim = nn.Linear(bert_dim * 2, bert_dim)

    def get_l_n_loss(self, x, lengths):
        x = F.relu(x)
        reg_loss = l_n_norm(x, n=self.loss_norm_n, dim=-1)
        reg_loss = F.relu(reg_loss - self.extraction_min_words)
        max_keyword_num = torch.max(torch.LongTensor([self.extraction_min_words]).to(lengths.device),
                                    (lengths.float() * self.keyword_ratio).ceil().long())
        max_limit_loss = F.relu(max_keyword_num.float() - l_n_norm(x, n=1, dim=-1))
        reg_loss = reg_loss + max_limit_loss
        return reg_loss.mean()

    def keyword_activation_loss(self, x, lengths):
        x = self.keyword_activation(x)
        reg_loss = self.get_l_n_loss(x, lengths)
        return x, reg_loss

    def softmax(self, x):
        x = F.softmax(x, dim=-1)
        return x

    def sparsemax(self, x):
        mm = Sparsemax(dim=-1)
        x = mm(x)
        return x

    def sigmoid(self, x):
        x = torch.sigmoid(x)
        return x

    def saturating_sigmoid(self, x):
        x = 1.2 * torch.sigmoid(x) - 0.1
        x = x.clamp(0, 1)
        return x

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

    def get_position_embeddings(self, input_ids):
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return self.net.bert.embeddings.position_embeddings(position_ids)

    def keyword_thresholding(self, p, lengths):
        if self.threshold_keyword:
            max_keyword_num = torch.max(torch.LongTensor([self.extraction_min_words]).to(lengths.device),
                                        (lengths.float() * self.keyword_ratio).ceil().long())
            p_sorted, _ = p.sort(dim=-1, descending=True)  # BC
            min_vals = p_sorted.gather(dim=-1, index=(max_keyword_num - 1).unsqueeze(-1))  # keep ties
            keyword_mask = (p >= min_vals)
            p = p * keyword_mask.float()
        return p

    def mask_keyword(self, p, sentences):
        # BV, BL
        src = torch.FloatTensor([0]).to(p.device).view(1, 1, 1)
        p = p.unsqueeze(1).expand(*sentences.shape, p.shape[-1])
        temp = torch.full_like(p, 0)
        sentences = sentences.unsqueeze(-1).expand(*sentences.shape, p.shape[-1])
        temp = temp.scatter(dim=-1, index=sentences,
                            src=p)
        # BLV
        p = temp.sum(dim=1)
        p = F.normalize(p, p=1, dim=-1)
        return p

    def forward(self, batch, **kwargs):
        sentences = batch.sentences
        targets = batch.targets
        lengths = batch.lengths

        attention_mask = sentences != self.tokenizer.pad_id
        extended_attention_mask = self.extend_attention_mask(sentences, attention_mask)

        encoder_out = self.net.bert(sentences, attention_mask=attention_mask)[0]
        encoder_out = self.pool(encoder_out, dim=1)  # BC
        encoder_out = self.net.cls(encoder_out)
        keyword_prob, reg_loss = self.keyword_activation_loss(encoder_out, lengths)
        keyword_prob = self.mask_keyword(keyword_prob, sentences)
        keyword_prob_t = self.keyword_thresholding(keyword_prob, lengths)
        keyword_att = torch.matmul(keyword_prob_t, self.net.bert.embeddings.word_embeddings.weight)

        L = sentences.shape[1]
        keyword_att = self.keyword_linear(keyword_att)
        if self.use_bert:
            keyword_att = keyword_att.unsqueeze(1).expand(-1, L, -1)
            decoder_in = keyword_att + self.get_position_embeddings(sentences)
            head_mask = self.get_head_mask()
            decoder_out = self.net.bert.encoder(decoder_in, extended_attention_mask,
                                                head_mask=head_mask)[0]
            logits = self.net.cls(decoder_out)
        else:
            dec_input, target = remove_cls_sep(targets, self.tokenizer)
            dec_input = self.net.bert.embeddings.word_embeddings(dec_input)
            dec_input = torch.cat((dec_input, keyword_att.unsqueeze(1).expand_as(dec_input)), dim=-1)
            dec_input = self.reduce_dim(dec_input)
            logits = self.decoder(keyword_att, dec_input, embedded=True)

        with torch.no_grad():
            stats = {
                'keyword_l2': ((keyword_prob ** 2).sum(dim=-1) ** 0.5).mean().item(),
                'keyword_l1': keyword_prob.sum(dim=-1).mean().item(),
                'keyword>0.1': (keyword_prob > 0.1).sum(dim=-1).float().mean().item(),
                'keyword>0': (keyword_prob > 0).sum(dim=-1).float().mean().item(),
            }
            if self.threshold_keyword:
                stats = {
                    **stats,
                    'keyword_nonzero': (keyword_prob_t > 0).sum(dim=-1).float().mean().item(),
                }
            scores, keywords = keyword_prob.sort(dim=-1, descending=True)  # BV
            keywords = keywords[:, :10]
            scores = scores[:, :10]

        if not self.use_reg_loss:
            reg_loss = None

        return logits, targets, reg_loss, stats, (keywords, scores)

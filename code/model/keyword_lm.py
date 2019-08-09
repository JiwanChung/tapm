from torch import nn


class LSTMKeywordLM(nn.Module):
    def __init__(self):
        super(LSTMKeywordLM, self).__init__()

    def forward(self, sentences, lengths, keywords, keyword_lengths, targets):
        # concat keyword and sentence
        return

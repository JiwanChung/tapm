import torch
from torch import nn

from exp import ex
from .awd import RNNModel


class PretrainedQRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn_type = getattr(self, 'rnn_type', 'qrnn')

        if not hasattr(self, 'args'):
            self.args = {
                'rnn_type': 'QRNN',
                'nhip': 400,
                'nhid': 1550,
                'nlayers': 4,
                'dropouth': 0.3,
                'dropouti': 0.4,
                'dropoute': 0.1,
                'wdrop': 0.5,
            }
        self.args['ntokens'] = None
        self.rnn = RNNModel(**self.args)
        pretrained_weight = self.load(self.rnn_type)
        self.rnn.load_state_dict(pretrained_weight.state_dict())

    @ex.capture
    def load(self, rnn_type, root):
        with open(root / 'data' / 'cache' / f"{rnn_type}.pkl", 'rb') as f:
            model, criterion, optimizer = torch.load(f)
        return model


class PretrainedLSTM(nn.Module):
    def __init__(self):
        self.rnn_type = 'lstm'

        self.args = {
            'rnn_type': 'QRNN',
            'nhip': 400,
            'nhid': 1150,
            'nlayers': 4,
            'dropouth': 0.25,
            'dropouti': 0.4,
            'dropoute': 0.1,
            'wdrop': 0.5,
        }

        super().__init__()

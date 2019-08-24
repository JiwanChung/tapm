import torch
from torch import nn
import torch.nn.functional as F

from loss import FocalLoss


class KeywordClassifier(nn.Module):
    def __init__(self, keyword_num, dim, feature_names,
                 video_dim, image_dim, gamma=2):
        super(KeywordClassifier, self).__init__()

        self.keyword_num = keyword_num
        self.feature_names = feature_names
        self.dim = dim
        self.video_dim = video_dim
        self.image_dim = image_dim

        for feature in self.feature_names:
            setattr(self, feature, nn.Linear(getattr(self, f"{feature}_dim"), self.dim))
        self.layer_norm = nn.LayerNorm(self.dim)
        self.out = nn.Linear(self.dim, self.keyword_num)

        self.loss = FocalLoss(gamma)

    def forward(self, keywords, features):
        # BVK, BVNC
        hypo = {}
        for feature in self.feature_names:
            hypo[feature] = getattr(self, feature)(features[feature])
            if hypo[feature].dim() > 3:
                hypo[feature] = hypo[feature].mean(dim=-2)
            hypo[feature] = F.relu(hypo[feature])
        # BVK
        hypo = torch.stack(list(hypo.values()), dim=0).mean(dim=0)
        hypo = self.layer_norm(hypo)
        hypo = self.out(hypo)
        hypo = torch.sigmoid(hypo)

        loss = None
        if keywords is not None:
            loss, _ = self.loss(hypo, keywords)

        return hypo, loss
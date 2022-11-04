import torch
import numpy as np

from torchvision import models
from torch.nn import functional as F

@torch.no_grad()
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.feature_shape = None

    def forward(self, X, indices=None):
        # self.feature_shape = X.shape
        if indices is None:
            indices = [7,21]
        out = []
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i+1) in indices:
                if self.feature_shape is None:
                    self.feature_shape = X.shape
                else:
                    X = F.interpolate(X,self.feature_shape[-2:],mode='nearest')#,align_corners=True)
                out.append(X)
        return torch.cat(out,1)
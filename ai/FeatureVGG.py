import torch
import torch.nn as nn
from torchvision.models import vgg11_bn

class FeatureVGG(nn.Module):

    def __init__(self):
        super(FeatureVGG, self).__init__()

        self.vgg = vgg11_bn(pretrained=True).features
        
    def forward(self, x):
        n = x.size(0)
        n_s = x.size(1)
        c = x.size(2)
        h = x.size(3)
        w = x.size(4)

        x = x.view(n*n_s, c, h, w)

        x1 = x[:, :3]
        x2 = x[:, 3:]

        x1 = self.vgg(x1)
        x2 = self.vgg(x2)
        
        x = torch.cat([x1, x2], dim=1)
        x = x.view(n, n_s, -1)

        return x

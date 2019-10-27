import torch
import torch.nn as nn
from torchvision.models import vgg11_bn

class FeatureVGG(nn.Module):

    def __init__(self):
        super(FeatureVGG, self).__init__()

        self.vgg = vgg11_bn(pretrained=True).features

        self.classifier1 = nn.Sequential(
            nn.Linear(1024*4*4, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(),

            nn.Linear(128, 32),
            nn.Dropout(0.4),
            nn.LeakyReLU(),

            nn.Linear(32, 4),
            nn.LogSoftmax(dim=-1)
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(512*4*4, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(),

            nn.Linear(128, 32),
            nn.Dropout(0.4),
            nn.LeakyReLU(),

            nn.Linear(32, 4),
            nn.LogSoftmax(dim=-1)
        )
        
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

        logps = self.classifier1(x.view(n*n_s, -1))
        logps1 = self.classifier2(x1.view(n*n_s, -1))
        logps2 = self.classifier2(x2.view(n*n_s, -1))

        return x, logps, logps1, logps2

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        print("Feature VGG was saved.")

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print("Feature VGG was loaded.")


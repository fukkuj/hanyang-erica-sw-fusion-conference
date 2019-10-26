import torch
import torch.nn as nn

from ai.FeatureCNN import FeatureCNN
from torchvision.models import vgg11_bn
from env import *


class TrashDetector(nn.Module):
    """
    Detector to detect whether there is some trash or not.
    """
    
    def __init__(self, fine_tune=False):
        super(TrashDetector, self).__init__()
       
        
        # self.features = nn.Sequential(
        #     # 32 x 128 x 128
        #     nn.Conv2d(6, 16, (5, 5), stride=1, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(),
            
        #     # 32 x 64 x 64
        #     nn.MaxPool2d((2, 2), stride=2, padding=0),
            
        #     # 64 x 64 x 64
        #     nn.Conv2d(16, 16, (5, 5), stride=1, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(),
            
        #     # 64 x 32 x 32
        #     nn.MaxPool2d((2, 2), stride=2, padding=0),
            
        #     # 128 x 32 x 32
        #     nn.Conv2d(16, 32, (5, 5), stride=1, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
            
        #     # 128 x 16 x 16
        #     nn.MaxPool2d((2, 2), stride=2, padding=0),
            
        #     # 256 x 16 x 16
        #     nn.Conv2d(32, 32, (5, 5), stride=1, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
            
        #     # 256 x 8 x 8
        #     nn.MaxPool2d((2, 2), stride=2, padding=0),
        # )
        
        

        self.features = FeatureCNN()
        self.features.load(CNN_CKPT_PATH)
        self.features.requires_grad_(False)
        self.fine_tune = fine_tune

        # construct classifier
        self.classifier = nn.Sequential(
            nn.Linear(8*8*32, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )

        # self.features = vgg11_bn(pretrained=True).features
        # self.features.requires_grad_(fine_tune)
        # self.fine_tune = fine_tune

        # self.conv = nn.Sequential(
        #     nn.Conv2d(1024, 1024, (3, 3), stride=2, padding=1),
        #     nn.BatchNorm2d(1024),
        #     nn.Tanh(),

        #     nn.Conv2d(1024, 1024, (2, 2), stride=1, padding=0),
        #     nn.BatchNorm2d(1024),
        #     nn.Tanh()
        # )

        # self.classifier = nn.Sequential(
        #     nn.Linear(1024, 128),
        #     nn.Tanh(),
        #     nn.Dropout(0.4),

        #     nn.Linear(128, 2),
        #     nn.LogSoftmax(dim=-1)
        # )
        
    def forward(self, x):
        """
        Forward propagation phase.
        
        Arguments:
        ----------
        :x images shape of (batch_size, 8, CHANNEL, HEIGHT, WIDTH)
        """

        if self.fine_tune is False:
            self.features.eval()

        # retrieve number of images
        n = x.size(0)
        
        # x = self.features(x)
        # x = x.view(n, -1)
        # x = self.classifier(x)

        # x1 = self.features(x[:, :3])
        # x2 = self.features(x[:, 3:])
        # x = torch.cat([x1, x2], dim=1)

        # x = self.conv(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        return x
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        print("Detector was saved.")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print("Detector was loaded.")

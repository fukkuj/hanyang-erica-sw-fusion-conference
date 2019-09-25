import torch
import torch.nn as nn

from ai.FeatureAE import FeatureAE
from env import *


class TrashDetector(nn.Module):
    """
    Detector to detect whether there is some trash or not.
    """
    
    def __init__(self):
        super(TrashDetector, self).__init__()
        
        # construct feature extractor.
        self.features = FeatureAE()
        self.features.load(AE_CKPT_PATH)
        for param in self.features.parameters():
            param.requires_grad_(False)
        
        self.cnn = nn.Sequential(
            # 128 x 8 x 8
            nn.Conv2d(128, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            # 32 x 4 x 4
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            
            # 16 x 4 x 4
            nn.Conv2d(32, 16, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )

        # construct classifier
        self.classifier = nn.Sequential(
            nn.Linear(4*4*16, 16),
            nn.Tanh(),
            nn.Dropout(0.5),

            nn.Linear(16, 2),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        """
        Forward propagation phase.
        
        Arguments:
        ----------
        :x images shape of (batch_size, 8, CHANNEL, HEIGHT, WIDTH)
        """

        # retrieve number of images
        n = x.size(0)
        
        _, x = self.features(x)
        x = self.cnn(x)
        x = x.view(n, -1)
        x = self.classifier(x)

        return x
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        print("Detector was saved.")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print("Detector was loaded.")

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
        
        self.features = nn.Sequential(
            # 32 x 128 x 128
            nn.Conv2d(IN_CHANNEL, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 32 x 64 x 64
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            
            # 64 x 64 x 64
            nn.Conv2d(32, 64, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # 64 x 32 x 32
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            
            # 128 x 32 x 32
            nn.Conv2d(64, 128, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            # 128 x 16 x 16
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            
            # 256 x 16 x 16
            nn.Conv2d(128, 256, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            # 256 x 8 x 8
            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )

        # construct classifier
        self.classifier = nn.Sequential(
            nn.Linear(8*8*256, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 2),
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
        
        x = self.features(x)
        x = x.view(n, -1)
        x = self.classifier(x)

        return x
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        print("Detector was saved.")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print("Detector was loaded.")

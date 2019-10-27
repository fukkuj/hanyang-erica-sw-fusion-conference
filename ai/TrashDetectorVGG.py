import torch
import torch.nn as nn

from ai.FeatureVGG import FeatureVGG
from env import *


class TrashDetector(nn.Module):
    """
    Detector to detect whether there is some trash or not.
    """

    def __init__(self, fine_tune=False):
        super(TrashDetector, self).__init__()
       
        self.features = FeatureVGG()
        self.features.load(VGG_CKPT_PATH)
        self.features.requires_grad_(fine_tune)
        self.fine_tune = fine_tune

        # construct classifier
        self.classifier = nn.Sequential(
            nn.Linear(4*4*1024, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        """
        Forward propagation phase.
        
        Arguments:
        ----------
        :x images shape of (batch_size*8, CHANNEL, HEIGHT, WIDTH)
        """

        if self.fine_tune is False:
            self.features.eval()

        # retrieve number of images
        n = x.size(0)

        x, _, _, _ = self.features(x.view(n//8, 8, IN_CHANNEL, HEIGHT, WIDTH))
        x = self.classifier(x.view(n, -1))

        return x
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        print("Detector was saved.")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print("Detector was loaded.")

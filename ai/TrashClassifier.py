import torch
import torch.nn as nn

from ai.FeatureCNN import FeatureCNN
from env import *


class TrashClassifier(nn.Module):
    """
    Neural network for classifying trash.
    It uses FeatureCNN for feature extraction.
    """

    def __init__(self):
        super(TrashClassifier, self).__init__()

        # construct feature extractor.
        self.fcnn = FeatureCNN()
        self.fcnn.load(CNN_CKPT_PATH)
        for param in self.fcnn.parameters():
            param.requires_grad_(False)

        # construct second feature extractor.
        self.features = nn.Sequential(
            # 64*8 x 8 x 8 -> 64 x 8 x 8
            nn.Conv2d(64*8, 64, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # 64 x 8 x 8 -> 64 x 4 x 4
            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )

        # construct classifier.
        self.classifier = nn.Sequential(
            nn.Linear(4*4*64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, len(TRASH_CAT)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """
        Forward propagation phase.
        This function is called by __call__ (maybe)
        
        Arguments:
        ----------
        :x shape of (batch_size, 8, channel=3, height=128, width=128)
        """

        # retrieve useful information
        n = x.size(0)               # number of images.
        num_of_one_shot = x.size(1) # 8
        c = x.size(2)               # channel. 3 since there are 3 channels, R, G, B

        # extract features
        x, _ = self.fcnn(x.view(n*num_of_one_shot, c, HEIGHT, WIDTH))  # processing images using FeatureCNN
        x = x.view(n, 64*8, 8, 8)          # reshape tensor

        # processing features using second feature extractor
        x = self.features(x)
        
        # reshape tensor
        x = x.view(n, -1)
        
        # prediction
        x = self.classifier(x)

        return x

    
    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        print("Classifier was saved.")
        
    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print("Classifier was loaded.")

        
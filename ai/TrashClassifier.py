import torch
import torch.nn as nn

from ai.FeatureCNN import FeatureCNN
from env import HEIGHT, WIDTH, TRASH_CAT


class TrashClassifier(nn.Module):
    """
    Neural network for classifying trash.
    It uses FeatureCNN for feature extraction.
    """

    def __init__(self):
        super(TrashClassifier, self).__init__()

        # construct feature extractor.
        self.fcnn = FeatureCNN()

        # construct second feature extractor.
        self.features = nn.Sequential(
            # 32 x 16 x 16 -> 64 x 16 x 16
            nn.Conv2d(64*8, 64, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            
            # 64 x 16 x 16 -> 64 x 8 x 8
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            # 64 x 8 x 8 -> 32 x 8 x 8
            nn.Conv2d(64, 32, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            # 32 x 8 x 8 -> 32 x 4 x 4
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            # 32 x 4 x 4 -> 16 x 4 x 4
            nn.Conv2d(32, 16, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )

        # construct classifier.
        self.classifier = nn.Sequential(
            nn.Linear(4*4*16, 32),
            nn.Tanh(),
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
        x = self.fcnn(x.view(n*num_of_one_shot, c, HEIGHT, WIDTH))  # processing images using FeatureCNN
        x = x.view(n, 64*8, HEIGHT//(2**3), WIDTH//(2**3))          # reshape tensor

        # processing features using second feature extractor
        x = self.features(x)
        
        # reshape tensor
        x = x.view(n, -1)
        
        # prediction
        x = self.classifier(x)

        return x


        
import torch
import torch.nn as nn

from ai.FeatureCNN import FeatureCNN


class TrashDetector(nn.Module):
    """
    Detector to detect whether there is some trash or not.
    """
    
    def __init__(self):
        super(TrashDetector, self).__init__()
        
        # construct feature extractor.
        self.fcnn = FeatureCNN()

        # construct classifier
        self.classifier = nn.Sequential(
            nn.Linear(3*3*16, 32),
            nn.Tanh(),
            nn.Dropout(0.5),

            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        """
        Forward propagation phase.
        
        Arguments:
        ----------
        :x images shape of (batch_size, )
        """

        # retrieve number of images
        n = x.size(0)

        # extract features using FeatureCNN
        x = self.fcnn(x)

        # flatten features
        x = x.view(n, -1)
        
        # prediction
        x = self.classifier(x)

        return x
        
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

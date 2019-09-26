import torch
import torch.nn as nn

from env import IN_CHANNEL


class FeatureCNN(nn.Module):
    """
    Neural network for feature extraction.
    """

    def __init__(self):
        super(FeatureCNN, self).__init__()

        # feature extractor 1 with 3x3 filters
        self.cnn1 = nn.Sequential(
            # 3 x 128 x 128 -> 32 x 128 x 128
            nn.Conv2d(IN_CHANNEL, 32, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # 32 x 128 x 128 -> 32 x 64 x 64
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            # 32 x 64 x 64 -> 32 x 64 x 64
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 32 x 64 x 64 -> 32 x 32 x 32
            nn.MaxPool2d((2, 2), stride=2, padding=0), # 

            # 32 x 32 x 32 -> 32 x 32 x 32
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 32 x 32 x 32 -> 32 x 16 x 16
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            # 32 x 16 x 16 -> 32 x 16 x 16
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 32 x 16 x 16 -> 32 x 8 x 8
            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )

        # feature extractor 2 with 7x7 filters
        self.cnn2 = nn.Sequential(
            # 3 x 128 x 128 -> 32 x 128 x 128
            nn.Conv2d(IN_CHANNEL, 32, (7, 7), stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 32 x 128 x 128 -> 32 x 64 x 64
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            # 32 x 64 x 64 -> 32 x 64 x 64
            nn.Conv2d(32, 32, (7, 7), stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 32 x 64 x 64 -> 32 x 32 x 32
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            # 32 x 32 x 32 -> 32 x 32 x 32
            nn.Conv2d(32, 32, (7, 7), stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 32 x 32 x 32 -> 32 x 16 x 16
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            # 32 x 16 x 16 -> 32 x 16 x 16
            nn.Conv2d(32, 32, (7, 7), stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 32 x 16 x 16 -> 32 x 8 x 8
            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64*8*8, 64),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            
            nn.Linear(64, 32),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            
            nn.Linear(32, 4),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """
        Arguments:
        ----------
        :x tensor storing images, shaped of (batch_size, channels, height, width) # (n, 3, 128, 128)
        """
        
        n = x.size(0)
        
        # feature extraction with 2 extractors.
        x1 = self.cnn1(x)               # n x 64 x 8 x 8
        x2 = self.cnn2(x)               # n x 64 x 8 x 8

        # concatenation 2 features.
        features = torch.cat([x1, x2], dim=1)  # n x 128 x 8 x 8
        predicted = self.classifier(features.view(n, -1))
        
        return features, predicted
    
    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        print("FeatureCNN was saved.")
        
    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print("FeatureCNN was loaded.")

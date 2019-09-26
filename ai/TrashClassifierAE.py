import torch
import torch.nn as nn

from ai.FeatureAE import FeatureAE
from env import HEIGHT, WIDTH, TRASH_CAT, AE_CKPT_PATH


class TrashClassifier(nn.Module):
    """
    Neural network for classifying trash.
    It uses FeatureCNN for feature extraction.
    """

    def __init__(self, fine_tune=True):
        super(TrashClassifier, self).__init__()

        # construct feature extractor.
        self.ae = FeatureAE()
        self.ae.load(AE_CKPT_PATH)
        if fine_tune is False:
            for param in self.ae.parameters():
                param.requires_grad_(False)

        # construct second feature extractor.
        self.features = nn.Sequential(
            # 128*8 x 8 x 8 -> 32 x 8 x 8
            nn.Conv2d(128*8, 32, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            # 32 x 8 x 8 -> 32 x 4 x 4
            nn.MaxPool2d((2, 2), stride=2, padding=0),

            # 32 x 4 x 4 -> 32 x 4 x 4
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            # 32 x 4 x 4 -> 32 x 2 x 2
            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )

        # construct classifier.
        self.classifier = nn.Sequential(
            nn.Linear(2*2*32, 16),
            nn.Tanh(),
            nn.Dropout(0.5),

            nn.Linear(16, len(TRASH_CAT)),
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
        n = x.size(0)
        one_shot = x.size(1)
        c = x.size(2)
        h = x.size(3)
        w = x.size(4)
        
        reconstructed, latent = self.ae(x.view(n*one_shot, c, h, w))
        latent = latent.view(n, one_shot*latent.size(1), latent.size(2), latent.size(3))
        
        x = self.features(latent)
        x = x.view(n, -1)
        
        x = self.classifier(x)

        return x

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        print("Trash Classifier was saved.")
        
    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print("Trash Classifier was loaded.")
        
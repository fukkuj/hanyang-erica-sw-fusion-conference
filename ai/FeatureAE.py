import torch
import torch.nn as nn
import numpy as np

from env import *


class FeatureAE(nn.Module):
    
    def __init__(self):
        super(FeatureAE, self).__init__()
        
        self.encoder1 = nn.Sequential(
            # 16 x 128 x 128
            nn.Conv2d(IN_CHANNEL, 16, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            
            # 16 x 128 x 128
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            
            # 32 x 64 x 64
            nn.Conv2d(16, 32, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            # 32 x 32 x 32
            nn.Conv2d(32, 32, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            # 32 x 16 x 16
            nn.Conv2d(32, 32, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            # 64 x 8 x 8
            nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        )
        
        self.encoder2 = nn.Sequential(
            # 16 x 128 x 128
            nn.Conv2d(IN_CHANNEL, 16, (7, 7), stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            
            # 16 x 128 x 128
            nn.Conv2d(16, 16, (7, 7), stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            
            # 32 x 64 x 64
            nn.Conv2d(16, 32, (7, 7), stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            # 32 x 32 x 32
            nn.Conv2d(32, 32, (7, 7), stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            # 32 x 16 x 16
            nn.Conv2d(32, 32, (7, 7), stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            # 64 x 8 x 8
            nn.Conv2d(32, 64, (7, 7), stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            
            nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            
            nn.ConvTranspose2d(32, 16, (3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            
            nn.ConvTranspose2d(16, IN_CHANNEL, (3, 3), stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )
        
        self.classifier_conv = nn.Sequential(
            # 128 x 8 x 8
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            
            # 128 x 4 x 4
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            
            # 128 x 4 x 4
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            
            # 128 x 2 x 2
            nn.MaxPool2d((2, 2), stride=2, padding=0)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2*2*128, 64),
            nn.Tanh(),
            nn.Dropout(0.5),
            
            nn.Linear(64, len(TRASH_CAT)),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        """
        Forward propagation (Inference)
        
        Arguments:
        ----------
        :x tensor storing images, shaped of (batch_size, 3, 128, 128)
        
        Returns:
        --------
        :reconstructed reconstructed images, same shaped as x
        """
        
        # compute latent vector using 2 encodeer
        self.latent1 = [x]
        self.latent2 = [x]
        
        for layer in self.encoder1.children():
            self.latent1.append(layer(self.latent1[-1]))
            
        for layer in self.encoder2.children():
            self.latent2.append(layer(self.latent2[-1]))
        
        # concatenate 2 latent vector into 1 latent vector
        latent = torch.cat([self.latent1[-1], self.latent2[-1]], dim=1)
        
        # reconstruct using decoder
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
    
    def classify(self, latent):
        x = self.classifier_conv(latent)
        x = x.view(latent.size(0), -1)
        
        x = self.classifier(x)
        
        return x
    
    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        print("FeatureAE was saved.")
        
    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print("FeatureAE was loaded.")
        
    def contractive_loss(self):
        
        W_1 = []
        W_2 = []
        
        for layer in self.encoder1.children():
            # convolution filters
            if layer is nn.Conv2d:
                param = next(layer.parameters())
                if len(param.size()) == 4:
                    W_1.append((param**2).sum(dim=[1, 2, 3]))
        
        for layer in self.encoder2.children():
            # convolution filters
            if layer is nn.Conv2d:
                param = next(layer.parameters())
                if len(param.size()) == 4:
                    W_2.append((param**2).sum(dim=[1, 2, 3]))
                
        contractive_loss_1 = 0.0
        for w, latent in zip(W_1, self.latent1):
            dlatent = latent * (1 - latent)
            c = dlatent.size(1)
            contractive_loss_1 += torch.mm(torch.permute(dlatent**2, [0, 2, 3, 1]).view(-1, c), w)
                
        contractive_loss_2 = 0.0
        for w, latent in zip(W_2, self.latent2):
            dlatent = latent * (1 - latent)
            c = dlatent.size(1)
            contractive_loss_2 += torch.mm(torch.permute(dlatent**2, [0, 2, 3, 1]).view(-1, c), w)
            
        contractive_loss = contractive_loss_1 + contractive_loss_2
        
        return contractive_loss

import torch
import torch.nn as nn

from ai.EncoderLayer import EncoderLayer
from ai.DecoderLayer import DecoderLayer


class AutoEncoder(nn.Module):

    def __init__(self, index, is_train):
        super(AutoEncoder, self).__init__()
        
        self.index = index
        self.is_train = is_train

        self.encoders1 = nn.Sequential(
            EncoderLayer(6, 32, 5),
            EncoderLayer(32, 32, 5),
            EncoderLayer(32, 64, 5),
            EncoderLayer(64, 64, 5)
        )
        
        self.encoders2 = nn.Sequential(
            EncoderLayer(6, 32, 13),
            EncoderLayer(32, 32, 13),
            EncoderLayer(32, 64, 13),
            EncoderLayer(64, 64, 13)
        )
        
        self.decoders1 = nn.Sequential(
            DecoderLayer(64, 64, 5),
            DecoderLayer(64, 32, 5),
            DecoderLayer(32, 32, 5),
            DecoderLayer(32, 6, 5),
        )
                    
        self.decoders2 = nn.Sequential(
            DecoderLayer(64, 64, 13),
            DecoderLayer(64, 32, 13),
            DecoderLayer(32, 32, 13),
            DecoderLayer(32, 6, 13),
        )
        
        if self.index >= 0:
            for i, l in enumerate(self.encoders1.modules()):
                if self.is_train is False or self.index != i:
                    for param in l.parameters():
                        param.requires_grad_(False)
                        
            for i, l in enumerate(self.encoders2.modules()):
                if self.is_train is False or self.index != i:
                    for param in l.parameters():
                        param.requires_grad_(False)
                        
            for i, l in enumerate(self.decoders1.modules()):
                if self.is_train is False or 3-self.index != i:
                    for param in l.parameters():
                        param.requires_grad_(False)
                        
            for i, l in enumerate(self.decoders2.modules()):
                if self.is_train is False or 3-self.index != i:
                    for param in l.parameters():
                        param.requires_grad_(False)
        
        self.classifiers1_1 = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.Tanh(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(32, 16, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.Tanh(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )
        self.classifiers1_2 = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.Tanh(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(32, 16, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.Tanh(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )
        self.classifiers1_3 = nn.Sequential(
            nn.Conv2d(128, 64, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.Tanh(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(64, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.Tanh(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )
        self.classifiers1_4 = nn.Sequential(
            nn.Conv2d(128, 64, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.Tanh(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(64, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.Tanh(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),
        )
        
        self.classifiers2_1 = nn.Sequential(
            nn.Linear(16*16*16, 32),
            nn.Dropout(0.4),
            nn.Tanh(),
                    
            nn.Linear(32, 4),
            nn.LogSoftmax(dim=1)
        )
        self.classifiers2_2 = nn.Sequential(
            nn.Linear(8*8*16, 32),
            nn.Dropout(0.4),
            nn.Tanh(),
                    
            nn.Linear(32, 4),
            nn.LogSoftmax(dim=1)
        )
        self.classifiers2_3 = nn.Sequential(
            nn.Linear(4*4*32, 32),
            nn.Dropout(0.4),
            nn.Tanh(),
                    
            nn.Linear(32, 4),
            nn.LogSoftmax(dim=1)
        )
        self.classifiers2_4 = nn.Sequential(
            nn.Linear(2*2*32, 32),
            nn.Dropout(0.4),
            nn.Tanh(),
                    
            nn.Linear(32, 4),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x1 = x
        x2 = x
        
        x1_list = [x1]
        x2_list = [x2]

        for encoder1, encoder2 in zip(
            [l for i, l in enumerate(self.encoders1.children()) if i <= self.index],
            [l for i, l in enumerate(self.encoders2.children()) if i <= self.index]):
            x1 = encoder1(x1)
            x2 = encoder2(x2)
            
            x1_list.append(x1)
            x2_list.append(x2)
            
        latent = torch.cat([x1, x2], dim=1)

        if self.is_train is False:
            return latent
        else:
            x1_rec = list(self.decoders1.children())[3-self.index](x1)
            x2_rec = list(self.decoders2.children())[3-self.index](x2)
            
            if self.index == 0:
                x = self.classifiers1_1(latent)
                x = x.view(x.size(0), -1)
                x = self.classifiers2_1(x)
            elif self.index == 1:
                x = self.classifiers1_2(latent)
                x = x.view(x.size(0), -1)
                x = self.classifiers2_2(x)
            elif self.index == 2:
                x = self.classifiers1_3(latent)
                x = x.view(x.size(0), -1)
                x = self.classifiers2_3(x)
            elif self.index == 3:
                x = self.classifiers1_4(latent)
                x = x.view(x.size(0), -1)
                x = self.classifiers2_4(x)
            
            return x1_list[self.index], x2_list[self.index], x1_rec, x2_rec, x
    
    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        print("AutoEncoder was saved!")
    
    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print("AutoEncoder was saved!")
        
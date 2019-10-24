import torch
from torch import nn
from ai.FeatureVGG import FeatureVGG
import numpy as np


class ClassifierVGG(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = FeatureVGG()
        self.features.requires_grad_(False)

        # self.lstm = nn.LSTM(
        #     4*4*1024,
        #     32,
        #     2,
        #     batch_first=True,
        #     bidirectional=True,
        #     dropout=0.4
        # )

        self.conv = nn.Sequential(
            nn.Conv1d(8, 32, (121,), stride=32, padding=0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32*8*2, 32),
            nn.Tanh(),
            nn.Dropout(0.4),

            nn.Linear(32, 4),
            nn.LogSoftmax(dim=-1)
        )

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        print("Classifier VGG was saved.")

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print("Classifier VGG was loaded.")

    def forward(self, x):

        x = self.features(x)

        res_out, res_hidden = self.lstm(x)
        res_out = res_out.contiguous()
        res_out = res_out.view(-1, 2 * 8 * 32)

        x = self.classifier(res_out)

        return x

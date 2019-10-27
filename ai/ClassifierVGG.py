import torch
from torch import nn
from ai.FeatureVGG import FeatureVGG
import numpy as np
from env import *


class ClassifierVGG(nn.Module):

    def __init__(self, fine_tune=False):
        super().__init__()

        self.features = FeatureVGG()
        self.features.requires_grad_(fine_tune)
        self.features.load(VGG_CKPT_PATH)
        self.features.eval()
        self.fine_tune = fine_tune

        # self.lstm = nn.LSTM(
        #     4*4*1024,
        #     32,
        #     2,
        #     batch_first=True,
        #     bidirectional=True,
        #     dropout=0.4
        # )

        self.conv = nn.Sequential(
            nn.Conv1d(8, 32, (121,), stride=16, padding=0), # 1017
            nn.BatchNorm1d(32),
            nn.Tanh(),

            nn.Conv1d(32, 64, (55,), stride=8, padding=0), # 121
            nn.BatchNorm1d(64),
            nn.Tanh(),

            nn.Conv1d(64, 128, (13,), stride=2, padding=0), # 55
            nn.BatchNorm1d(128),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(55*128, 32),
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
        if self.fine_tune is False:
            self.features.eval()

        x, _, _, _ = self.features(x)

        # res_out, res_hidden = self.lstm(x)
        # res_out = res_out.contiguous()
        # res_out = res_out.view(-1, 2 * 8 * 32)

        # x = self.classifier(res_out)

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

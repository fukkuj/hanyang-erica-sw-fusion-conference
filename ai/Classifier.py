import torch
from torch import nn
import numpy as np
import sys
from ai.FeatureCNN import FeatureCNN
from ai.ClassifierRNN import ClassifierRNN
from env import *


class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_classes = 4

        self.input_size = 256
        self.hidden_size = 32
        self.num_layers = 1
        self.drop_rate = 0.4
        # self.hidden = None

        self.features = FeatureCNN()
        self.features.load(CNN_CKPT_PATH)
        for param in self.features.parameters():
            param.requires_grad_(False)
        self.features.eval()

        self.classifier = ClassifierRNN()

    def save(self, ckpt):
        model = {
            # "short_term": self.hidden[0],
            # "long_term": self.hidden[1],
            "state_dict": self.state_dict()
        }
        torch.save(model, ckpt)
        print("Classifier was saved.")

    def load(self, ckpt):
        model = torch.load(ckpt)
        # self.hidden = (model["short_term"], model["long_term"])
        self.load_state_dict(model["state_dict"])
        print("Classifier was loaded.")

    def forward(self, x):
        self.features.eval()

        n_b = x.size(0)
        n_s = x.size(1)
        n = n_b * n_s

        x = x.view(n, 3, 128, 128)

        # print(self.features(x))
        x, _ = self.features(x)
        x = x.view(n_b, n_s, -1)

        # print(x.size())

        x, hidden = self.classifier(x, None)
        # self.hidden = (hidden[0].data, hidden[1].data)

        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            ps = torch.exp(x)
            print(ps)
            cls_ps, top_k = ps.topk(1, dim=1)
            return top_k.squeeze().data


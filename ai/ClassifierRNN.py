import torch
from torch import nn
import numpy as np


class ClassifierRNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_classes = 4
        self.num_step = 8

        self.hidden_size = 64

        self.lstm = nn.LSTM(64*8*8,
                            hidden_size,
                            2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.4)

        self.classifier = nn.Sequential(
            nn.Linear(2 * num_step * hidden_size, 256),
            nn.Tanh(),
            nn.Dropout(drop_rate),

            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Dropout(drop_rate),

            nn.Linear(64, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def save(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        print("Classifier RNN was saved.")

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print("Classifier RNN was loaded.")

    def forward(self, x, hidden):
        assert(x.shape[1] == self.num_step)

        res_out, res_hidden = self.lstm(x, hidden)
        res_out = res_out.contiguous()
        res_out = res_out.view(-1, 2 * self.num_step * self.hidden_size)

        x = self.classifier(res_out)

        return x, res_hidden
import torch
from torch import nn


class EncoderLayer(nn.Module):

    def __init__(self, in_ch, out_ch, f):
        super(EncoderLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (f, f), stride=1, padding=f//2),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),

            nn.Conv2d(out_ch, out_ch, (f, f), stride=2, padding=f//2),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x
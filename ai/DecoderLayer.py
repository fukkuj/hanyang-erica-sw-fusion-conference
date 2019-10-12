import torch
from torch import nn


class DecoderLayer(nn.Module):

    def __init__(self, in_ch, out_ch, f):
        super(DecoderLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, (f, f), stride=1, padding=f//2, output_padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Tanh(),

            nn.ConvTranspose2d(out_ch, out_ch, (f, f), stride=2, padding=f//2, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer(x)

        return x
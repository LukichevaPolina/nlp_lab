import torch
from torch import nn, Tensor

class LinPsychoNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self._global_statistic = nn.AdaptiveAvgPool2d(1)
        self._downsample = nn.AdaptiveAvgPool2d((1, 1024))

        self._layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 32),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self._last = nn.Linear(17, out_channels)
    
    def forward(self, x: Tensor):
        glob_avg = self._global_statistic(x)

        downsampled_x = self._downsample(x)
        out_layer = self._layer(downsampled_x)

        concat_x = torch.concat((glob_avg, out_layer), dim=3)

        out = self._last(concat_x)

        return out

    @staticmethod
    def initialize(net) -> None:
        if isinstance(net, nn.Linear):
            if net.weight is not None:
                nn.init.xavier_uniform_(net.weight.data, gain=0.02)
            if net.bias is not None:
                nn.init.constant_(net.bias.data, 0.0)


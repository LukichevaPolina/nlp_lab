from torch import Tensor, nn

from src.models.cnn.layers.conv import conv1x1, conv3x3
from src.models.cnn.layers.simple_gate import gate

class ConvPsychoNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self._first_conv = conv3x3(
            in_channels, 
            in_channels, 
            act=nn.LeakyReLU(negative_slope=0.02, inplace=True),
            norm=None,
            bias=True
        )

        # TODO: before main_layer resize
        self._layer1 = nn.Sequential(
            conv3x3(in_channels, 4, act=nn.LeakyReLU(negative_slope=0.02, inplace=True), norm=nn.BatchNorm2d(4), bias=False),        
            conv3x3(4, 16, act=nn.LeakyReLU(negative_slope=0.02, inplace=True), norm=nn.BatchNorm2d(16), bias=False),
        )

        self._gated = gate(966, 64, 966)

        self._layer2 = nn.Sequential(
            conv3x3(16, 32, act=nn.LeakyReLU(negative_slope=0.02, inplace=True), norm=nn.BatchNorm2d(32), bias=False),        
            conv3x3(32, 64, act=nn.LeakyReLU(negative_slope=0.02, inplace=True), norm=nn.BatchNorm2d(64), bias=False),
        )

        self._projection = conv1x1(64, 1, act=nn.LeakyReLU(negative_slope=0.02, inplace=True), norm=nn.BatchNorm2d(1), bias=False)

        self._last_layer = nn.Linear(528, out_channels)
    
    def forward(self, x):
        out_first_conv = self._first_conv(x)
        #print(f"{out_first_conv.shape=}")

        resized_out = out_first_conv.reshape(out_first_conv.shape[0], out_first_conv.shape[1], -1, 89)
        #print(f"{resized_out.shape=}")

        out_layer1 = self._layer1(resized_out)
        #print(f"{out_layer1.shape=}")

        gated_out = self._gated(out_layer1.reshape(out_layer1.shape[0], out_layer1.shape[1], 1, -1))
        #print(f"{gated_out.shape=}")

        gated_out = gated_out.reshape_as(out_layer1)
        #print(f"{gated_out.shape=}")

        out_layer2 = self._layer2(gated_out)
        #print(f"{out_layer2.shape=}")

        out_projection = self._projection(out_layer2)
        #print(f"{out_projection.shape=}")

        out_last = self._last_layer(out_projection.flatten())
        #print(f"{out_last.shape=}")

        return out_last

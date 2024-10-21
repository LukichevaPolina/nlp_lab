from torch import Tensor, nn

def conv3x3(in_channels, out_channels, act=None, norm=None, bias=False) -> nn.Conv2d:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias),
        nn.BatchNorm2d(out_channels) if norm is not None else nn.Identity(),
        nn.LeakyReLU(negative_slope=0.02, inplace=True) if act is not None else nn.Identity()
    )

def conv1x1(in_channels, out_channels, act=None, norm=None, bias=False) -> nn.Conv2d:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        norm if norm is not None else nn.Identity(),
        act if act is not None else nn.Identity()
    )
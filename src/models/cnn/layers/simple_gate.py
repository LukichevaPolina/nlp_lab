from torch import Tensor, nn

def gate(in_channels, hidden_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, hidden_channels),
        nn.Linear(hidden_channels, out_channels),
    )
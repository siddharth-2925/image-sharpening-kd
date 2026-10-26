"""A small residual CNN that accepts and returns full-resolution RGB frames."""

import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(self.activation(self.depthwise(x)))


class StudentNet(nn.Module):
    """Process features at half resolution and predict a full-resolution correction.

    Even height and width are required. The residual connection means an untrained
    model starts at the degraded input, which is a useful controlled baseline.
    """

    def __init__(self, channels: int = 8, blocks: int = 3):
        super().__init__()
        self.channels = channels
        self.blocks_count = blocks
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*(DepthwiseBlock(channels) for _ in range(blocks)))
        self.reconstruction = nn.Conv2d(channels, 12, 1)
        self.shuffle = nn.PixelShuffle(2)
        nn.init.zeros_(self.reconstruction.weight)
        nn.init.zeros_(self.reconstruction.bias)

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        # These shapes are fixed to even dimensions by the caller. A fixed 2x
        # scale keeps the exported ONNX graph simple for OpenVINO.
        low = F.interpolate(degraded, scale_factor=0.5, mode="bilinear", align_corners=False)
        features = self.blocks(self.stem(low))
        correction = self.shuffle(self.reconstruction(features))
        return degraded + correction


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


"""Trainable semantic fusion of frozen SAM embeddings and local RGB detail."""
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, outgoing), nn.GELU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing), nn.GELU())


class SemanticDecoder(nn.Module):
    """GroupNorm supports small batches; all spatial fusion is learned.

    RGB detail features at 1/2, 1/4, 1/8, and 1/16 resolution are combined
    with aligned SAM semantics. SAM's prompt decoder is not used in this mode.
    """
    def __init__(self, classes, width=64):
        super().__init__()
        if width < 16 or width % 8:
            raise ValueError("decoder_width must be a multiple of 8 and >=16")
        self.detail = nn.ModuleList([
            ConvBlock(3, width, 2), ConvBlock(width, width, 2),
            ConvBlock(width, width * 2, 2), ConvBlock(width * 2, width * 2, 2)])
        self.semantic = ConvBlock(256, width * 2)
        self.fuse = nn.ModuleList([
            ConvBlock(width * 4, width * 2), ConvBlock(width * 4, width * 2),
            ConvBlock(width * 3, width), ConvBlock(width * 2, width)])
        self.head = nn.Conv2d(width, classes, 1)

    def forward(self, embeddings, image):
        features = []
        x = image * 2 - 1
        for block in self.detail:
            x = block(x)
            features.append(x)
        x = self.semantic(embeddings)
        for block, skip in zip(self.fuse, reversed(features)):
            x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return F.interpolate(self.head(x), image.shape[-2:], mode="bilinear", align_corners=False)

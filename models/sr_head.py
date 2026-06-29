# -*- coding: utf-8 -*-
"""Super-resolution output head for HDR feature maps."""

import torch.nn as nn


class SRHead(nn.Module):
    """A lightweight 2x super-resolution head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        scale (int): Upsampling factor. The default architecture is designed
            for ``scale=2`` and uses PixelShuffle(2).

    Shape:
        Input:  [B, C, H, W]
        Output: [B, 3, 2H, 2W] when ``scale=2``.
    """

    def __init__(self, in_channels, scale=2):
        super(SRHead, self).__init__()
        if scale != 2:
            raise ValueError("SRHead currently supports only scale=2.")

        self.scale = scale
        self.body = nn.Sequential(
            # Expand channels from C to C * 4 so PixelShuffle(2) can convert
            # channel information into a 2x larger spatial feature map.
            nn.Conv2d(in_channels, in_channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
            # Project the upsampled feature map back to a 3-channel HDR image.
            nn.Conv2d(in_channels, 3, 3, 1, 1),
            # Keep the output range aligned with the original HDRTransformer
            # output head, which applies sigmoid after the last convolution.
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Upsample an HDR feature map from [B, C, H, W] to [B, 3, 2H, 2W]."""
        return self.body(x)

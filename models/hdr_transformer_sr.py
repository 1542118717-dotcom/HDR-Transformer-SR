# -*- coding: utf-8 -*-
"""HDR-Transformer with a super-resolution output head."""

import torch

from models.hdr_transformer import HDRTransformer
from models.sr_head import SRHead


class HDRTransformerSR(HDRTransformer):
    """HDRTransformer variant that predicts a 2x super-resolved HDR image.

    This class reuses the original HDRTransformer feature extraction and
    Context-aware Transformer reconstruction backbone. The only architectural
    change is replacing the same-resolution ``conv_last + sigmoid`` image head
    with ``SRHead``, which upsamples the reconstructed feature map by ``scale``.
    """

    def __init__(self, scale=2, *args, **kwargs):
        super(HDRTransformerSR, self).__init__(*args, **kwargs)
        self.scale = scale
        # Replace the original same-resolution RGB output convolution with a
        # pixel-shuffle super-resolution head. The original HDRTransformer class
        # remains untouched in models/hdr_transformer.py.
        self.sr_head = SRHead(self.embed_dim, scale=scale)

    def forward(self, x1, x2, x3):
        """Forward pass with the same input interface as HDRTransformer.

        Args:
            x1, x2, x3: LDR/HDR-aligned inputs with shape [B, 6, H, W].

        Returns:
            Super-resolved HDR image with shape [B, 3, 2H, 2W] for scale=2.
        """
        # Coarse feature extraction for the three input frames/exposures.
        f1 = self.conv_f1(x1)
        f2 = self.conv_f2(x2)
        f3 = self.conv_f3(x3)

        # Spatial attention follows the original HDRTransformer fusion path.
        f1_att_m = self.att_module_h(f1, f2)
        f1_att = f1 * f1_att_m
        f3_att_m = self.att_module_l(f3, f2)
        f3_att = f3 * f3_att_m
        x = self.conv_first(torch.cat((f1_att, f2, f3_att), dim=1))

        # Reuse the original HDR reconstruction backbone to produce features at
        # input resolution, then send those features to the SR output head.
        res = self.conv_after_body(self.forward_features(x) + x)
        return self.sr_head(f2 + res)

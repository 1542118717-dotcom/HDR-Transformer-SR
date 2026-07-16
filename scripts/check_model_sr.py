# -*- coding: utf-8 -*-
"""Random tensor shape and HDR radiance range test for HDRTransformerSR."""

import os
import sys

import torch
import torch.nn as nn

# Allow running this script directly from the repository root, e.g.:
#   python scripts/check_model_sr.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.hdr_transformer_sr import HDRTransformerSR


def main():
    torch.manual_seed(0)

    x1 = torch.randn(1, 6, 64, 64)
    x2 = torch.randn(1, 6, 64, 64)
    x3 = torch.randn(1, 6, 64, 64)

    model = HDRTransformerSR(
        embed_dim=60,
        depths=[6, 6, 6],
        num_heads=[6, 6, 6],
        mlp_ratio=2,
        in_chans=6,
        scale=2,
    )
    model.eval()

    final_activation = model.sr_head.body[-1]
    if not isinstance(final_activation, nn.Softplus):
        raise AssertionError(
            "Expected HDRTransformerSR SR head final activation to be Softplus, "
            "got {}".format(final_activation.__class__.__name__)
        )

    # Make the range check deterministic: with zero final-convolution weights
    # and a positive bias, Softplus can produce values above 1, whereas a
    # sigmoid head would remain capped at 1.
    final_conv = model.sr_head.body[-2]
    nn.init.zeros_(final_conv.weight)
    nn.init.constant_(final_conv.bias, 2.0)

    with torch.no_grad():
        y = model(x1, x2, x3)

    print("x1 shape:", list(x1.shape))
    print("x2 shape:", list(x2.shape))
    print("x3 shape:", list(x3.shape))
    print("output shape:", list(y.shape))
    print("output min:", float(y.min()))
    print("output max:", float(y.max()))

    expected_shape = [1, 3, 128, 128]
    if list(y.shape) != expected_shape:
        raise AssertionError(
            "Expected output shape {}, got {}".format(expected_shape, list(y.shape))
        )

    if float(y.min()) < 0.0:
        raise AssertionError(
            "Expected non-negative HDR output, got minimum {}".format(float(y.min()))
        )

    if float(y.max()) <= 1.0:
        raise AssertionError(
            "Expected HDR output not to be capped at 1, got maximum {}".format(float(y.max()))
        )

    print("HDRTransformerSR shape and HDR radiance range test passed.")


if __name__ == "__main__":
    main()

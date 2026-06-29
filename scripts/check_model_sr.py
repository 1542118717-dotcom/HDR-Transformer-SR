# -*- coding: utf-8 -*-
"""Random tensor shape test for HDRTransformerSR."""

import os
import sys

import torch

# Allow running this script directly from the repository root, e.g.:
#   python scripts/check_model_sr.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.hdr_transformer_sr import HDRTransformerSR


def main():
    x1 = torch.randn(1, 6, 128, 128)
    x2 = torch.randn(1, 6, 128, 128)
    x3 = torch.randn(1, 6, 128, 128)

    model = HDRTransformerSR(
        embed_dim=60,
        depths=[6, 6, 6],
        num_heads=[6, 6, 6],
        mlp_ratio=2,
        in_chans=6,
        scale=2,
    )
    model.eval()

    with torch.no_grad():
        y = model(x1, x2, x3)

    print("x1 shape:", list(x1.shape))
    print("x2 shape:", list(x2.shape))
    print("x3 shape:", list(x3.shape))
    print("output shape:", list(y.shape))

    expected_shape = [1, 3, 256, 256]
    if list(y.shape) != expected_shape:
        raise AssertionError("Expected output shape {}, got {}".format(expected_shape, list(y.shape)))

    print("HDRTransformerSR shape test passed.")


if __name__ == "__main__":
    main()

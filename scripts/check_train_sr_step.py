# -*- coding: utf-8 -*-
"""Quick shape check for one HDR-Transformer-SR training step."""

import argparse
import os
import sys

import torch

# Allow running this script directly from the repository root, e.g.:
#   python scripts/check_train_sr_step.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.dataset_sig17_sr import SIG17_SR_Training_Dataset
from models.hdr_transformer_sr import HDRTransformerSR


def parse_args():
    parser = argparse.ArgumentParser(description='Check one HDR-SR train-step forward shape.')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Optional SIG17/Kalantari17 dataset root. If omitted, use synthetic tensors.')
    parser.add_argument('--sub_set', type=str, default='sig17_training_crop128_stride64',
                        help='Training subset directory when --dataset_dir is provided.')
    parser.add_argument('--scale', type=int, default=2, help='Super-resolution scale factor.')
    parser.add_argument('--height', type=int, default=64, help='Synthetic LR input height.')
    parser.add_argument('--width', type=int, default=64, help='Synthetic LR input width.')
    parser.add_argument('--embed_dim', type=int, default=60, help='Model embedding dimension.')
    parser.add_argument('--quick', action='store_true',
                        help='Use shallow transformer depths [1, 1, 1] for faster synthetic checks.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA.')
    return parser.parse_args()


def make_batch(args):
    if args.dataset_dir is not None:
        dataset = SIG17_SR_Training_Dataset(
            root_dir=args.dataset_dir,
            sub_set=args.sub_set,
            is_training=True,
            scale=args.scale,
        )
        sample = dataset[0]
        return {key: value.unsqueeze(0) for key, value in sample.items()}

    lr_h, lr_w = args.height, args.width
    hr_h, hr_w = lr_h * args.scale, lr_w * args.scale
    return {
        'input0': torch.randn(1, 6, lr_h, lr_w),
        'input1': torch.randn(1, 6, lr_h, lr_w),
        'input2': torch.randn(1, 6, lr_h, lr_w),
        'label': torch.rand(1, 3, hr_h, hr_w),
    }


def main():
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    depths = [1, 1, 1] if args.quick else [6, 6, 6]
    model = HDRTransformerSR(
        embed_dim=args.embed_dim,
        depths=depths,
        num_heads=[6, 6, 6],
        mlp_ratio=2,
        in_chans=6,
        scale=args.scale,
    ).to(device)
    model.eval()

    batch = make_batch(args)
    input0 = batch['input0'].to(device)
    input1 = batch['input1'].to(device)
    input2 = batch['input2'].to(device)
    label = batch['label'].to(device)

    with torch.no_grad():
        pred = model(input0, input1, input2)

    print('input0 shape:', list(input0.shape))
    print('input1 shape:', list(input1.shape))
    print('input2 shape:', list(input2.shape))
    print('label shape:', list(label.shape))
    print('output shape:', list(pred.shape))

    if pred.shape != label.shape:
        raise AssertionError('Expected output shape {}, got {}'.format(list(label.shape), list(pred.shape)))

    print('HDR-SR train step shape test passed.')


if __name__ == '__main__':
    main()

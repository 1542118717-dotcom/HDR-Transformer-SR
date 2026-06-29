# -*- coding: utf-8 -*-
"""Quick train-step shape check for HDRTransformerSR."""

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.dataset_sig17_sr import SIG17_SR_Training_Dataset
from models.hdr_transformer_sr import HDRTransformerSR
from train_sr import crop_sr_batch_to_window


def parse_args():
    parser = argparse.ArgumentParser(description='Check one HDR-SR training forward shape.')
    parser.add_argument('--quick', action='store_true', help='use synthetic tensors instead of real SIG17 files')
    parser.add_argument('--height', type=int, default=64, help='synthetic LR input height')
    parser.add_argument('--width', type=int, default=64, help='synthetic LR input width')
    parser.add_argument('--scale', type=int, default=2, help='super-resolution scale factor')
    parser.add_argument('--window_size', type=int, default=8, help='HDRTransformer window size for LR input cropping')
    parser.add_argument('--dataset_dir', type=str, default='./data', help='SIG17/Kalantari17 dataset root directory')
    parser.add_argument('--sub_set', type=str, default='sig17_training_crop128_stride64', help='training subset directory')
    parser.add_argument('--index', type=int, default=0, help='dataset sample index for non-quick mode')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA')
    return parser.parse_args()


def build_model(scale, device):
    model = HDRTransformerSR(
        embed_dim=60,
        depths=[6, 6, 6],
        num_heads=[6, 6, 6],
        mlp_ratio=2,
        in_chans=6,
        scale=scale,
    )
    model.to(device)
    model.eval()
    return model


def synthetic_batch(args, device):
    input0 = torch.randn(1, 6, args.height, args.width, device=device)
    input1 = torch.randn(1, 6, args.height, args.width, device=device)
    input2 = torch.randn(1, 6, args.height, args.width, device=device)
    label = torch.randn(1, 3, args.height * args.scale, args.width * args.scale, device=device)
    return input0, input1, input2, label


def dataset_batch(args, device):
    dataset = SIG17_SR_Training_Dataset(
        root_dir=args.dataset_dir,
        sub_set=args.sub_set,
        is_training=True,
        scale=args.scale,
    )
    sample = dataset[args.index]
    input0 = sample['input0'].unsqueeze(0).to(device)
    input1 = sample['input1'].unsqueeze(0).to(device)
    input2 = sample['input2'].unsqueeze(0).to(device)
    label = sample['label'].unsqueeze(0).to(device)
    return input0, input1, input2, label


def main():
    args = parse_args()
    if not args.quick and not os.path.isdir(args.dataset_dir):
        raise FileNotFoundError('Dataset directory not found: {}'.format(args.dataset_dir))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = build_model(args.scale, device)
    if args.quick:
        input0, input1, input2, label = synthetic_batch(args, device)
    else:
        input0, input1, input2, label = dataset_batch(args, device)

    input0, input1, input2, label = crop_sr_batch_to_window(
        input0, input1, input2, label, window_size=args.window_size, scale=args.scale
    )

    with torch.no_grad():
        output = model(input0, input1, input2)

    if output.shape != label.shape:
        raise AssertionError('Expected output shape {}, got {}'.format(tuple(label.shape), tuple(output.shape)))

    print('HDR-SR train step shape test passed.')


if __name__ == '__main__':
    main()

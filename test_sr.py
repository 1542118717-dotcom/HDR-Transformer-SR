# -*- coding: utf-8 -*-
"""Independent HDR-Transformer-SR inference script for SIG17/Kalantari data."""

import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.dataset_sig17_sr import SIG17_SR_Validation_Dataset
from models.hdr_transformer_sr import HDRTransformerSR
from train_sr import crop_sr_batch_to_window
from utils.utils import range_compressor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run HDR-Transformer-SR inference on SIG17/Kalantari validation samples.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset_dir', type=str, default='./data', help='SIG17/Kalantari17 dataset root directory')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints_sr/hdr_sr_best_checkpoint.pth', help='HDR-SR checkpoint path')
    parser.add_argument('--save_dir', type=str, default='./results/hdr_transformer_sr', help='directory for .npy and PNG outputs')
    parser.add_argument('--scale', type=int, default=2, help='super-resolution scale factor')
    parser.add_argument('--window_size', type=int, default=8, help='HDRTransformer window size for LR input cropping')
    parser.add_argument('--val_lr_patch_size', type=int, default=64, help='LR patch size for validation/inference crops; <=0 uses largest aligned crop')
    parser.add_argument('--index', type=int, default=None, help='sample index to run; omit to run all validation samples')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='force CPU inference')
    return parser.parse_args()


def get_device(no_cuda=False):
    if not no_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_model(scale, device):
    model = HDRTransformerSR(
        embed_dim=60,
        depths=[6, 6, 6],
        num_heads=[6, 6, 6],
        mlp_ratio=2,
        in_chans=6,
        scale=scale,
    ).to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def normalize_state_dict_for_model(state_dict, model):
    model_is_dp = isinstance(model, nn.DataParallel)
    normalized = {}
    for key, value in state_dict.items():
        if model_is_dp and not key.startswith('module.'):
            normalized['module.' + key] = value
        elif (not model_is_dp) and key.startswith('module.'):
            normalized[key[7:]] = value
        else:
            normalized[key] = value
    return normalized


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(normalize_state_dict_for_model(state_dict, model), strict=True)
    model.eval()
    return model


def tensor_shape(tensor):
    return tuple(tensor.shape)


def tonemap_to_uint8(chw_hdr):
    """Convert a CHW HDR prediction to a viewable HWC uint8 image."""
    hwc = np.transpose(chw_hdr, (1, 2, 0))
    mapped = range_compressor(np.maximum(hwc, 0.0))
    return np.clip(mapped * 255.0, 0.0, 255.0).astype(np.uint8)


def save_outputs(save_dir, sample_index, output):
    os.makedirs(save_dir, exist_ok=True)
    output_np = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
    npy_path = osp.join(save_dir, 'sample_{:04d}_output.npy'.format(sample_index))
    png_path = osp.join(save_dir, 'sample_{:04d}_output_tonemap.png'.format(sample_index))
    np.save(npy_path, output_np)
    png_rgb = tonemap_to_uint8(output_np)
    cv2.imwrite(png_path, cv2.cvtColor(png_rgb, cv2.COLOR_RGB2BGR))
    return npy_path, png_path


def prepare_sample(dataset, index, args, device):
    sample = dataset[index]
    input0 = sample['input0'].unsqueeze(0).to(device)
    input1 = sample['input1'].unsqueeze(0).to(device)
    input2 = sample['input2'].unsqueeze(0).to(device)
    label = sample['label'].unsqueeze(0).to(device)
    patch_size = args.val_lr_patch_size if args.val_lr_patch_size > 0 else None
    return crop_sr_batch_to_window(
        input0,
        input1,
        input2,
        label,
        window_size=args.window_size,
        scale=args.scale,
        patch_size=patch_size,
        random_crop=False,
    )


def run_inference(args):
    if not osp.isdir(args.dataset_dir):
        raise FileNotFoundError('Dataset directory not found: {}'.format(args.dataset_dir))
    if not osp.isfile(args.checkpoint):
        raise FileNotFoundError('Checkpoint not found: {}'.format(args.checkpoint))

    device = get_device(args.no_cuda)
    print('Using device: {}'.format(device))
    model = load_checkpoint(build_model(args.scale, device), args.checkpoint, device)
    dataset = SIG17_SR_Validation_Dataset(
        root_dir=args.dataset_dir,
        is_training=False,
        crop=True,
        crop_size=max(args.val_lr_patch_size, args.window_size) * args.scale if args.val_lr_patch_size > 0 else 512,
        scale=args.scale,
    )

    if args.index is None:
        indices = range(len(dataset))
    else:
        if args.index < 0 or args.index >= len(dataset):
            raise IndexError('index {} out of range for dataset length {}'.format(args.index, len(dataset)))
        indices = [args.index]

    with torch.no_grad():
        for sample_index in indices:
            input0, input1, input2, label = prepare_sample(dataset, sample_index, args, device)
            output = model(input0, input1, input2)
            print(
                'sample {} shapes: input0={} input1={} input2={} label={} output={}'.format(
                    sample_index,
                    tensor_shape(input0),
                    tensor_shape(input1),
                    tensor_shape(input2),
                    tensor_shape(label),
                    tensor_shape(output),
                )
            )
            if output.shape != label.shape:
                raise RuntimeError(
                    'Output shape {} does not match label shape {} for sample {}'.format(
                        tuple(output.shape), tuple(label.shape), sample_index
                    )
                )
            npy_path, png_path = save_outputs(args.save_dir, sample_index, output)
            print('Saved: {} and {}'.format(npy_path, png_path))


def main():
    run_inference(parse_args())


if __name__ == '__main__':
    main()

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

from dataset.dataset_sig17_sr import SIG17_SR_Training_Dataset, SIG17_SR_Validation_Dataset
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
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help='device to use for inference; auto prefers cuda, then mps, then cpu')
    parser.add_argument('--sub_set', type=str, default='Training', help='dataset subset used with --val_scene_list_file')
    parser.add_argument('--val_scene_list_file', type=str, default=None, help='optional fixed validation scene list file from --sub_set')
    return parser.parse_args()


def resolve_device(args):
    if args.no_cuda:
        return torch.device('cpu')

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device was requested with --device cuda, but torch.cuda.is_available() is False.')
        return torch.device('cuda')

    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS device was requested with --device mps, but torch.backends.mps.is_available() is False.')
        return torch.device('mps')

    if args.device == 'cpu':
        return torch.device('cpu')

    if args.device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    raise ValueError('Unsupported device option: {}'.format(args.device))


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


def build_dataset(args):
    if args.val_scene_list_file is not None:
        return SIG17_SR_Training_Dataset(
            root_dir=args.dataset_dir,
            sub_set=args.sub_set,
            is_training=False,
            scale=args.scale,
            scene_list_file=args.val_scene_list_file,
        )
    return SIG17_SR_Validation_Dataset(
        root_dir=args.dataset_dir,
        is_training=False,
        crop=True,
        crop_size=max(args.val_lr_patch_size, args.window_size) * args.scale if args.val_lr_patch_size > 0 else 512,
        scale=args.scale,
    )


def get_scene_name(dataset, index):
    if hasattr(dataset, 'scenes_list') and index < len(dataset.scenes_list):
        return dataset.scenes_list[index]
    return 'unknown'


def tensor_shape(tensor):
    return tuple(tensor.shape)


def to_chw_numpy(tensor):
    """Convert a BCHW tensor with batch size 1 to a CHW float32 numpy array."""
    chw = tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if chw.ndim != 3:
        raise ValueError('Expected CHW tensor after squeezing batch dimension, got shape {}'.format(chw.shape))
    return chw


def first_three_channels(chw_image, image_name):
    """Return the first three BGR channels from a CHW image."""
    if chw_image.shape[0] < 3:
        raise ValueError('{} must have at least 3 channels for visualization, got shape {}'.format(image_name, chw_image.shape))
    return chw_image[:3]


def chw_bgr_to_hwc_bgr(chw_bgr, image_name):
    bgr = first_three_channels(chw_bgr, image_name)
    return np.transpose(bgr, (1, 2, 0))


def normalize_tonemapped_bgr_to_uint8(hwc_bgr, lo, hi):
    if hi <= lo:
        normalized = np.zeros_like(hwc_bgr, dtype=np.float32)
    else:
        normalized = (hwc_bgr - lo) / (hi - lo)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def hdr_chw_bgr_to_tonemapped_hwc_bgr(chw_hdr_bgr):
    """Convert a CHW HDR BGR image to a tone-mapped HWC BGR float image."""
    hwc_bgr = chw_bgr_to_hwc_bgr(chw_hdr_bgr, 'HDR tone-map input')
    return range_compressor(np.maximum(hwc_bgr, 0.0)).astype(np.float32)


def hdr_pair_to_uint8_bgr(output_chw_bgr, label_chw_bgr):
    """Tone-map output/label HDR BGR with shared label-derived percentile scaling."""
    output_tonemapped_bgr = hdr_chw_bgr_to_tonemapped_hwc_bgr(output_chw_bgr)
    label_tonemapped_bgr = hdr_chw_bgr_to_tonemapped_hwc_bgr(label_chw_bgr)
    lo, hi = np.percentile(label_tonemapped_bgr, [0.5, 99.5])
    output_bgr = normalize_tonemapped_bgr_to_uint8(output_tonemapped_bgr, lo, hi)
    label_bgr = normalize_tonemapped_bgr_to_uint8(label_tonemapped_bgr, lo, hi)
    return output_bgr, label_bgr


def input1_to_uint8_bgr(input1_chw):
    """Visualize input1 using original LDR BGR channels 3:6."""
    if input1_chw.shape[0] < 6:
        raise ValueError('input1 must have 6 channels, got shape {}'.format(input1_chw.shape))
    input1_hwc_bgr = np.transpose(input1_chw[3:6], (1, 2, 0))
    return (np.clip(input1_hwc_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_bgr_png(path, bgr_image):
    cv2.imwrite(path, bgr_image)


def resize_bgr_to_shape(bgr_image, target_height, target_width):
    """Resize an HWC BGR image for side-by-side visualization when needed."""
    if bgr_image.shape[:2] == (target_height, target_width):
        return bgr_image
    return cv2.resize(bgr_image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def save_outputs(save_dir, sample_index, input1, output, label):
    os.makedirs(save_dir, exist_ok=True)
    input1_np = to_chw_numpy(input1)
    output_np = to_chw_numpy(output)
    label_np = to_chw_numpy(label)

    if output_np.shape != label_np.shape:
        raise RuntimeError(
            'Output shape {} does not match label shape {} for sample {}'.format(
                output_np.shape, label_np.shape, sample_index
            )
        )

    npy_path = osp.join(save_dir, 'sample_{:04d}_output.npy'.format(sample_index))
    output_png_path = osp.join(save_dir, 'sample_{:04d}_output_tonemap.png'.format(sample_index))
    label_png_path = osp.join(save_dir, 'sample_{:04d}_label_tonemap.png'.format(sample_index))
    input1_png_path = osp.join(save_dir, 'sample_{:04d}_input1_tonemap.png'.format(sample_index))
    comparison_png_path = osp.join(save_dir, 'sample_{:04d}_comparison.png'.format(sample_index))

    np.save(npy_path, output_np)
    input1_bgr = input1_to_uint8_bgr(input1_np)
    output_bgr, label_bgr = hdr_pair_to_uint8_bgr(output_np, label_np)
    input1_comparison_bgr = resize_bgr_to_shape(input1_bgr, output_bgr.shape[0], output_bgr.shape[1])
    comparison_bgr = np.concatenate([input1_comparison_bgr, output_bgr, label_bgr], axis=1)

    save_bgr_png(output_png_path, output_bgr)
    save_bgr_png(label_png_path, label_bgr)
    save_bgr_png(input1_png_path, input1_bgr)
    save_bgr_png(comparison_png_path, comparison_bgr)
    return npy_path, output_png_path, label_png_path, input1_png_path, comparison_png_path


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

    device = resolve_device(args)
    print('Using device: {}'.format(device))
    model = load_checkpoint(build_model(args.scale, device), args.checkpoint, device)
    dataset = build_dataset(args)

    if args.index is None:
        indices = range(len(dataset))
    else:
        if args.index < 0 or args.index >= len(dataset):
            raise IndexError('index {} out of range for dataset length {}'.format(args.index, len(dataset)))
        indices = [args.index]

    with torch.no_grad():
        for sample_index in indices:
            scene_name = get_scene_name(dataset, sample_index)
            input0, input1, input2, label = prepare_sample(dataset, sample_index, args, device)
            output = model(input0, input1, input2)
            print(
                'sample {} scene {} shapes: input0={} input1={} input2={} label={} output={}'.format(
                    sample_index,
                    scene_name,
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
            saved_paths = save_outputs(args.save_dir, sample_index, input1, output, label)
            print('Saved: {}'.format(', '.join(saved_paths)))


def main():
    run_inference(parse_args())


if __name__ == '__main__':
    main()

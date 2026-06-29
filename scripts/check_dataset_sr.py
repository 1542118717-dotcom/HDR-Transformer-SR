# -*- coding: utf-8 -*-
"""Shape check for the SIG17 HDR-SR dataset."""

import argparse
import os
import sys

# Allow running this script directly from the repository root, e.g.:
#   python scripts/check_dataset_sr.py --root_dir ./data
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.dataset_sig17_sr import SIG17_SR_Training_Dataset, SIG17_SR_Validation_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Check HDR-SR dataset input/label shapes.')
    parser.add_argument('--root_dir', type=str, default='./data', help='SIG17/Kalantari17 dataset root directory.')
    parser.add_argument('--split', type=str, default='Training', choices=['Training', 'Validation'],
                        help='Use cropped training data or Test validation data.')
    parser.add_argument('--sub_set', type=str, default='Training', help='Training subset directory name.')
    parser.add_argument('--scale', type=int, default=2, help='LDR downsampling scale.')
    parser.add_argument('--index', type=int, default=0, help='Sample index to inspect.')
    parser.add_argument('--crop_size', type=int, default=512, help='Validation crop size.')
    parser.add_argument('--no_crop', action='store_true', help='Disable validation crop.')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.split == 'Training':
        dataset = SIG17_SR_Training_Dataset(
            root_dir=args.root_dir,
            sub_set=args.sub_set,
            is_training=True,
            scale=args.scale,
        )
    else:
        dataset = SIG17_SR_Validation_Dataset(
            root_dir=args.root_dir,
            is_training=False,
            crop=not args.no_crop,
            crop_size=args.crop_size,
            scale=args.scale,
        )

    print('Loaded {} HDR-SR samples from {}.'.format(len(dataset), args.root_dir))
    sample = dataset[args.index]

    for key in ['input0', 'input1', 'input2', 'label']:
        print('{} shape: {}'.format(key, list(sample[key].shape)))

    label_h, label_w = sample['label'].shape[-2:]
    expected_h = label_h // args.scale
    expected_w = label_w // args.scale

    for key in ['input0', 'input1', 'input2']:
        input_h, input_w = sample[key].shape[-2:]
        if input_h != expected_h or input_w != expected_w:
            raise AssertionError(
                '{} spatial shape should be [{}, {}], got [{}, {}]'.format(
                    key, expected_h, expected_w, input_h, input_w
                )
            )

    print('HDR-SR dataset shape test passed.')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""Check SIG17 HDR-SR training dataset loading from an explicit scene list."""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.dataset_sig17_sr import SIG17_SR_Training_Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check HDR-SR training dataset loading with a scene list.'
    )
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='SIG17/Kalantari17 dataset root directory.')
    parser.add_argument('--sub_set', type=str, required=True,
                        help='Training subset directory name under dataset_dir.')
    parser.add_argument('--scene_list_file', type=str, required=True,
                        help='Text file containing one scene name per line.')
    parser.add_argument('--scale', type=int, default=2,
                        help='LDR downsampling scale. Default: 2.')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = SIG17_SR_Training_Dataset(
        root_dir=args.dataset_dir,
        sub_set=args.sub_set,
        is_training=True,
        scale=args.scale,
        scene_list_file=args.scene_list_file,
    )

    print('Loaded scene count: {}'.format(len(dataset)))
    print('First scene name: {}'.format(dataset.scenes_list[0]))

    sample = dataset[0]
    for key in ['input0', 'input1', 'input2', 'label']:
        print('{} shape: {}'.format(key, list(sample[key].shape)))


if __name__ == '__main__':
    main()

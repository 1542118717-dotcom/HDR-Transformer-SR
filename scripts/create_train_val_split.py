# -*- coding: utf-8 -*-
"""Create a reproducible scene-level train/validation split.

The script scans ``<dataset_dir>/Training`` for valid scene directories and
writes scene-name lists to ``train_scenes.txt`` and ``val_scenes.txt``. A valid
scene is a directory containing both ``exposure.txt`` and ``label.hdr``.
Original dataset files are never moved, copied, or modified.
"""

import argparse
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a reproducible scene-level train/validation split.'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Dataset root directory containing the Training directory.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where train_scenes.txt and val_scenes.txt are written.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=443,
        help='Random seed used for reproducible scene-level splitting.',
    )
    parser.add_argument(
        '--num_val',
        type=int,
        default=10,
        help='Number of validation scenes.',
    )
    return parser.parse_args()


def is_valid_scene(scene_dir):
    """Return True when a scene directory has the required files."""
    return (
        os.path.isdir(scene_dir)
        and os.path.isfile(os.path.join(scene_dir, 'exposure.txt'))
        and os.path.isfile(os.path.join(scene_dir, 'label.hdr'))
    )


def collect_valid_scenes(training_dir):
    """Collect valid scene directory names in a stable order."""
    if not os.path.isdir(training_dir):
        raise FileNotFoundError('Training directory does not exist: {}'.format(training_dir))

    valid_scenes = []
    for scene_name in sorted(os.listdir(training_dir)):
        scene_dir = os.path.join(training_dir, scene_name)
        if is_valid_scene(scene_dir):
            valid_scenes.append(scene_name)
    return valid_scenes


def split_scenes(scenes, seed, num_val):
    """Shuffle scenes deterministically and split into train/validation lists."""
    if num_val < 0:
        raise ValueError('--num_val must be non-negative, got {}'.format(num_val))
    if num_val > len(scenes):
        raise ValueError(
            '--num_val ({}) cannot exceed valid scene count ({})'.format(
                num_val, len(scenes)
            )
        )

    shuffled_scenes = list(scenes)
    rng = random.Random(seed)
    rng.shuffle(shuffled_scenes)

    val_scenes = sorted(shuffled_scenes[:num_val])
    train_scenes = sorted(shuffled_scenes[num_val:])
    return train_scenes, val_scenes


def write_scene_list(path, scenes):
    """Write one scene name per line."""
    with open(path, 'w') as f:
        for scene in scenes:
            f.write('{}\n'.format(scene))


def main():
    args = parse_args()

    training_dir = os.path.join(args.dataset_dir, 'Training')
    valid_scenes = collect_valid_scenes(training_dir)
    train_scenes, val_scenes = split_scenes(valid_scenes, args.seed, args.num_val)

    os.makedirs(args.output_dir, exist_ok=True)
    write_scene_list(os.path.join(args.output_dir, 'train_scenes.txt'), train_scenes)
    write_scene_list(os.path.join(args.output_dir, 'val_scenes.txt'), val_scenes)

    print('Total valid scenes: {}'.format(len(valid_scenes)))
    print('Training scenes: {}'.format(len(train_scenes)))
    print('Validation scenes: {}'.format(len(val_scenes)))
    print('Validation scene names:')
    for scene in val_scenes:
        print(scene)


if __name__ == '__main__':
    main()

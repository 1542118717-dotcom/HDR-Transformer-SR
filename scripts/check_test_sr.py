# -*- coding: utf-8 -*-
"""Static smoke checks for the independent HDR-SR inference script."""

import argparse
import ast
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SR_PATH = os.path.join(REPO_ROOT, 'test_sr.py')


def main():
    parser = argparse.ArgumentParser(description='Static smoke check for test_sr.py')
    parser.add_argument('--no_cuda', action='store_true', help='accepted for consistency with other smoke scripts')
    parser.parse_args()

    with open(TEST_SR_PATH, 'r') as f:
        source = f.read()
    tree = ast.parse(source, filename=TEST_SR_PATH)

    required_args = {
        '--dataset_dir',
        '--checkpoint',
        '--save_dir',
        '--scale',
        '--window_size',
        '--val_lr_patch_size',
        '--index',
        '--no_cuda',
    }
    found_args = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith('--'):
                    found_args.add(arg.value)

    missing_args = sorted(required_args - found_args)
    if missing_args:
        raise AssertionError('test_sr.py missing arguments: {}'.format(missing_args))

    required_names = {
        'SIG17_SR_Validation_Dataset',
        'HDRTransformerSR',
        'load_checkpoint',
        'save_outputs',
        'tonemap_to_uint8',
    }
    found_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    missing_names = sorted(required_names - found_names)
    if missing_names:
        raise AssertionError('test_sr.py missing required references: {}'.format(missing_names))

    print('test_sr static smoke check passed.')


if __name__ == '__main__':
    main()

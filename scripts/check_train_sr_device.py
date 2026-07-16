# -*- coding: utf-8 -*-
"""Lightweight checks for train_sr device resolution."""

import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_sr import resolve_device


def make_args(device, no_cuda=False):
    return argparse.Namespace(device=device, no_cuda=no_cuda)


def expect_runtime_error(device):
    try:
        resolve_device(make_args(device))
    except RuntimeError as exc:
        if device not in str(exc).lower():
            raise AssertionError('Error for --device {} should mention requested device: {}'.format(device, exc))
        return
    raise AssertionError('Expected --device {} to fail when unavailable'.format(device))


def main():
    cpu_device = resolve_device(make_args('cpu'))
    if cpu_device.type != 'cpu':
        raise AssertionError('Expected --device cpu to resolve to cpu, got {}'.format(cpu_device.type))

    no_cuda_device = resolve_device(make_args('cuda', no_cuda=True))
    if no_cuda_device.type != 'cpu':
        raise AssertionError('Expected --no_cuda to force cpu, got {}'.format(no_cuda_device.type))

    auto_device = resolve_device(make_args('auto'))
    if torch.cuda.is_available():
        expected_auto = 'cuda'
    elif torch.backends.mps.is_available():
        expected_auto = 'mps'
    else:
        expected_auto = 'cpu'
    if auto_device.type != expected_auto:
        raise AssertionError('Expected --device auto to resolve to {}, got {}'.format(expected_auto, auto_device.type))

    if torch.cuda.is_available():
        cuda_device = resolve_device(make_args('cuda'))
        if cuda_device.type != 'cuda':
            raise AssertionError('Expected --device cuda to resolve to cuda, got {}'.format(cuda_device.type))
    else:
        expect_runtime_error('cuda')

    if torch.backends.mps.is_available():
        mps_device = resolve_device(make_args('mps'))
        if mps_device.type != 'mps':
            raise AssertionError('Expected --device mps to resolve to mps, got {}'.format(mps_device.type))
    else:
        expect_runtime_error('mps')

    print('train_sr device resolution checks passed: auto={}, cpu=cpu'.format(auto_device.type))


if __name__ == '__main__':
    main()

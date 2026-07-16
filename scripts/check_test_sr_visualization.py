# -*- coding: utf-8 -*-
"""Lightweight checks for test_sr device, dataset, and visualization helpers."""

import argparse
import os
import tempfile
import unittest
from unittest import mock

import importlib.util
import sys
import types

import numpy as np
import torch

if importlib.util.find_spec('cv2') is None:
    cv2_stub = types.ModuleType('cv2')
    cv2_stub.IMREAD_UNCHANGED = -1
    cv2_stub.INTER_CUBIC = 2

    def _imwrite(path, image):
        with open(path, 'wb') as f:
            np.save(f, image)
        return True

    def _imread(path, flags=None):
        with open(path, 'rb') as f:
            return np.load(f)

    def _resize(image, dsize, interpolation=None):
        target_width, target_height = dsize
        y_idx = np.linspace(0, image.shape[0] - 1, target_height).round().astype(np.int64)
        x_idx = np.linspace(0, image.shape[1] - 1, target_width).round().astype(np.int64)
        return image[y_idx][:, x_idx]

    cv2_stub.imwrite = _imwrite
    cv2_stub.imread = _imread
    cv2_stub.resize = _resize
    sys.modules['cv2'] = cv2_stub

import cv2
import test_sr
from dataset.dataset_sig17_sr import SIG17_SR_Training_Dataset


class TestSRInferenceHelpers(unittest.TestCase):
    def test_auto_resolves_to_mps_when_available(self):
        args = argparse.Namespace(no_cuda=False, device='auto')
        with mock.patch('torch.cuda.is_available', return_value=False), \
                mock.patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(test_sr.resolve_device(args), torch.device('mps'))

    def test_fixed_scene_list_builds_validation_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_name = 'scene_001'
            subset_dir = os.path.join(tmpdir, 'Training')
            scene_dir = os.path.join(subset_dir, scene_name)
            os.makedirs(scene_dir)
            with open(os.path.join(scene_dir, 'exposure.txt'), 'w') as f:
                f.write('1\n2\n4\n')
            open(os.path.join(scene_dir, 'label.hdr'), 'wb').close()
            for idx in range(3):
                open(os.path.join(scene_dir, 'input_{}.tif'.format(idx)), 'wb').close()
            scene_list_file = os.path.join(tmpdir, 'val_scenes.txt')
            with open(scene_list_file, 'w') as f:
                f.write(scene_name + '\n')

            args = argparse.Namespace(
                dataset_dir=tmpdir,
                sub_set='Training',
                scale=2,
                val_scene_list_file=scene_list_file,
                val_lr_patch_size=64,
                window_size=8,
            )
            dataset = test_sr.build_dataset(args)
            self.assertIsInstance(dataset, SIG17_SR_Training_Dataset)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.scenes_list[0], scene_name)

    def test_visualization_outputs_are_hwc_three_channel_and_concat(self):
        input1 = np.zeros((6, 64, 64), dtype=np.float32)
        input1[3] = 1.0
        input1[4] = 0.5
        input1[5] = 0.25
        output = np.ones((3, 128, 128), dtype=np.float32) * 0.4
        label = np.linspace(0.0, 1.0, 3 * 128 * 128, dtype=np.float32).reshape(3, 128, 128)

        input1_bgr = test_sr.input1_to_uint8_bgr(input1)
        output_bgr, label_bgr = test_sr.hdr_pair_to_uint8_bgr(output, label)
        self.assertEqual(input1_bgr.shape, (64, 64, 3))
        self.assertEqual(output_bgr.shape, (128, 128, 3))
        self.assertEqual(label_bgr.shape, (128, 128, 3))

        resized_input1_bgr = test_sr.resize_bgr_to_shape(input1_bgr, output_bgr.shape[0], output_bgr.shape[1])
        comparison_bgr = np.concatenate([resized_input1_bgr, output_bgr, label_bgr], axis=1)
        self.assertEqual(comparison_bgr.shape, (128, 384, 3))

    def test_bgr_save_has_no_extra_red_blue_swap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'bgr.png')
            bgr = np.zeros((2, 2, 3), dtype=np.uint8)
            bgr[:, :, 0] = 255
            test_sr.save_bgr_png(path, bgr)
            loaded = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(loaded, bgr)


if __name__ == '__main__':
    unittest.main()

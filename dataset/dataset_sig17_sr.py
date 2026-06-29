# -*- coding: utf-8 -*-
"""SIG17/Kalantari17 dataset variants for HDR super-resolution.

These dataset classes keep the original HDR ground truth at high resolution while
creating low-resolution LDR inputs by bicubic downsampling the original
high-resolution input tensors. Exposure reading, LDR-to-HDR preprocessing, and
HDR label reading follow dataset/dataset_sig17.py.
"""

import os
import os.path as osp
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.utils import *


class _SIG17SRMixin(object):
    """Shared helpers for SIG17 HDR-SR datasets."""

    def _downsample_ldr_input(self, img):
        """Downsample one 6-channel HR-LDR input tensor to LR-LDR.

        Args:
            img (torch.Tensor): Tensor with shape [6, H, W]. The first three
                channels are exposure-linearized LDR values and the last three
                channels are the original LDR values, matching the original
                dataset implementation.

        Returns:
            torch.Tensor: Bicubic downsampled tensor with shape
                [6, H / scale, W / scale].
        """
        lr_size = (img.shape[-2] // self.scale, img.shape[-1] // self.scale)
        return F.interpolate(
            img.unsqueeze(0),
            size=lr_size,
            mode='bicubic',
            align_corners=False,
        ).squeeze(0)

    def _make_sr_sample(self, ldr_images, expoTimes, label):
        """Build LR-LDR inputs and an unchanged HR-HDR label."""
        # Keep the original exposure conversion and 6-channel input construction:
        # [linearized LDR RGB, original LDR RGB].
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)

        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)

        img0 = torch.from_numpy(pre_img0.astype(np.float32).transpose(2, 0, 1))
        img1 = torch.from_numpy(pre_img1.astype(np.float32).transpose(2, 0, 1))
        img2 = torch.from_numpy(pre_img2.astype(np.float32).transpose(2, 0, 1))

        # Do not alter the HDR ground-truth value range; only convert layout/type.
        label = torch.from_numpy(label.astype(np.float32).transpose(2, 0, 1))

        sample = {
            'input0': self._downsample_ldr_input(img0),
            'input1': self._downsample_ldr_input(img1),
            'input2': self._downsample_ldr_input(img2),
            'label': label,
        }
        return sample


class SIG17_SR_Training_Dataset(_SIG17SRMixin, Dataset):
    """Training split for LR-LDR to HR-HDR super-resolution experiments."""

    def __init__(self, root_dir, sub_set, is_training=True, scale=2):
        self.root_dir = root_dir
        self.is_training = is_training
        self.sub_set = sub_set
        self.scale = scale

        self.scenes_dir = osp.join(root_dir, self.sub_set)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Exposure reading is intentionally identical to the original dataset.
        expoTimes = read_expo_times(self.image_list[index][0])
        ldr_images = read_images(self.image_list[index][1])
        label = read_label(self.image_list[index][2], 'label.hdr')
        return self._make_sr_sample(ldr_images, expoTimes, label)

    def __len__(self):
        return len(self.scenes_list)


class SIG17_SR_Validation_Dataset(_SIG17SRMixin, Dataset):
    """Validation/Test split for LR-LDR to HR-HDR shape checks."""

    def __init__(self, root_dir, is_training=False, crop=True, crop_size=512, scale=2):
        self.root_dir = root_dir
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size
        self.scale = scale

        self.scenes_dir = osp.join(root_dir, 'Test')
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Exposure reading is intentionally identical to the original dataset.
        expoTimes = read_expo_times(self.image_list[index][0])
        ldr_images = read_images(self.image_list[index][1])
        label = read_label(self.image_list[index][2], 'HDRImg.hdr')

        if self.crop:
            x = 0
            y = 0
            ldr_images = [ldr[x:x + self.crop_size, y:y + self.crop_size] for ldr in ldr_images]
            label = label[x:x + self.crop_size, y:y + self.crop_size]

        return self._make_sr_sample(ldr_images, expoTimes, label)

    def __len__(self):
        return len(self.scenes_list)

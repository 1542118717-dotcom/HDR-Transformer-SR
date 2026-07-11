# -*- coding: utf-8 -*-
"""Smoke test train_sr dataset construction with fixed train/val scene lists."""

import os
import shutil
import sys
import tempfile
from argparse import Namespace

import types

# Keep this smoke test dependency-light: dataset construction only needs file
# listing and torch Dataset/DataLoader symbols, not tensor operations.
utils_pkg = types.ModuleType('utils')
utils_mod = types.ModuleType('utils.utils')
def list_all_files_sorted(folder_name, extension=''):
    return sorted(
        os.path.join(folder_name, name)
        for name in os.listdir(folder_name)
        if name.endswith(extension)
    )
utils_mod.list_all_files_sorted = list_all_files_sorted
utils_mod.AverageMeter = object
utils_mod.batch_psnr = lambda *args, **kwargs: None
utils_mod.batch_psnr_mu = lambda *args, **kwargs: None
utils_mod.init_parameters = lambda *args, **kwargs: None
utils_mod.set_random_seed = lambda *args, **kwargs: None
utils_pkg.utils = utils_mod
sys.modules.setdefault('utils', utils_pkg)
sys.modules.setdefault('utils.utils', utils_mod)

class _Dataset(object):
    pass

torch_mod = types.ModuleType('torch')
torch_nn_mod = types.ModuleType('torch.nn')
torch_utils_mod = types.ModuleType('torch.utils')
torch_utils_data_mod = types.ModuleType('torch.utils.data')
torch_functional_mod = types.ModuleType('torch.nn.functional')
torch_utils_data_mod.Dataset = _Dataset
torch_utils_data_mod.DataLoader = object
torch_utils_mod.data = torch_utils_data_mod
torch_nn_mod.functional = torch_functional_mod
torch_mod.nn = torch_nn_mod
torch_mod.utils = torch_utils_mod
sys.modules.setdefault('torch', torch_mod)
sys.modules.setdefault('torch.nn', torch_nn_mod)
sys.modules.setdefault('torch.nn.functional', torch_functional_mod)
sys.modules.setdefault('torch.utils', torch_utils_mod)
sys.modules.setdefault('torch.utils.data', torch_utils_data_mod)
sys.modules.setdefault('numpy', types.ModuleType('numpy'))

models_pkg = types.ModuleType('models')
hdr_sr_mod = types.ModuleType('models.hdr_transformer_sr')
class HDRTransformerSR(object):
    pass
hdr_sr_mod.HDRTransformerSR = HDRTransformerSR
loss_mod = types.ModuleType('models.loss')
loss_mod.L1MuLoss = object
loss_mod.JointReconPerceptualLoss = object
sys.modules.setdefault('models', models_pkg)
sys.modules.setdefault('models.hdr_transformer_sr', hdr_sr_mod)
sys.modules.setdefault('models.loss', loss_mod)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.dataset_sig17_sr import SIG17_SR_Training_Dataset
from train_sr import build_datasets


def _write_scene(scene_dir):
    os.makedirs(scene_dir, exist_ok=True)
    with open(os.path.join(scene_dir, 'exposure.txt'), 'w') as f:
        f.write('1\n2\n4\n')
    with open(os.path.join(scene_dir, 'label.hdr'), 'wb') as f:
        f.write(b'')
    for idx in range(3):
        with open(os.path.join(scene_dir, 'input_{}.tif'.format(idx)), 'wb') as f:
            f.write(b'')


def _write_scene_list(path, scene_names):
    with open(path, 'w') as f:
        for scene_name in scene_names:
            f.write(scene_name + '\n')


def main():
    tmp_dir = tempfile.mkdtemp(prefix='hdr_sr_scene_lists_')
    try:
        sub_set = 'Training'
        train_scene_names = ['train_scene_{:02d}'.format(idx) for idx in range(64)]
        val_scene_names = ['val_scene_{:02d}'.format(idx) for idx in range(10)]
        for scene_name in train_scene_names + val_scene_names:
            _write_scene(os.path.join(tmp_dir, sub_set, scene_name))

        train_list_file = os.path.join(tmp_dir, 'train_scenes.txt')
        val_list_file = os.path.join(tmp_dir, 'val_scenes.txt')
        _write_scene_list(train_list_file, train_scene_names)
        _write_scene_list(val_list_file, val_scene_names)

        args = Namespace(
            dataset_dir=tmp_dir,
            sub_set=sub_set,
            scale=2,
            crop_size=512,
            train_scene_list_file=train_list_file,
            val_scene_list_file=val_list_file,
        )
        train_dataset, val_dataset = build_datasets(args)

        assert isinstance(train_dataset, SIG17_SR_Training_Dataset)
        assert isinstance(val_dataset, SIG17_SR_Training_Dataset)
        assert len(train_dataset) == 64, 'expected 64 training scenes, got {}'.format(len(train_dataset))
        assert len(val_dataset) == 10, 'expected 10 validation scenes, got {}'.format(len(val_dataset))
        assert train_dataset.scene_list_file == train_list_file
        assert val_dataset.scene_list_file == val_list_file
        assert val_dataset.root_dir == tmp_dir
        assert val_dataset.sub_set == sub_set
        assert val_dataset.is_training is False
        assert val_dataset.scale == 2

        print('Training size: {}'.format(len(train_dataset)))
        print('Validation size: {}'.format(len(val_dataset)))
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    main()

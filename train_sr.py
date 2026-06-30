# -*- coding: utf-8 -*-
"""Independent training entry point for HDR-Transformer super-resolution."""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.dataset_sig17_sr import SIG17_SR_Training_Dataset, SIG17_SR_Validation_Dataset
from models.hdr_transformer_sr import HDRTransformerSR
from models.loss import L1MuLoss, JointReconPerceptualLoss
from utils.utils import AverageMeter, batch_psnr, batch_psnr_mu, init_parameters, set_random_seed


def get_args():
    parser = argparse.ArgumentParser(
        description='HDR-Transformer-SR training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset_dir', type=str, default='./data', help='SIG17/Kalantari17 dataset root directory')
    parser.add_argument('--sub_set', type=str, default='sig17_training_crop128_stride64', help='training subset directory')
    parser.add_argument('--scale', type=int, default=2, help='super-resolution scale factor')
    parser.add_argument('--window_size', type=int, default=8, help='HDRTransformer window size for LR input cropping')
    parser.add_argument('--train_lr_patch_size', type=int, default=64, help='LR patch size used for training crops')
    parser.add_argument('--val_lr_patch_size', type=int, default=64, help='LR patch size used for validation crops')
    parser.add_argument('--crop_size', type=int, default=512, help='validation crop size before downsampling')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_sr', help='directory for HDR-SR checkpoints')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N', help='number of dataloader workers')

    parser.add_argument('--resume', type=str, default=None, help='load HDR-SR checkpoint from a .pth file')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=443, metavar='S', help='random seed')
    parser.add_argument('--init_weights', action='store_true', default=False, help='initialize model weights')
    parser.add_argument('--loss_func', type=int, default=1, choices=[0, 1], help='0: L1MuLoss, 1: JointReconPerceptualLoss')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate')
    parser.add_argument('--lr_decay_interval', type=int, default=100, help='decay learning rate every N epochs')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='start epoch')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='validation batch size')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='batches between train logs')
    parser.add_argument('--max_train_batches', type=int, default=None, help='maximum training batches per epoch for smoke tests')
    parser.add_argument('--max_val_batches', type=int, default=None, help='maximum validation batches per epoch for smoke tests')
    return parser.parse_args()


def assert_output_label_shape(output, label, stage):
    if output.shape != label.shape:
        raise RuntimeError(
            '{} output shape {} does not match label shape {}'.format(
                stage, tuple(output.shape), tuple(label.shape)
            )
        )


def validate_lr_patch_size(patch_size, window_size, stage):
    if patch_size <= 0:
        raise ValueError('{} LR patch size must be positive, got {}'.format(stage, patch_size))
    if window_size <= 0:
        raise ValueError('window_size must be positive, got {}'.format(window_size))
    if patch_size % window_size != 0:
        raise ValueError(
            '{} LR patch size ({}) must be divisible by window_size ({})'.format(
                stage, patch_size, window_size
            )
        )


def crop_sr_batch_to_window(input0, input1, input2, label, window_size=8, scale=2, patch_size=None, random_crop=False):
    """Crop an HDR-SR batch to an LR patch aligned with the HR label.

    The three LDR inputs are LR tensors with shape ``[B, 6, H, W]``. The HDR
    label is the aligned HR tensor with shape ``[B, 3, scale * H, scale * W]``.
    When ``patch_size`` is provided, LR inputs are cropped to
    ``[B, 6, patch_size, patch_size]`` and labels to
    ``[B, 3, patch_size * scale, patch_size * scale]``. Otherwise this keeps
    the previous behavior of cropping to the largest top-left LR region whose
    dimensions are divisible by ``window_size``.
    """
    if window_size <= 0:
        raise ValueError('window_size must be positive, got {}'.format(window_size))
    if scale <= 0:
        raise ValueError('scale must be positive, got {}'.format(scale))
    if patch_size is not None:
        if patch_size <= 0:
            raise ValueError('LR patch size must be positive, got {}'.format(patch_size))
        if patch_size % window_size != 0:
            raise ValueError(
                'LR patch size ({}) must be divisible by window_size ({})'.format(
                    patch_size, window_size
                )
            )

    lr_height, lr_width = input0.shape[-2], input0.shape[-1]
    for name, tensor in (('input1', input1), ('input2', input2)):
        if tensor.shape[-2:] != (lr_height, lr_width):
            raise RuntimeError(
                '{} LR shape {} does not match input0 LR shape {}'.format(
                    name, tuple(tensor.shape[-2:]), (lr_height, lr_width)
                )
            )

    if patch_size is None:
        lr_crop_height = (lr_height // window_size) * window_size
        lr_crop_width = (lr_width // window_size) * window_size
        if lr_crop_height == 0 or lr_crop_width == 0:
            raise RuntimeError(
                'LR input shape {} is too small for window_size {}'.format(
                    (lr_height, lr_width), window_size
                )
            )
        lr_top = 0
        lr_left = 0
    else:
        lr_crop_height = patch_size
        lr_crop_width = patch_size
        if lr_height < patch_size or lr_width < patch_size:
            raise RuntimeError(
                'LR input shape {} is smaller than requested patch size {}'.format(
                    (lr_height, lr_width), patch_size
                )
            )
        if random_crop:
            lr_top = torch.randint(0, lr_height - patch_size + 1, (1,), device=input0.device).item()
            lr_left = torch.randint(0, lr_width - patch_size + 1, (1,), device=input0.device).item()
        else:
            lr_top = (lr_height - patch_size) // 2
            lr_left = (lr_width - patch_size) // 2

    hr_top = lr_top * scale
    hr_left = lr_left * scale
    hr_crop_height = lr_crop_height * scale
    hr_crop_width = lr_crop_width * scale
    if label.shape[-2] < hr_top + hr_crop_height or label.shape[-1] < hr_left + hr_crop_width:
        raise RuntimeError(
            'Label HR shape {} is smaller than required aligned crop at {} with size {}'.format(
                tuple(label.shape[-2:]), (hr_top, hr_left), (hr_crop_height, hr_crop_width)
            )
        )

    return (
        input0[..., lr_top:lr_top + lr_crop_height, lr_left:lr_left + lr_crop_width],
        input1[..., lr_top:lr_top + lr_crop_height, lr_left:lr_left + lr_crop_width],
        input2[..., lr_top:lr_top + lr_crop_height, lr_left:lr_left + lr_crop_width],
        label[..., hr_top:hr_top + hr_crop_height, hr_left:hr_left + hr_crop_width],
    )


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def checkpoint_state(epoch, model, optimizer):
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    return {
        'epoch': epoch + 1,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
    }


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    for batch_idx, batch_data in enumerate(train_loader):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
        data_time.update(time.time() - end)
        input0 = batch_data['input0'].to(device)
        input1 = batch_data['input1'].to(device)
        input2 = batch_data['input2'].to(device)
        label = batch_data['label'].to(device)
        input0, input1, input2, label = crop_sr_batch_to_window(
            input0,
            input1,
            input2,
            label,
            window_size=args.window_size,
            scale=args.scale,
            patch_size=args.train_lr_patch_size,
            random_crop=True,
        )

        output = model(input0, input1, input2)
        assert_output_label_shape(output, label, 'train')
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx * args.batch_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx * args.batch_size / len(train_loader.dataset),
                    loss.item(),
                    batch_time=batch_time,
                    data_time=data_time,
                )
            )


def validate(args, model, device, val_loader, optimizer, epoch, criterion, best_mu_psnr):
    model.eval()
    val_psnr = AverageMeter()
    val_mu_psnr = AverageMeter()
    val_loss = AverageMeter()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                break
            input0 = batch_data['input0'].to(device)
            input1 = batch_data['input1'].to(device)
            input2 = batch_data['input2'].to(device)
            label = batch_data['label'].to(device)
            input0, input1, input2, label = crop_sr_batch_to_window(
                input0,
                input1,
                input2,
                label,
                window_size=args.window_size,
                scale=args.scale,
                patch_size=args.val_lr_patch_size,
                random_crop=False,
            )

            output = model(input0, input1, input2)
            assert_output_label_shape(output, label, 'validation')
            loss = criterion(output, label)

            val_loss.update(loss.item())
            val_psnr.update(batch_psnr(output, label, 1.0).item())
            val_mu_psnr.update(batch_psnr_mu(output, label, 1.0).item())

    print('Validation set: Average Loss: {:.4f}'.format(val_loss.avg))
    print('Validation set: Average PSNR: {:.4f}, mu_law: {:.4f}'.format(val_psnr.avg, val_mu_psnr.avg))

    save_dict = checkpoint_state(epoch, model, optimizer)
    torch.save(save_dict, os.path.join(args.save_dir, 'hdr_sr_val_latest_checkpoint.pth'))
    if val_mu_psnr.avg > best_mu_psnr[0]:
        best_mu_psnr[0] = val_mu_psnr.avg
        torch.save(save_dict, os.path.join(args.save_dir, 'hdr_sr_best_checkpoint.pth'))
        with open(os.path.join(args.save_dir, 'hdr_sr_best_checkpoint.txt'), 'w') as f:
            f.write('best epoch: {}\n'.format(epoch))
            f.write('Validation set: Average PSNR: {:.4f}, PSNR_mu_law: {:.4f}\n'.format(val_psnr.avg, val_mu_psnr.avg))


def main():
    args = get_args()
    validate_lr_patch_size(args.train_lr_patch_size, args.window_size, 'Training')
    validate_lr_patch_size(args.val_lr_patch_size, args.window_size, 'Validation')

    if args.seed is not None:
        set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = HDRTransformerSR(
        embed_dim=60,
        depths=[6, 6, 6],
        num_heads=[6, 6, 6],
        mlp_ratio=2,
        in_chans=6,
        scale=args.scale,
    )
    if args.init_weights:
        init_parameters(model)

    loss_dict = {0: L1MuLoss, 1: JointReconPerceptualLoss}
    criterion = loss_dict[args.loss_func]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)

    model.to(device)
    if torch.cuda.device_count() > 1 and use_cuda:
        model = nn.DataParallel(model)

    if args.resume:
        if os.path.isfile(args.resume):
            print('===> Loading checkpoint from: {}'.format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            target_model = model.module if isinstance(model, nn.DataParallel) else model
            target_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('===> Loaded checkpoint: epoch {}'.format(checkpoint['epoch']))
        else:
            raise FileNotFoundError('No checkpoint found at {}'.format(args.resume))

    train_dataset = SIG17_SR_Training_Dataset(
        root_dir=args.dataset_dir,
        sub_set=args.sub_set,
        is_training=True,
        scale=args.scale,
    )
    val_dataset = SIG17_SR_Validation_Dataset(
        root_dir=args.dataset_dir,
        is_training=False,
        crop=True,
        crop_size=args.crop_size,
        scale=args.scale,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    print('''===> Start training HDR-Transformer-SR

        Dataset dir:     {}
        Subset:          {}
        Scale:           {}
        Window size:     {}
        Train LR patch:  {}
        Val LR patch:    {}
        Epochs:          {}
        Batch size:      {}
        Loss function:   {}
        Learning rate:   {}
        Training size:   {}
        Device:          {}
        Save dir:        {}
        '''.format(
            args.dataset_dir,
            args.sub_set,
            args.scale,
            args.window_size,
            args.train_lr_patch_size,
            args.val_lr_patch_size,
            args.epochs,
            args.batch_size,
            args.loss_func,
            args.lr,
            len(train_loader.dataset),
            device.type,
            args.save_dir,
        )
    )

    best_mu_psnr = [-1.0]
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch - 1)
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        validate(args, model, device, val_loader, optimizer, epoch, criterion, best_mu_psnr)
        torch.save(checkpoint_state(epoch, model, optimizer), os.path.join(args.save_dir, 'hdr_sr_epoch_{:04d}.pth'.format(epoch)))


if __name__ == '__main__':
    main()

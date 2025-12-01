# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard library imports
import os
import sys
import time
import math
import json
import argparse
import subprocess
from pathlib import Path
import signal

# Third-party imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

# Local imports
import utils
import vision_transformer as vits
from uncertainty_head import UncertaintyHead
from uncertainty_loss import UncertaintyAwareDINOLoss
from uncertainty_metrics import UncertaintyMetrics
from vision_transformer import UncertaintyAwareDINOHead
from data_augmentation import DataAugmentationDINO

import argparse

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils
import vision_transformer as vits
from uncertainty_head import UncertaintyHead
from uncertainty_loss import UncertaintyAwareDINOLoss
from uncertainty_metrics import UncertaintyMetrics

import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
from uncertainty_head import UncertaintyHead
from uncertainty_loss import UncertaintyAwareDINOLoss
from uncertainty_metrics import UncertaintyMetrics
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

# Global flag for handling interrupts
training_running = True


def signal_handler(signum, frame):
    global training_running
    print("\nSignal received. Stopping training gracefully...")
    training_running = False


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination request


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base',
                                 'xcit', 'deit_tiny', 'deit_small']
                        + torchvision_archs +
                        torch.hub.list("facebookresearch/xcit:main"),
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float,
                        default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str,
                        help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int,
                        help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")
    return parser


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used.
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features
        return self.head(output)


class UncertaintyDINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True):
        super().__init__()

        # Mean prediction layers
        mean_layers = []
        mean_layers.append(nn.Linear(in_dim, in_dim))
        mean_layers.append(nn.GELU())
        if use_bn:
            mean_layers.append(nn.BatchNorm1d(in_dim))
        mean_layers.append(nn.Linear(in_dim, in_dim))
        mean_layers.append(nn.GELU())
        if use_bn:
            mean_layers.append(nn.BatchNorm1d(in_dim))
        mean_layers.append(nn.Linear(in_dim, out_dim))
        self.mean_layers = nn.Sequential(*mean_layers)

        # Variance prediction layers with positive output
        var_layers = []
        var_layers.append(nn.Linear(in_dim, in_dim))
        var_layers.append(nn.GELU())
        if use_bn:
            var_layers.append(nn.BatchNorm1d(in_dim))
        var_layers.append(nn.Linear(in_dim, in_dim))
        var_layers.append(nn.GELU())
        if use_bn:
            var_layers.append(nn.BatchNorm1d(in_dim))
        var_layers.append(nn.Linear(in_dim, out_dim))
        var_layers.append(nn.Softplus())  # Ensure positive variance
        self.var_layers = nn.Sequential(*var_layers)

        self.norm_last_layer = norm_last_layer

        # Initialize weights with different scales
        self.apply(self._init_weights)
        if norm_last_layer:
            nn.init.constant_(self.mean_layers[-1].bias, 0)
            self.mean_layers[-1].weight.data.normal_(mean=0.0, std=0.01)

        # Initialize variance layers with smaller weights
        for m in self.var_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, -1.0)  # Start with small variance

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Returns: tuple (mean, variance)
        """
        # Ensure x requires gradient
        if not x.requires_grad:
            x = x.requires_grad_(True)

        # Get mean and variance predictions
        mean = self.mean_layers(x)
        var = self.var_layers(x)  # Already positive due to Softplus

        # Optional normalization of mean
        if self.norm_last_layer:
            mean = nn.functional.normalize(mean, dim=-1, p=2)

        return mean, var + 1e-6  # Add small constant for numerical stability


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust the learning rate with warmup and cosine schedule.
    """
    # Warmup
    if epoch < args.warmup_epochs:
        lr = args.lr * (epoch + 1) / args.warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - args.warmup_epochs) / \
            (args.epochs - args.warmup_epochs)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * \
            (1 + math.cos(math.pi * progress))

    # Update optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_dino(args, dataset):
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    # Create models
    student = create_model(args, is_student=True)
    teacher = create_model(args, is_student=False)

    # Create optimizer with correct parameters
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create loss
    dino_loss = UncertaintyAwareDINOLoss(
        out_dim=args.out_dim,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
    ).to(args.device)

    # Track statistics
    training_stats = {
        'losses': [],
        'lrs': [],
        'epochs': []
    }

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        # Set train mode
        student.train()
        teacher.eval()  # Teacher in eval mode

        # Reset metrics
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f'Epoch: [{epoch}/{args.epochs}]'

        # Update learning rate
        lr = adjust_learning_rate(optimizer, epoch, args)

        for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            # Move to device
            images = [im.to(args.device) for im in images]

            # Forward passes
            with torch.no_grad():
                teacher_output = teacher(images[:2])  # Only global views
            student_output = student(images)

            # Compute loss
            loss = dino_loss(student_output, teacher_output, epoch)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update teacher
            with torch.no_grad():
                m = args.momentum_teacher
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # Log
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Print epoch stats
        print(f"Epoch {epoch} loss: {metric_logger.loss.global_avg:.4f}")

        # Store stats
        training_stats['losses'].append(metric_logger.loss.global_avg)
        training_stats['lrs'].append(optimizer.param_groups[0]["lr"])
        training_stats['epochs'].append(epoch)

    return student, training_stats


def create_model(args, is_student=True):
    # Create backbone
    backbone = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate if is_student else 0,
    )

    # Create head
    head = UncertaintyDINOHead(
        backbone.embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer if is_student else False,
    )

    # Create model
    model = MultiCropWrapper(backbone, head)
    model = model.to(args.device)

    # Set requires_grad
    for p in model.parameters():
        p.requires_grad = is_student

    return model


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q *
                                 F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / \
            (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + \
            batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(
                96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)

# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline for BRATS Challenge")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument(
    "--data_dir",
    default="/path/to/Task01_BrainTumour/",
    type=str,
    help="MSD Task01_BrainTumour root",
)
parser.add_argument(
    "--json_list",
    default="./jsons/task01_brats.json",
    type=str,
    help="Task01 train/val json (387/97)",
)
parser.add_argument("--cache_dir", default="", type=str, help="PersistentDataset cache dir (optional)")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=2, type=int, help="PosNeg crop samples / sliding-window batch")
parser.add_argument("--optim_lr", default=3e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="TC / WT / ET")
parser.add_argument("--cache_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")
parser.add_argument(
    "--per_step_lr",
    action="store_true",
    default=True,
    help="step LR each training iteration; use --no_per_step_lr for epoch schedule",
)
parser.add_argument(
    "--no_per_step_lr",
    action="store_false",
    dest="per_step_lr",
    help="step LR once per epoch",
)
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument(
    "--pretrained_dir",
    default="./pretrained_models/",
    type=str,
    help="pretrained checkpoint directory",
)
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument(
    "--ssl_pretrained_ckpt",
    default="",
    type=str,
    help="path to SSL pretrained checkpoint (required with --use_ssl_pretrained)",
)
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")

def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    args.out_channels = 3
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=True,
    )

    if args.resume_ckpt:
        model_dict = torch.load(pretrained_pth)["state_dict"]
        model.load_state_dict(model_dict)
        print("Using pretrained weights")

    if args.use_ssl_pretrained:
        ssl_ckpt = args.ssl_pretrained_ckpt
        if not ssl_ckpt:
            ssl_ckpt = os.path.join(pretrained_dir, args.pretrained_model_name)
        if not os.path.isfile(ssl_ckpt):
            raise FileNotFoundError(
                f"SSL pretrained checkpoint not found: {ssl_ckpt}. "
                "Pass --ssl_pretrained_ckpt or --pretrained_dir/--pretrained_model_name."
            )
        model_dict = torch.load(ssl_ckpt, map_location="cpu")
        state_dict = model_dict["state_dict"]
        if "module." in list(state_dict.keys())[0]:
            print("Tag 'module.' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.", "")] = state_dict.pop(key)
        if "swin_vit" in list(state_dict.keys())[0]:
            print("Tag 'swin_vit' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
        model_dict = model.state_dict()
        loadable = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and tuple(model_dict[k].shape) == tuple(v.shape)
        }
        skipped = sorted(set(state_dict) - set(loadable))
        missing = sorted(set(model_dict) - set(loadable))
        model_dict.update(loadable)
        model.load_state_dict(model_dict, strict=True)
        print(f"Using pretrained self-supervised Swin UNETR backbone weights from {ssl_ckpt}")
        print(f"Loaded {len(loadable)}/{len(state_dict)} SSL tensors; skipped {len(skipped)} (shape/key mismatch)")
        if skipped[:8]:
            print("  e.g. skipped:", ", ".join(skipped[:8]))

    if args.squared_dice:
        dice_loss = DiceLoss(
            to_onehot_y=False,
            sigmoid=True,
            squared_pred=True,
            smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr,
        )
    else:
        dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_label = None
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    print("Loss/metric: sigmoid Dice on TC/WT/ET")
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        if args.per_step_lr:
            max_steps = args.max_epochs * len(loader[0])
            warmup_steps = args.warmup_epochs * len(loader[0])
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=warmup_steps, max_epochs=max_steps
            )
            print(f"Per-step warmup cosine: warmup_steps={warmup_steps}, max_steps={max_steps}")
        else:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
            )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    semantic_classes = ["Dice_Val_TC", "Dice_Val_WT", "Dice_Val_ET"]

    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
        semantic_classes=semantic_classes,
    )
    return accuracy


if __name__ == "__main__":
    main()

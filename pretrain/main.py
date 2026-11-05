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
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from losses.loss import Loss
from models.ssl_head import SSLHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from utils.data_utils import get_loader
from utils.ops import aug_rand, rot_rand
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def pack_state(global_step, best_val_recon, best_val_total):
        return {
            "global_step": global_step,
            "best_val": best_val_recon,
            "best_val_recon": best_val_recon,
            "best_val_total": best_val_total,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
        }

    def train(args, global_step, train_loader, val_best_recon, val_best_total, scaler):
        model.train()
        loss_train = []
        loss_train_recon = []
        print('len(train_loader)', len(train_loader))

        for step, batch in enumerate(train_loader):
            t1 = time()
            x = batch["image"].cuda()
            x1, rot1 = rot_rand(args, x)
            x2, rot2 = rot_rand(args, x)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)
            with autocast(enabled=args.amp):
                rot1_p, mu1, logvar1, p1, z1, rec_x1 = model(x1_augment)
                rot2_p, mu2, logvar2, p2, z2, rec_x2 = model(x2_augment)
                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rots = torch.cat([rot1, rot2], dim=0)
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                imgs = torch.cat([x1, x2], dim=0)
                loss, losses_tasks = loss_function(rot_p, rots, mu1, logvar1, mu2, logvar2, p1, p2, z1, z2, imgs_recon, imgs)

            loss_train.append(loss.item())
            loss_train_recon.append(losses_tasks[2].item())
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            if args.distributed:
                if dist.get_rank() == 0:
                    print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss, time() - t1))
            else:
                print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss, time() - t1))

            global_step += 1
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                val_metrics = validation(args, test_loader)
                val_loss = val_metrics["total"]
                val_loss_recon = val_metrics["recon"]
                print(
                    "Train Loss:{:.4f} Recon:{:.4f} | Val Loss:{:.4f} Recon:{:.4f}".format(
                        np.mean(loss_train), np.mean(loss_train_recon), val_loss, val_loss_recon
                    )
                )

                if val_loss_recon < val_best_recon:
                    val_best_recon = val_loss_recon
                    save_ckp(
                        pack_state(global_step, val_best_recon, val_best_total),
                        os.path.join(logdir, "model_bestValRMSE.pt"),
                    )
                    print(
                        "Model was saved ! Best Recon. Val Loss: {:.4f}, Recon. Val Loss: {:.4f}".format(
                            val_best_recon, val_loss_recon
                        )
                    )
                else:
                    print(
                        "Model was not saved ! Best Recon. Val Loss: {:.4f} Recon. Val Loss: {:.4f}".format(
                            val_best_recon, val_loss_recon
                        )
                    )

                if val_loss < val_best_total:
                    val_best_total = val_loss
                    save_ckp(
                        pack_state(global_step, val_best_recon, val_best_total),
                        os.path.join(logdir, "model_bestValTotal.pt"),
                    )
                    print(
                        "Model was saved ! Best Total Val Loss: {:.4f}, Total Val Loss: {:.4f}".format(
                            val_best_total, val_loss
                        )
                    )
        return global_step, loss, val_best_recon, val_best_total

    def validation(args, test_loader):
        model.eval()
        meters = {k: [] for k in ("total", "rot", "contrast", "recon", "dcl", "scl", "reg")}
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                val_inputs = batch["image"].cuda()
                x1, rot1 = rot_rand(args, val_inputs)
                x2, rot2 = rot_rand(args, val_inputs)
                x1_augment = aug_rand(args, x1)
                x2_augment = aug_rand(args, x2)
                with autocast(enabled=args.amp):
                    rot1_p, mu1, logvar1, p1, z1, rec_x1 = model(x1_augment)
                    rot2_p, mu2, logvar2, p2, z2, rec_x2 = model(x2_augment)
                    rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                    rots = torch.cat([rot1, rot2], dim=0)
                    imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                    imgs = torch.cat([x1, x2], dim=0)
                    loss, losses_tasks = loss_function(rot_p, rots, mu1, logvar1, mu2, logvar2, p1, p2, z1, z2, imgs_recon, imgs)
                meters["total"].append(loss.item())
                meters["rot"].append(losses_tasks[0].item())
                meters["contrast"].append(losses_tasks[1].item())
                meters["recon"].append(losses_tasks[2].item())
                meters["dcl"].append(losses_tasks[3].item())
                meters["scl"].append(losses_tasks[4].item())
                meters["reg"].append(losses_tasks[5].item())
                print(
                    "Validation step:{}, Loss:{:.4f}, Loss Reconstruction:{:.4f}".format(
                        step, loss, losses_tasks[2]
                    )
                )

        return {k: float(np.mean(v)) for k, v in meters.items()}

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save checkpoints")
    parser.add_argument("--data_path", default="", help="data root path")
    parser.add_argument("--json_path", default="", help="data json path")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=500, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=6e-6, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--local-rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    parser.add_argument("--persistent_dataset", action="store_true", help="use monai persistent Dataset")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir for PersistentDataset")
    parser.add_argument("--num_workers", default=16, type=int, help="dataloader workers")


    parser.add_argument("--alpha", default=1, type=float, help="contrast loss")
    parser.add_argument("--beta", default=1, type=float, help="reconstruct loss")
    parser.add_argument("--gamma", default=1, type=float, help="rotation prediction loss")
    parser.add_argument("--alpha1", default=1, type=float, help="contrast loss dcl")
    parser.add_argument("--alpha2", default=1, type=float, help="contrast loss scl")
    parser.add_argument("--alpha3", default=0.1, type=float, help="contrast loss reg")

    args = parser.parse_args()
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    logdir = "./runs/" + args.logdir
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)

    print('args', args)

    model = SSLHead(args)
    model.cuda()

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    else:
        raise ValueError("Unsupported optimizer")
    print('init_optimizer', optimizer)
    scheduler = None
    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":
            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        else:
            raise ValueError("Unsupported LR scheduler")

    loss_function = Loss(args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank])

    train_loader, test_loader = get_loader(args)

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    global_step = 0
    best_val_recon = 1e8
    best_val_total = 1e8
    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth, map_location="cpu")
        state_dict = model_dict["state_dict"]
        if "module." in list(state_dict.keys())[0]:
            print("Tag 'module.' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=False)
        if "optimizer" in model_dict:
            try:
                optimizer.load_state_dict(model_dict["optimizer"])
                print("Successfully load optimizer")
                print('optimizer', optimizer)
            except Exception as e:
                print(f"Warning: could not load optimizer state: {e}")
        if scheduler is not None and "scheduler" in model_dict and model_dict["scheduler"] is not None:
            try:
                scheduler.load_state_dict(model_dict["scheduler"])
                print("Successfully load scheduler")
            except Exception as e:
                print(f"Warning: could not load scheduler state: {e}")
        if scaler is not None and "scaler" in model_dict and model_dict["scaler"] is not None:
            try:
                scaler.load_state_dict(model_dict["scaler"])
                print("Successfully load scaler")
            except Exception as e:
                print(f"Warning: could not load scaler state: {e}")

        global_step = model_dict["global_step"]
        best_val_recon = model_dict.get("best_val_recon", model_dict.get("best_val", 1e8))
        best_val_total = model_dict.get("best_val_total", 1e8)
        print(
            f"Resumed from {args.resume} at global_step={global_step}, "
            f"best_val_recon={best_val_recon:.4f}, best_val_total={best_val_total:.4f}"
        )

    while global_step < args.num_steps:
        global_step, loss, best_val_recon, best_val_total = train(
            args, global_step, train_loader, best_val_recon, best_val_total, scaler
        )

    if args.rank == 0:
        save_ckp(
            pack_state(global_step, best_val_recon, best_val_total),
            os.path.join(logdir, "model_final_epoch.pt"),
        )
        print(f"Final model saved to {os.path.join(logdir, 'model_final_epoch.pt')}")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()



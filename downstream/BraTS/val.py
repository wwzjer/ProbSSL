import argparse
import os
from functools import partial

import numpy as np
import torch
from trainer import val_epoch
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description="BraTS Task01 evaluation")
parser.add_argument("--pretrained_dir", default="./runs/probssl/", type=str)
parser.add_argument("--pretrained_model_name", default="model.pt", type=str)
parser.add_argument("--data_dir", default="/path/to/Task01_BrainTumour", type=str)
parser.add_argument("--json_list", default="./jsons/task01_brats.json", type=str)
parser.add_argument("--feature_size", default=48, type=int)
parser.add_argument("--infer_overlap", default=0.75, type=float)
parser.add_argument("--sw_batch_size", default=2, type=int)
parser.add_argument("--in_channels", default=4, type=int)
parser.add_argument("--out_channels", default=3, type=int)
parser.add_argument("--roi_x", default=128, type=int)
parser.add_argument("--roi_y", default=128, type=int)
parser.add_argument("--roi_z", default=128, type=int)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--use_checkpoint", action="store_true")
parser.add_argument("--cache_dir", default="", type=str)


def main():
    args = parser.parse_args()
    args.test_mode = True
    args.distributed = False
    args.rank = 0
    args.max_epochs = 1
    args.amp = True

    loader = get_loader(args)
    ckpt_path = os.path.join(args.pretrained_dir, args.pretrained_model_name)
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Data dir: {args.data_dir}")
    print(f"infer_overlap={args.infer_overlap}")

    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
        use_v2=True,
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    model.cuda()
    model.eval()

    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    acc = val_epoch(
        model=model,
        loader=loader,
        epoch=0,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        post_pred=post_pred,
    )
    mean = float(np.mean(acc))
    print(f"mean Dice={mean:.6f} ({100 * mean:.2f}%)")


if __name__ == "__main__":
    main()

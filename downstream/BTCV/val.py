import argparse
import os
from functools import partial

import numpy as np
import torch
from utils.utils import AverageMeter

from monai import data, transforms
from monai.data import decollate_batch, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description="BTCV evaluation")
parser.add_argument("--pretrained_dir", default="./runs/probssl/", type=str)
parser.add_argument("--pretrained_model_name", default="model.pt", type=str)
parser.add_argument("--data_dir", default="/path/to/BTCV", type=str)
parser.add_argument("--json_list", default="dataset_0.json", type=str)
parser.add_argument("--feature_size", default=48, type=int)
parser.add_argument("--infer_overlap", default=0.75, type=float)
parser.add_argument("--sw_batch_size", default=4, type=int)
parser.add_argument("--in_channels", default=1, type=int)
parser.add_argument("--out_channels", default=14, type=int)
parser.add_argument("--a_min", default=-175.0, type=float)
parser.add_argument("--a_max", default=250.0, type=float)
parser.add_argument("--b_min", default=0.0, type=float)
parser.add_argument("--b_max", default=1.0, type=float)
parser.add_argument("--space_x", default=1.5, type=float)
parser.add_argument("--space_y", default=1.5, type=float)
parser.add_argument("--space_z", default=2.0, type=float)
parser.add_argument("--roi_x", default=96, type=int)
parser.add_argument("--roi_y", default=96, type=int)
parser.add_argument("--roi_z", default=96, type=int)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--use_checkpoint", action="store_true")


def get_val_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    val_ds = data.Dataset(data=val_files, transform=val_transform)
    return data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True
    )


def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_val_loader(args)
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
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    state = {k: v for k, v in state.items() if not k.startswith("ds_heads")}
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    acc_func = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    post_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)
    run_acc = AverageMeter()
    case_means = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            data_t, target = batch["image"].to(device), batch["label"].to(device)
            name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            logits = model_inferer(data_t)
            labels = [post_label(x) for x in decollate_batch(target)]
            preds = [post_pred(x) for x in decollate_batch(logits)]
            acc_func.reset()
            acc_func(y_pred=preds, y=labels)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            case_mean = float(np.mean(acc.cpu().numpy()))
            case_means.append(case_mean)
            print(f"[{idx}] {name} mean_dice={case_mean:.6f} running={float(np.mean(run_acc.avg)):.6f}")

    overall = float(np.mean(run_acc.avg))
    print(f"Overall Mean Dice (DiceMetric): {overall:.6f} ({100*overall:.2f}%)")
    print(f"Case-average of per-case means: {float(np.mean(case_means)):.6f}")
    print(f"Per-class mean: {run_acc.avg}")


if __name__ == "__main__":
    main()

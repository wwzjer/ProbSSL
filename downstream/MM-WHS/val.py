import argparse
import os
from functools import partial

import numpy as np
import torch
from utils.data_utils import Convert_WHS_label
from utils.utils import AverageMeter

from monai import data, transforms
from monai.data import decollate_batch, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description="MM-WHS evaluation")
parser.add_argument("--pretrained_dir", default="./runs/probssl/", type=str)
parser.add_argument("--pretrained_model_name", default="model.pt", type=str)
parser.add_argument("--data_dir", default="/path/to/MM-WHS", type=str)
parser.add_argument("--json_list", default="./dataset.json", type=str)
parser.add_argument("--feature_size", default=48, type=int)
parser.add_argument("--infer_overlap", default=0.75, type=float)
parser.add_argument("--sw_batch_size", default=4, type=int)
parser.add_argument("--in_channels", default=1, type=int)
parser.add_argument("--out_channels", default=8, type=int)
parser.add_argument("--a_min", default=0.0, type=float)
parser.add_argument("--a_max", default=1700.0, type=float)
parser.add_argument("--b_min", default=0.0, type=float)
parser.add_argument("--b_max", default=1.0, type=float)
parser.add_argument("--space_x", default=1.5, type=float)
parser.add_argument("--space_y", default=1.5, type=float)
parser.add_argument("--space_z", default=1.5, type=float)
parser.add_argument("--roi_x", default=64, type=int)
parser.add_argument("--roi_y", default=64, type=int)
parser.add_argument("--roi_z", default=64, type=int)
parser.add_argument("--workers", default=4, type=int)
parser.add_argument("--use_checkpoint", action="store_true", default=True)


def get_val_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
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
            Convert_WHS_label(keys="label"),
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
    acc_func = DiceMetric(reduction=MetricReduction.MEAN, get_not_nans=True)
    acc_func_cls = DiceMetric(reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    run_acc = AverageMeter()
    case_means = []
    class_sums = None
    class_counts = None
    class_names = ["BG", "MYO", "LA", "LV", "RA", "RV", "AA", "PA"]

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            data_t, target = batch["image"].to(device), batch["label"].to(device)
            try:
                name = batch["image"].meta["filename_or_obj"][0].split("/")[-1]
            except Exception:
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

            acc_func_cls.reset()
            acc_func_cls(y_pred=preds, y=labels)
            acc_c, nn_c = acc_func_cls.aggregate()
            acc_c = np.asarray(acc_c.cpu().numpy(), dtype=np.float64).reshape(-1)
            nn_c = np.asarray(nn_c.cpu().numpy(), dtype=np.float64).reshape(-1)
            if class_sums is None:
                class_sums = np.zeros_like(acc_c)
                class_counts = np.zeros_like(nn_c)
            class_sums += acc_c * nn_c
            class_counts += nn_c
            print(f"[{idx}] {name} mean_dice={case_mean:.6f} running={float(np.mean(run_acc.avg)):.6f}")

    overall = float(np.mean(run_acc.avg))
    print(f"Overall Mean Dice (DiceMetric): {overall:.6f} ({100*overall:.2f}%)")
    print(f"Case-average of per-case means: {float(np.mean(case_means)):.6f}")
    per_class = class_sums / np.maximum(class_counts, 1e-8)
    print(f"Per-class mean: {per_class}")
    for i, v in enumerate(per_class):
        cname = class_names[i] if i < len(class_names) else f"c{i}"
        print(f"  {cname}: {float(v):.6f}")


if __name__ == "__main__":
    main()

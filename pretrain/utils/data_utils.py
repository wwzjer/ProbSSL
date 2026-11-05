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

import os
import pickle

from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    PersistentDataset,
    SmartCacheDataset,
    load_decathlon_datalist,
)
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)


def _image_only(datalist):
    """Keep image paths only (e.g. drop unused Luna label entries)."""
    return [{"image": item["image"]} for item in datalist]


def get_loader(args):
    json_root = args.json_path
    data_root = args.data_path

    json_btcv = os.path.join(json_root, "btcv.json")
    json_covid = os.path.join(json_root, "dataset_TCIAcovid19_0.json")
    json_luna = os.path.join(json_root, "dataset_LUNA16_0.json")

    dir_btcv = os.path.join(data_root, "BTCV")
    dir_covid = os.path.join(data_root, "TCIAcovid19")
    dir_luna = os.path.join(data_root, "Luna")

    datalist1 = load_decathlon_datalist(json_btcv, False, "training", base_dir=dir_btcv)
    datalist2 = load_decathlon_datalist(json_covid, False, "training", base_dir=dir_covid)
    datalist3 = _image_only(load_decathlon_datalist(json_luna, False, "training", base_dir=dir_luna))
    print("Dataset 1 BTCV: number of data: {}".format(len(datalist1)))
    print("Dataset 2 TCIA Covid19: number of data: {}".format(len(datalist2)))
    print("Dataset 3 Luna: number of data: {}".format(len(datalist3)))

    vallist1 = load_decathlon_datalist(json_btcv, False, "validation", base_dir=dir_btcv)
    vallist2 = load_decathlon_datalist(json_covid, False, "validation", base_dir=dir_covid)
    vallist3 = _image_only(load_decathlon_datalist(json_luna, False, "validation", base_dir=dir_luna))

    datalist = datalist1 + datalist2 + datalist3
    val_files = vallist1 + vallist2 + vallist3
    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))

    num_workers = getattr(args, "num_workers", 16)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForegroundd(
                keys=["image"],
                source_key="image",
                k_divisible=[args.roi_x, args.roi_y, args.roi_z],
            ),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForegroundd(
                keys=["image"],
                source_key="image",
                k_divisible=[args.roi_x, args.roi_y, args.roi_z],
            ),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(
            data=datalist, transform=train_transforms, cache_rate=0.5, num_workers=num_workers
        )
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    elif args.persistent_dataset:
        cache_dir = getattr(args, "cache_dir", None) or "./cache"
        print("Using MONAI Persistent Dataset (cache_dir={})".format(cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        train_ds = PersistentDataset(
            data=datalist,
            transform=train_transforms,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
            cache_dir=cache_dir,
        )
    else:
        print("Using generic dataset")
        train_ds = Dataset(data=datalist, transform=train_transforms)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )

    return train_loader, val_loader

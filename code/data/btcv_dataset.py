from typing import Optional, Sequence, Union

import torch
import torch.distributed as ptdist
import pytorch_lightning as pl

from monai.data import (
    CacheDataset,
    Dataset,
    partition_dataset,
    PersistentDataset,
    load_decathlon_datalist,
    list_data_collate,
    decollate_batch,
)
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    RandShiftIntensityd,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)
from monai.data.utils import pad_list_data_collate


class BTCVDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        cache_dir: str,
        downsample_ratio: Sequence[float],
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        cache_num: int = 0,
        cache_rate: float = 0.0,
        is_ssl: bool = False,
        dist: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.json_path = json_path
        self.cache_dir = cache_dir
        self.downsample_ratio = downsample_ratio
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.cache_num = cache_num
        self.cache_rate = cache_rate
        self.is_ssl = is_ssl
        self.dist = dist

        self.train_list = load_decathlon_datalist(
            base_dir=self.root_dir,
            data_list_file_path=self.json_path,
            is_segmentation=True,
            data_list_key="training",
        )

        self.val_list = load_decathlon_datalist(
            base_dir=self.root_dir,
            data_list_file_path=self.json_path,
            is_segmentation=True,
            data_list_key="validation",
        )

        # self.test_list = load_decathlon_datalist(
        #     base_dir=self.root_dir,
        #     data_list_file_path=self.json_path,
        #     is_segmentation=True,
        #     data_list_key="test",
        # )

    def val_transforms(self, is_ssl=False):
        if not is_ssl:
            transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(
                        keys=["image", "label"],
                        pixdim=self.downsample_ratio,
                        mode=("bilinear", "nearest"),
                    ),
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=-175,
                        a_max=250,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    CropForegroundd(keys=["image", "label"], source_key="image"),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            transforms = self.train_transforms(is_ssl)

        return transforms

    def train_transforms(self, is_ssl=False):
        if not is_ssl:
            transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(
                        keys=["image", "label"],
                        pixdim=self.downsample_ratio,
                        mode=("bilinear", "nearest"),
                    ),
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=-175,
                        a_max=250,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    CropForegroundd(keys=["image", "label"], source_key="image"),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=(96, 96, 96),
                        pos=1,
                        neg=1,
                        num_samples=4,
                        image_key="image",
                        image_threshold=0,
                    ),
                    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10,),
                    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10,),
                    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10,),
                    RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3,),
                    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50,),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            transforms = Compose(
                [
                    # load 4 Nifti images and stack them together
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(
                        keys=["image", "label"],
                        pixdim=self.downsample_ratio,
                        mode=("bilinear", "nearest"),
                    ),
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=-175,
                        a_max=250,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    CropForegroundd(keys=["image", "label"], source_key="image"),
                    RandSpatialCropSamplesd(
                        keys=["image", "label"],
                        roi_size=(96, 96, 96),
                        random_size=False,
                        num_samples=4,
                    ),
                    # RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                    # RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                    ToTensord(keys=["image", "label"]),
                ]
            )

        return transforms

    def test_transforms(self):
        transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=self.downsample_ratio,
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                CenterSpatialCropd(keys=["image", "label"], roi_size=(192, 192, 96)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        return transforms

    def setup(self, stage: Optional[str] = None):
        # Assign Train split(s) for use in Dataloaders
        if stage in [None, "fit"]:
            if self.dist:
                train_partition = partition_dataset(
                    data=self.train_list,
                    num_partitions=ptdist.get_world_size(),
                    shuffle=True,
                    even_divisible=True,
                    drop_last=False,
                )[ptdist.get_rank()]
                valid_partition = partition_dataset(
                    data=self.val_list,
                    num_partitions=ptdist.get_world_size(),
                    shuffle=False,
                    even_divisible=True,
                    drop_last=False,
                )[ptdist.get_rank()]
                # self.cache_num //= ptdist.get_world_size()
            else:
                train_partition = self.train_list
                valid_partition = self.val_list

            if any([self.cache_num, self.cache_rate]) > 0:
                self.train_ds = CacheDataset(
                    train_partition,
                    cache_num=self.cache_num,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                    transform=self.train_transforms(self.is_ssl),
                )
                self.valid_ds = CacheDataset(
                    valid_partition,
                    cache_num=self.cache_num // 4,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                    transform=self.val_transforms(self.is_ssl),
                )
            else:
                self.train_ds = PersistentDataset(
                    train_partition,
                    transform=self.train_transforms(self.is_ssl),
                    cache_dir=self.cache_dir,
                )
                self.valid_ds = PersistentDataset(
                    valid_partition,
                    transform=self.val_transforms(self.is_ssl),
                    cache_dir=self.cache_dir,
                )

        if stage in [None, "test"]:
            if any([self.cache_num, self.cache_rate]) > 0:
                self.test_ds = CacheDataset(
                    self.val_list,
                    cache_num=self.cache_num // 4,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                    transform=self.val_transforms(self.is_ssl),
                )
            else:
                self.test_ds = PersistentDataset(
                    self.val_list,
                    transform=self.val_transforms(self.is_ssl),
                    cache_dir=self.cache_dir,
                )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=pad_list_data_collate,
            # drop_last=False,
            # prefetch_factor=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            # drop_last=False,
            collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            # drop_last=False,
            collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )

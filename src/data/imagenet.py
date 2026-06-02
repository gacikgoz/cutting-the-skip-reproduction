"""ImageNet-1k data loader for supervised training and DINO pretraining."""

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DINODataAugmentation:
    """Multi-crop augmentation for DINO pretraining.

    Produces 2 global crops (224x224) and several local crops (96x96).
    Global crops use a larger scale range; local crops use a smaller one.
    """

    def __init__(
        self,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        num_local_crops: int = 8,
        global_crop_scale: Tuple[float, float] = (0.4, 1.0),
        local_crop_scale: Tuple[float, float] = (0.05, 0.4),
    ):
        self.num_local_crops = num_local_crops

        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Shared color jitter and grayscale (applied probabilistically)
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)

        # GaussianBlur kernel scales with crop size so small crops (e.g. 32px
        # for Tiny ImageNet) don't get a kernel larger than the image.
        def _blur(crop_size: int):
            k = max(3, int(round(crop_size * 0.1)) | 1)  # odd, ~10% of crop
            return transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))

        # --- Global crop transforms (2 crops) ---
        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size, scale=global_crop_scale, interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            _blur(global_crop_size),
            transforms.ToTensor(),
            normalize,
        ])
        self.global_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size, scale=global_crop_scale, interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([_blur(global_crop_size)], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        # --- Local crop transform ---
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crop_size, scale=local_crop_scale, interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            _blur(local_crop_size),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, image):
        """Return a list of crops: [global1, global2, local1, ..., localN]."""
        crops = [
            self.global_transform_1(image),
            self.global_transform_2(image),
        ]
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))
        return crops


def get_train_transform():
    """Standard ImageNet training transform."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform():
    """Standard ImageNet validation transform."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_imagenet_loaders(
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 8,
    distributed: bool = False,
    dino: bool = False,
    num_local_crops: int = 8,
    pin_memory: bool = True,
    global_crop_size: int = 224,
    local_crop_size: int = 96,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create ImageNet train and validation data loaders.

    Args:
        data_root: Path to ImageNet root (should contain 'train/' and 'val/').
        batch_size: Batch size per GPU.
        num_workers: Number of data loading workers.
        distributed: Whether to use DistributedSampler.
        dino: If True, use DINO multi-crop augmentation for training.
        num_local_crops: Number of local crops for DINO augmentation.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        (train_loader, val_loader). val_loader is None when dino=True.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"ImageNet training directory not found at {train_dir}. "
            "Please download ImageNet-1k from https://image-net.org/download.php "
            "and extract it so that data_root contains 'train/' and 'val/' folders."
        )

    # --- Training set ---
    if dino:
        train_transform = DINODataAugmentation(
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            num_local_crops=num_local_crops,
        )
    else:
        train_transform = get_train_transform()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # --- Validation set (skip for DINO pretraining) ---
    val_loader = None
    if not dino and os.path.isdir(val_dir):
        val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transform())
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader

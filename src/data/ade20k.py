"""ADE20K data loader for semantic segmentation evaluation."""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision import transforms
from PIL import Image


ADE20K_NUM_CLASSES = 150
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ADE20KDataset(Dataset):
    """ADE20K semantic segmentation dataset.

    Expects the standard ADEChallengeData2016 layout:
        data_root/
            images/
                training/    (20,210 images)
                validation/  (2,000 images)
            annotations/
                training/
                validation/
    """

    def __init__(
        self,
        data_root: str,
        split: str = "training",
        image_size: int = 480,
    ):
        """
        Args:
            data_root: Root of ADEChallengeData2016.
            split: 'training' or 'validation'.
            image_size: Resize images and masks to this size.
        """
        self.image_dir = os.path.join(data_root, "images", split)
        self.anno_dir = os.path.join(data_root, "annotations", split)

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"ADE20K images not found at {self.image_dir}. "
                "Download from http://sceneparsing.csail.mit.edu/ and extract "
                "so that data_root points to the ADEChallengeData2016 directory."
            )

        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith((".jpg", ".png"))
        ])
        self.image_size = image_size

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        # Annotation files use .png extension
        anno_name = img_name.replace(".jpg", ".png")

        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.anno_dir, anno_name))

        image = self.img_transform(image)

        mask = mask.resize(
            (self.image_size, self.image_size), resample=Image.NEAREST
        )
        # ADE20K annotations are 1-indexed (1..150), convert to 0-indexed (0..149)
        # Pixels with value 0 are unlabeled -> map to ignore index
        mask = torch.from_numpy(np.array(mask)).long()
        mask = mask - 1  # shift: 0 becomes -1 (ignore), 1..150 become 0..149
        mask[mask < 0] = 255  # use 255 as ignore index

        return image, mask


def get_ade20k_loaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 480,
    num_samples: Optional[int] = None,
    distributed: bool = False,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create ADE20K train and val data loaders.

    Args:
        data_root: Path to ADEChallengeData2016 directory.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        image_size: Resize images and masks to this size.
        num_samples: If set, randomly sample this many training images
            (paper uses 3000 due to resource constraints).
        distributed: Whether to use DistributedSampler.
        pin_memory: Pin memory for GPU transfer.
        seed: Random seed for reproducible subsampling.

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = ADE20KDataset(data_root, split="training", image_size=image_size)
    val_dataset = ADE20KDataset(data_root, split="validation", image_size=image_size)

    # Subsample training set if requested
    if num_samples is not None and num_samples < len(train_dataset):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(train_dataset), size=num_samples, replace=False)
        train_dataset = Subset(train_dataset, indices.tolist())

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader

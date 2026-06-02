"""COCO-Stuff data loader for semantic segmentation evaluation."""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision import transforms
from PIL import Image


COCO_STUFF_NUM_CLASSES = 182  # 1..182 in raw PNGs -> 0..181 after our remap.
                              # 0 in PNG = unlabeled -> mapped to 255 (ignore).
                              # The 80+91=171 actual COCO-Stuff classes occupy a
                              # subset of 0..181 with gaps at the discontinued
                              # COCO category IDs (10, 25, 28, ...). Unused
                              # outputs of the linear probe head get no training
                              # samples and are skipped by the IoU computation.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _list_images(image_dir: str) -> List[str]:
    return sorted([
        f for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".png"))
    ])


class COCOStuffDataset(Dataset):
    """COCO-Stuff 164k semantic segmentation dataset.

    Expects one of these layouts under ``data_root``:

    1. Full COCO-Stuff (preferred):
           images/train2017/, images/val2017/
           annotations/train2017/, annotations/val2017/

    2. val2017-only (what ``setup_datasets.sh`` fetches -- the full 19 GB
       train2017 images are NOT redistributable automatically):
           images/val2017/
           annotations/val2017/
       In this case, the "train" and "val" splits below are produced by a
       deterministic 80/20 split of val2017 when ``ids`` is ``None``.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 480,
        ids: Optional[List[str]] = None,
        physical_split: str = "train",
    ):
        """
        Args:
            data_root: Root directory of COCO-Stuff dataset.
            split: 'train' or 'val' -- used for error messages only.
            image_size: Resize images and masks to this size.
            ids: Optional explicit list of image stems (without extension).
                When given, ``physical_split`` selects which folder to read
                the files from. This is used by the val2017-split fallback.
            physical_split: 'train' or 'val' -- the physical folder under
                images/ and annotations/ to read from.
        """
        self.image_dir = os.path.join(data_root, "images", f"{physical_split}2017")
        self.anno_dir = os.path.join(data_root, "annotations", f"{physical_split}2017")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(
                f"COCO-Stuff images not found at {self.image_dir}. "
                "Download from https://github.com/nightrome/cocostuff and set up:\n"
                "  1. COCO 2017 images: https://cocodataset.org/#download\n"
                "  2. COCO-Stuff annotations: "
                "https://github.com/nightrome/cocostuff#downloads\n"
                "Arrange as data_root/images/{train,val}2017/ and "
                "data_root/annotations/{train,val}2017/."
            )

        if not os.path.isdir(self.anno_dir):
            raise FileNotFoundError(
                f"COCO-Stuff annotations not found at {self.anno_dir}."
            )

        if ids is None:
            self.images = _list_images(self.image_dir)
        else:
            # Filter requested ids to ones actually present on disk.
            present = set(os.path.splitext(f)[0] for f in _list_images(self.image_dir))
            self.images = [f"{sid}.jpg" for sid in ids if sid in present]
        if not self.images:
            raise RuntimeError(
                f"COCO-Stuff {split} split is empty at {self.image_dir}."
            )
        self.split = split
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
        # Annotation files: same stem with .png extension
        anno_name = os.path.splitext(img_name)[0] + ".png"

        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.anno_dir, anno_name))

        image = self.img_transform(image)

        mask = mask.resize(
            (self.image_size, self.image_size), resample=Image.NEAREST
        )
        # Raw stuffthingmaps PNG values:
        #   0           -> unlabeled (we map to 255 = ignore_index)
        #   1..91       -> COCO "thing" class IDs (with gaps at 12,26,29,...)
        #   92..182     -> COCO-Stuff "stuff" class IDs (contiguous)
        #   255         -> already ignore in some flavours
        # Linear probe expects targets in [0, num_classes-1]. Subtract 1 from
        # all valid labels so they live in [0, 181], and route unlabeled +
        # already-255 to the cross-entropy ignore_index. Without this remap
        # the NLL kernel asserts t<n_classes and the eval crashes.
        mask = torch.from_numpy(np.array(mask)).long()
        ignore = (mask == 0) | (mask == 255)
        mask = mask - 1               # 1..182 -> 0..181 ; 0 -> -1 (handled below)
        mask[ignore] = 255             # unlabeled (was 0) and pre-existing 255
        return image, mask


def _annotations_have_pngs(data_root: str, split: str) -> bool:
    """Return True iff ``data_root/annotations/<split>2017`` has at least
    one label-map .png on disk.

    The ``src/data/download.py`` path sometimes leaves the annotations
    directory behind in a half-extracted state (directory exists but is
    empty, or the stuffthingmaps zip got hoisted into an unexpected
    subdir). Verifying actual .png files is more reliable than just
    ``os.path.isdir``.
    """
    ann_dir = os.path.join(data_root, "annotations", f"{split}2017")
    if not os.path.isdir(ann_dir):
        return False
    try:
        for f in os.listdir(ann_dir):
            if f.endswith(".png"):
                return True
    except OSError:
        return False
    return False


def _find_annotations_dir(data_root: str) -> Optional[str]:
    """Locate an annotations/<split>2017 that actually has .png files.

    Preference order: val2017 > train2017 (we'd rather use val images +
    val annotations as the base for the fallback split). Returns the
    split name or None.
    """
    for split in ("val", "train"):
        if _annotations_have_pngs(data_root, split):
            return split
    return None


def _val_only_split(data_root: str, seed: int,
                    physical_split: str = "val") -> Tuple[List[str], List[str]]:
    """Deterministic 80/20 split of available image stems.

    Used when only one of train2017/val2017 is available locally (no
    train2017 download fetched automatically, or when the annotations
    were only partially extracted).
    """
    img_dir = os.path.join(data_root, "images", f"{physical_split}2017")
    ann_dir = os.path.join(data_root, "annotations", f"{physical_split}2017")
    img_stems = {os.path.splitext(f)[0] for f in _list_images(img_dir)}
    ann_stems = {os.path.splitext(f)[0] for f in os.listdir(ann_dir)
                 if f.endswith(".png")}
    # Keep only images that actually have a matching label map.
    stems = sorted(img_stems & ann_stems)
    if not stems:
        raise RuntimeError(
            f"No image/annotation pairs found for split {physical_split}2017 "
            f"at {data_root}. images_with_pngs={len(img_stems)} "
            f"annotations_found={len(ann_stems)}"
        )
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(stems))
    n_train = int(0.8 * len(stems))
    train_ids = [stems[i] for i in idx[:n_train]]
    val_ids = [stems[i] for i in idx[n_train:]]
    return train_ids, val_ids


def get_coco_stuff_loaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 480,
    num_samples: Optional[int] = None,
    distributed: bool = False,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create COCO-Stuff train and val data loaders.

    Three layouts are accepted, in decreasing order of preference:

    1. Full: ``annotations/{train,val}2017/`` + ``images/{train,val}2017/``.
    2. Val-only: ``annotations/val2017/`` + ``images/val2017/``. We
       deterministically split val2017 80/20 into train/val.
    3. Train-only: ``annotations/train2017/`` + ``images/train2017/``.
       Same 80/20 split applied to train2017 instead. This triggers when
       the stuff-maps zip only extracted ``train2017`` annotations, which
       happens with certain mirrors of ``stuffthingmaps_trainval2017.zip``.

    If neither annotations/val2017 nor annotations/train2017 has any
    label .png files, we raise a clear error.
    """
    train_img_dir = os.path.join(data_root, "images", "train2017")
    val_img_dir = os.path.join(data_root, "images", "val2017")
    have_full = (
        os.path.isdir(train_img_dir) and os.path.isdir(val_img_dir)
        and _annotations_have_pngs(data_root, "train")
        and _annotations_have_pngs(data_root, "val")
    )

    if have_full:
        train_dataset = COCOStuffDataset(
            data_root, split="train", image_size=image_size,
            physical_split="train",
        )
        val_dataset = COCOStuffDataset(
            data_root, split="val", image_size=image_size,
            physical_split="val",
        )
    else:
        # Pick whichever split actually has both images and .png label maps.
        candidate = None
        for split in ("val", "train"):
            split_img = os.path.join(data_root, "images", f"{split}2017")
            if os.path.isdir(split_img) and _annotations_have_pngs(data_root, split):
                candidate = split
                break
        if candidate is None:
            raise FileNotFoundError(
                f"No usable COCO-Stuff split at {data_root}. Expected at "
                "least one of:\n"
                f"  {data_root}/annotations/val2017/*.png  + "
                f"{data_root}/images/val2017/*.jpg\n"
                f"  {data_root}/annotations/train2017/*.png + "
                f"{data_root}/images/train2017/*.jpg\n"
                "Re-run `python -c \"from src.data.download import "
                "download_coco_stuff; from pathlib import Path; "
                "download_coco_stuff(Path('./data'))\"` to fetch/repair."
            )
        print(
            f"[coco_stuff] full layout missing; using deterministic 80/20 "
            f"split of {candidate}2017 (both images and annotation .pngs "
            f"verified)."
        )
        train_ids, val_ids = _val_only_split(data_root, seed, candidate)
        train_dataset = COCOStuffDataset(
            data_root, split="train", image_size=image_size,
            ids=train_ids, physical_split=candidate,
        )
        val_dataset = COCOStuffDataset(
            data_root, split="val", image_size=image_size,
            ids=val_ids, physical_split=candidate,
        )

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

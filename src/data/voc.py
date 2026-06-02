"""PASCAL VOC 2012 data loader for semantic segmentation evaluation."""

import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from PIL import Image


# 21 classes: background + 20 object categories
VOC_NUM_CLASSES = 21
# Class 255 is the border/ignore label in VOC
VOC_IGNORE_INDEX = 255

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _resolve_voc2012_root(data_root: str) -> str:
    """Find the VOC2012 directory given any of the layouts users pass in.

    Accepts all three common roots:
      * ``<root>`` containing ``VOCdevkit/VOC2012/``
      * ``<root>/VOCdevkit`` (one level down)
      * ``<root>/VOCdevkit/VOC2012`` (the dataset itself)
    """
    candidates = [
        os.path.join(data_root, "VOCdevkit", "VOC2012"),
        os.path.join(data_root, "VOC2012"),
        data_root,
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "JPEGImages")) and \
           os.path.isdir(os.path.join(c, "SegmentationClass")):
            return c
    raise FileNotFoundError(
        f"VOC2012 layout not found under {data_root}. Expected one of:\n"
        f"  {data_root}/VOCdevkit/VOC2012/\n"
        f"  {data_root}/VOC2012/\n"
        f"  {data_root}/ (with JPEGImages/ and SegmentationClass/ inside)\n"
        "Download from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/ "
        "or run `python run.py download --only voc`."
    )


class VOCSegmentation(Dataset):
    """PASCAL VOC 2012 semantic segmentation dataset.

    Reads images + class label maps directly from the filesystem so we don't
    depend on torchvision's rigid ``<root>/VOCdevkit/VOC2012`` expectation.
    """

    SPLIT_FILE = {
        "train":    os.path.join("ImageSets", "Segmentation", "train.txt"),
        "val":      os.path.join("ImageSets", "Segmentation", "val.txt"),
        "trainval": os.path.join("ImageSets", "Segmentation", "trainval.txt"),
    }

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 480,
        download: bool = False,
    ):
        """
        Args:
            data_root: VOC root, VOCdevkit, or VOC2012 dir -- we resolve any.
            split: 'train', 'val', or 'trainval'.
            image_size: Size to resize images to. Use 0 to skip resizing.
            download: Kept for API compatibility; downloading lives in
                ``src/data/download.py`` now, so this flag is ignored.
        """
        del download  # no-op for API compatibility

        if split not in self.SPLIT_FILE:
            raise ValueError(f"Unknown VOC split '{split}'.")

        self.root = _resolve_voc2012_root(data_root)
        split_file = os.path.join(self.root, self.SPLIT_FILE[split])
        if not os.path.isfile(split_file):
            raise FileNotFoundError(
                f"VOC segmentation split file not found: {split_file}"
            )
        with open(split_file) as fh:
            self.ids = [ln.strip() for ln in fh if ln.strip()]
        if not self.ids:
            raise RuntimeError(
                f"VOC split {split} at {split_file} is empty."
            )

        self.image_size = image_size
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        img_path = os.path.join(self.root, "JPEGImages", f"{image_id}.jpg")
        mask_path = os.path.join(
            self.root, "SegmentationClass", f"{image_id}.png"
        )

        image = Image.open(img_path).convert("RGB")
        target = Image.open(mask_path)

        image = self.img_transform(image)
        target = target.resize(
            (self.image_size, self.image_size), resample=Image.NEAREST
        )
        target = torch.from_numpy(np.array(target)).long()
        # VOC uses 255 as the border/ignore region.
        target[target == 255] = VOC_IGNORE_INDEX
        return image, target


def get_voc_loaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 480,
    distributed: bool = False,
    download: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create PASCAL VOC 2012 train and val data loaders.

    Args:
        data_root: Root directory containing the VOC dataset.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        image_size: Resize images and labels to this size.
        distributed: Whether to use DistributedSampler.
        download: Whether to download the dataset.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = VOCSegmentation(
        data_root, split="train", image_size=image_size, download=download
    )
    val_dataset = VOCSegmentation(
        data_root, split="val", image_size=image_size, download=False
    )

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


# ---------------------------------------------------------------------------
# VOC 2012 with bounding-box annotations (used by TokenCut)
# ---------------------------------------------------------------------------

class VOCBBox(Dataset):
    """PASCAL VOC 2012 with a single largest bounding box per image.

    Used by ``src/eval/tokencut.py`` to compute CorLoc: the TokenCut output
    is a single predicted box and we need one ground-truth box per image.
    We pick the largest-area box in the XML annotation, matching the
    protocol used by Wang et al. (TokenCut).

    Images are resized to a fixed ``image_size`` and the bounding boxes are
    rescaled accordingly.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "trainval",
        image_size: int = 224,
    ):
        """
        Parameters
        ----------
        data_root : str
            Any of: project data root, ``VOCdevkit/``, or ``VOC2012/``.
        split : str
            'train', 'val' or 'trainval'.
        image_size : int
            Target image size (square).
        """
        self.root = _resolve_voc2012_root(data_root)

        split_file = os.path.join(
            self.root, "ImageSets", "Main", f"{split}.txt"
        )
        if not os.path.isfile(split_file):
            raise FileNotFoundError(
                f"VOC split file not found: {split_file}"
            )

        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.image_size = image_size
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def _load_largest_bbox(self, xml_path: str, orig_w: int, orig_h: int) -> Tuple[int, int, int, int]:
        """Return the largest (by area) non-difficult bbox, scaled to ``image_size``."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        best = None
        best_area = 0
        for obj in root.findall("object"):
            difficult = obj.find("difficult")
            if difficult is not None and int(difficult.text) == 1:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x1 = float(bb.find("xmin").text)
            y1 = float(bb.find("ymin").text)
            x2 = float(bb.find("xmax").text)
            y2 = float(bb.find("ymax").text)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)

        if best is None:
            # No non-difficult objects -- fall back to the full image.
            return (0, 0, self.image_size, self.image_size)

        # Rescale the box to ``image_size``.
        sx = self.image_size / orig_w
        sy = self.image_size / orig_h
        x1, y1, x2, y2 = best
        return (
            int(round(x1 * sx)),
            int(round(y1 * sy)),
            int(round(x2 * sx)),
            int(round(y2 * sy)),
        )

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        img_path = os.path.join(self.root, "JPEGImages", f"{image_id}.jpg")
        xml_path = os.path.join(self.root, "Annotations", f"{image_id}.xml")

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = self.img_transform(image)

        bbox = self._load_largest_bbox(xml_path, orig_w, orig_h)
        bbox_tensor = torch.tensor(bbox, dtype=torch.int64)

        return image, bbox_tensor


def get_voc_bbox_dataset(
    data_root: str,
    split: str = "trainval",
    image_size: int = 224,
) -> VOCBBox:
    """Convenience constructor for :class:`VOCBBox`."""
    return VOCBBox(data_root=data_root, split=split, image_size=image_size)

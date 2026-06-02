"""Dataset auto-download helpers.

Everything this module fetches is legally redistributable and does not
require a registered account. ImageNet-1k is the notable exception -- it
requires manual registration at https://image-net.org/ so we cannot
automate that download.

All functions are idempotent: if the expected target directory already
exists with the expected contents, the download is skipped.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional


LOG = logging.getLogger("cutting-the-skip.download")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Download ``url`` to ``dest`` with a human-readable progress bar.

    Uses ``urllib`` so it has no extra dependency. A partial download is
    resumed if ``dest.part`` already exists with content.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "ceng502-cutting-the-skip/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        LOG.info("Resuming %s from byte %d", url, existing)
    else:
        LOG.info("Downloading %s -> %s", url, dest)

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total_str = resp.headers.get("Content-Length")
            total = int(total_str) + existing if total_str else None
            with tmp.open("ab") as f:
                got = existing
                last_log = 0.0
                import time as _time
                t0 = _time.time()
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
                    now = _time.time()
                    if now - last_log > 2.0:
                        if total:
                            pct = 100.0 * got / total
                            mb = got / 1e6
                            LOG.info("  %.1f%%  (%.1f MB / %.1f MB)",
                                     pct, mb, total / 1e6)
                        else:
                            LOG.info("  %.1f MB", got / 1e6)
                        last_log = now
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    tmp.rename(dest)


def _extract(archive: Path, dest: Path) -> None:
    """Extract a ``.zip`` / ``.tar`` / ``.tar.gz`` archive."""
    dest.mkdir(parents=True, exist_ok=True)
    LOG.info("Extracting %s -> %s", archive, dest)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise RuntimeError(f"Unknown archive format: {archive}")


def _ensure_dataset(
    name: str,
    target: Path,
    url: str,
    archive_name: str,
    check: Callable[[Path], bool],
    data_root: Path,
) -> bool:
    """Download + extract ``url`` to ``target`` if ``check`` is False.

    Returns ``True`` if the dataset is present after this call, ``False`` if
    the download failed (caller decides whether to continue).
    """
    if check(target):
        LOG.info("[%s] already present at %s", name, target)
        return True

    archive = data_root / "_downloads" / archive_name
    try:
        if not archive.is_file():
            _download_file(url, archive)
        _extract(archive, target.parent if target.parent != data_root else data_root)
    except Exception as exc:
        LOG.error("[%s] download failed: %s", name, exc)
        return False

    if not check(target):
        LOG.error("[%s] post-extract check failed at %s", name, target)
        return False

    LOG.info("[%s] ready at %s", name, target)
    return True


# ---------------------------------------------------------------------------
# Per-dataset URLs and checks
# ---------------------------------------------------------------------------

# Tiny ImageNet: 200-class ImageNet-1k subset with 500 training + 50 val
# + 50 test images per class (native 64x64 JPEGs, ~250 MB). This is our
# default "bigger ImageNet" dataset that fits in a T4 budget while still
# exercising multi-class training at real ImageNet scale.
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

# ImageNette: a 10-class ImageNet subset (~340 MB) from fast.ai. Has the
# identical train/<class>/*.JPEG val/<class>/*.JPEG layout as ImageNet-1k so
# every ImageFolder-based loader works unchanged. Kept as a smaller fallback.
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

VOC_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
    "VOCtrainval_11-May-2012.tar"
)
ADE20K_URL = (
    "http://data.csail.mit.edu/places/ADEchallenge/"
    "ADEChallengeData2016.zip"
)
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOT_URL = (
    "http://images.cocodataset.org/annotations/"
    "annotations_trainval2017.zip"
)
COCO_STUFF_ANNOT_URL = (
    "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/"
    "cocostuffdataset/stuffthingmaps_trainval2017.zip"
)


def _tiny_imagenet_ready(root: Path) -> bool:
    """Check for the ImageFolder-normalized Tiny ImageNet layout."""
    train = root / "train"
    val = root / "val"
    if not (train.is_dir() and val.is_dir()):
        return False
    # After normalization, val has per-class subdirs, not an images/ folder.
    if (val / "images").is_dir():
        return False
    return any(val.iterdir())


def _normalize_tiny_imagenet(root: Path) -> None:
    """Convert Tiny ImageNet's shipped layout into an ImageFolder layout.

    Ships as:
        tiny-imagenet-200/
          train/<wnid>/images/*.JPEG  (+ <wnid>_boxes.txt)
          val/images/*.JPEG           (+ val_annotations.txt)

    We normalize to:
        tiny-imagenet-200/
          train/<wnid>/*.JPEG
          val/<wnid>/*.JPEG

    Idempotent.
    """
    train = root / "train"
    val = root / "val"

    # --- Flatten train/<wnid>/images/*.JPEG -> train/<wnid>/*.JPEG ---
    if train.is_dir():
        for wnid_dir in train.iterdir():
            if not wnid_dir.is_dir():
                continue
            images_sub = wnid_dir / "images"
            if images_sub.is_dir():
                for f in images_sub.iterdir():
                    if f.is_file():
                        target = wnid_dir / f.name
                        if not target.exists():
                            f.rename(target)
                try:
                    images_sub.rmdir()
                except OSError:
                    pass
            # Remove the unused bboxes txt file.
            for leftover in wnid_dir.glob("*_boxes.txt"):
                try:
                    leftover.unlink()
                except OSError:
                    pass

    # --- Reorganize val/images/*.JPEG by wnid using val_annotations.txt ---
    val_images = val / "images"
    val_annot = val / "val_annotations.txt"
    if val_images.is_dir() and val_annot.is_file():
        mapping = {}
        with val_annot.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    fname, wnid = parts[0], parts[1]
                    mapping[fname] = wnid

        for fname, wnid in mapping.items():
            src = val_images / fname
            if not src.is_file():
                continue
            dst_dir = val / wnid
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / fname
            if not dst.exists():
                src.rename(dst)

        # Clean up the emptied images/ directory + annotations file.
        try:
            val_images.rmdir()
        except OSError:
            pass
        try:
            val_annot.unlink()
        except OSError:
            pass


def download_tiny_imagenet(data_root: Path) -> bool:
    """Download + normalize Tiny ImageNet (200 classes, 100k training images)."""
    target = data_root / "tiny-imagenet-200"

    if _tiny_imagenet_ready(target):
        LOG.info("[TinyImageNet] already present at %s", target)
    else:
        ok = _ensure_dataset(
            name="TinyImageNet",
            target=target,
            url=TINY_IMAGENET_URL,
            archive_name="tiny-imagenet-200.zip",
            check=lambda p: (p / "train").is_dir() and (p / "val").is_dir(),
            data_root=data_root,
        )
        if not ok:
            return False
        _normalize_tiny_imagenet(target)
        if not _tiny_imagenet_ready(target):
            LOG.error("[TinyImageNet] post-normalize check failed at %s", target)
            return False
        LOG.info("[TinyImageNet] normalized to ImageFolder layout at %s", target)

    # Provide a stable ``imagenet`` alias so paths used elsewhere still work.
    alias = data_root / "imagenet"
    if not alias.exists():
        try:
            alias.symlink_to(target.resolve())
        except OSError:
            shutil.copytree(target, alias)
    return True


def download_imagenette(data_root: Path) -> bool:
    """Download ImageNette (a small, redistributable ImageNet subset)."""
    target = data_root / "imagenette2-320"
    ok = _ensure_dataset(
        name="ImageNette",
        target=target,
        url=IMAGENETTE_URL,
        archive_name="imagenette2-320.tgz",
        check=lambda p: (p / "train").is_dir() and (p / "val").is_dir(),
        data_root=data_root,
    )
    if not ok:
        return False

    # ImageNette ships with directories named "train/" and "val/" at the top
    # level of the extracted folder. Both are already ImageFolder-compatible.
    return True


def download_voc(data_root: Path) -> bool:
    """Download PASCAL VOC 2012 (trainval).

    The official tarball contains ``VOCdevkit/VOC2012/...`` at its root, so we
    must extract into ``data_root`` itself (not ``data_root/VOCdevkit``) to
    avoid a nested ``data_root/VOCdevkit/VOCdevkit/...`` layout.
    """
    target = data_root / "VOCdevkit" / "VOC2012"

    def _check(p: Path) -> bool:
        return (p / "JPEGImages").is_dir() and (p / "SegmentationClass").is_dir()

    if _check(target):
        LOG.info("[VOC2012] already present at %s", target)
        return True

    archive = data_root / "_downloads" / "VOCtrainval_11-May-2012.tar"
    try:
        if not archive.is_file():
            _download_file(VOC_URL, archive)
        # Extract into data_root -- the tar already has VOCdevkit/ at the top.
        _extract(archive, data_root)
    except Exception as exc:
        LOG.error("[VOC2012] download failed: %s", exc)
        return False

    if not _check(target):
        LOG.error("[VOC2012] post-extract check failed at %s", target)
        return False
    LOG.info("[VOC2012] ready at %s", target)
    return True


def download_ade20k(data_root: Path) -> bool:
    """Download ADE20K (ADEChallengeData2016)."""
    target = data_root / "ade20k" / "ADEChallengeData2016"
    ok = _ensure_dataset(
        name="ADE20K",
        target=target,
        url=ADE20K_URL,
        archive_name="ADEChallengeData2016.zip",
        check=lambda p: (p / "images" / "training").is_dir()
                        and (p / "annotations" / "training").is_dir(),
        data_root=data_root / "ade20k",
    )
    return ok


def download_coco_val(data_root: Path) -> bool:
    """Download COCO val2017 images + annotations (used by PCA viz + TokenCut)."""
    root = data_root / "coco"
    root.mkdir(parents=True, exist_ok=True)

    images_ok = _ensure_dataset(
        name="COCO val2017 images",
        target=root / "val2017",
        url=COCO_VAL_URL,
        archive_name="val2017.zip",
        check=lambda p: p.is_dir()
                        and any(p.glob("*.jpg")),
        data_root=root,
    )
    annot_ok = _ensure_dataset(
        name="COCO 2017 annotations",
        target=root / "annotations",
        url=COCO_ANNOT_URL,
        archive_name="annotations_trainval2017.zip",
        check=lambda p: p.is_dir()
                        and (p / "instances_val2017.json").is_file(),
        data_root=root,
    )
    return images_ok and annot_ok


def download_coco_stuff(data_root: Path) -> bool:
    """Download COCO-Stuff stuffthingmaps (uses COCO 2017 images).

    Requires COCO images to already be present (``download_coco_val``). The
    stuffthingmaps zip layout is flat -- it extracts ``train2017/`` and
    ``val2017/`` directories straight into the destination, so we extract
    into the ``annotations`` target directly.
    """
    root = data_root / "coco_stuff"
    images_src = data_root / "coco" / "val2017"
    if not images_src.is_dir():
        LOG.warning(
            "COCO-Stuff needs COCO 2017 images first. "
            "Call download_coco_val() first."
        )
        return False

    # Expected layout used by src/data/coco_stuff.py:
    #   coco_stuff/images/val2017/  (symlink from coco/val2017)
    #   coco_stuff/annotations/val2017/  (from stuffthingmaps)
    (root / "images").mkdir(parents=True, exist_ok=True)
    val_img_link = root / "images" / "val2017"
    if not val_img_link.exists():
        try:
            val_img_link.symlink_to(images_src.resolve())
        except OSError:
            shutil.copytree(images_src, val_img_link)

    ann_dir = root / "annotations"

    def _has_any_png(p: Path) -> bool:
        """True iff ``p`` is a dir that contains at least one .png file."""
        if not p.is_dir():
            return False
        try:
            for f in p.iterdir():
                if f.suffix.lower() == ".png":
                    return True
        except OSError:
            pass
        return False

    def _check(p: Path) -> bool:
        """True iff ``p/val2017`` (or a post-hoist equivalent) has .pngs.

        The old version only verified directory existence, which let
        half-extracted layouts look 'ready' and then crash the downstream
        segmentation loader. This version insists on at least one .png.
        """
        if _has_any_png(p / "val2017"):
            return True
        # Some zips ship a top-level wrapper folder; hoist its contents
        # up into annotations/ so downstream expects annotations/val2017.
        for wrapper in ("stuffthingmaps_trainval2017", "stuffthingmaps"):
            nested_val = p / wrapper / "val2017"
            if _has_any_png(nested_val):
                parent = nested_val.parent
                for sub in parent.iterdir():
                    dst = p / sub.name
                    if dst.exists():
                        continue
                    sub.rename(dst)
                try:
                    parent.rmdir()
                except OSError:
                    pass
                return _has_any_png(p / "val2017")
        return False

    if _check(ann_dir):
        LOG.info("[COCO-Stuff] already present at %s", ann_dir)
        return True

    archive = root / "_downloads" / "stuffthingmaps_trainval2017.zip"
    try:
        if not archive.is_file():
            _download_file(COCO_STUFF_ANNOT_URL, archive)
        ann_dir.mkdir(parents=True, exist_ok=True)
        _extract(archive, ann_dir)
    except Exception as exc:
        LOG.error("[COCO-Stuff] download failed: %s", exc)
        # Don't remove the half-extracted tree -- src/data/coco_stuff.py's
        # loader will pick up whichever split does have .pngs (train2017
        # if val2017 is missing, or vice versa) via its own fallback.
        if _has_any_png(ann_dir / "val2017") or _has_any_png(ann_dir / "train2017"):
            LOG.warning(
                "[COCO-Stuff] partial extract usable -- loader will split "
                "whichever of train2017/val2017 is complete."
            )
            return True
        return False

    if not _check(ann_dir):
        # Extraction "succeeded" but val2017/ had no .pngs. Accept a
        # train-only extract if it exists -- the dataset loader handles it.
        if _has_any_png(ann_dir / "train2017"):
            LOG.warning(
                "[COCO-Stuff] val2017 annotations missing but train2017 "
                "annotations present -- loader will 80/20-split train2017."
            )
            return True
        LOG.error("[COCO-Stuff] post-extract check failed at %s", ann_dir)
        return False
    LOG.info("[COCO-Stuff] ready at %s", ann_dir)
    return True


# ---------------------------------------------------------------------------
# Top-level entry-point used by run.py
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "tiny_imagenet": download_tiny_imagenet,
    "imagenette": download_imagenette,
    "voc": download_voc,
    "ade20k": download_ade20k,
    "coco": download_coco_val,
    "coco_stuff": download_coco_stuff,
}


def download_all(data_root: Path, which: Optional[list] = None) -> dict:
    """Download every redistributable dataset to ``data_root``.

    Parameters
    ----------
    data_root : Path
        Root directory under which datasets are placed.
    which : list or None
        Subset of ``DOWNLOADERS`` keys. Defaults to all.

    Returns
    -------
    dict
        Mapping ``name -> success(bool)``.
    """
    data_root = Path(data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    # NB: coco_stuff depends on coco, so we enforce order. Tiny ImageNet
    # runs first because it establishes the ``imagenet`` alias used by
    # downstream training scripts.
    order = ["tiny_imagenet", "imagenette", "voc", "ade20k", "coco", "coco_stuff"]
    targets = which or order

    results: dict = {}
    for name in order:
        if name not in targets:
            continue
        try:
            results[name] = DOWNLOADERS[name](data_root)
        except Exception as exc:
            LOG.exception("[%s] uncaught error during download", name)
            results[name] = False
    return results

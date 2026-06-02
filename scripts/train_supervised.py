#!/usr/bin/env python3
"""
Supervised training of ViT-Base on ImageNet-1k.

Reproduces Table 1 and Figure 1 from
"Cutting the Skip: Training Residual-Free Transformers".

Supports three modes:
  - skip          : standard ViT with residual connections
  - skipless      : residual-free ViT (standard init)
  - skipless_init : residual-free ViT with the paper's skipless initialization

Usage:
    python scripts/train_supervised.py --config configs/supervised_vit_base.yaml
    python scripts/train_supervised.py --mode skipless_init --epochs 300 --data_dir /data/imagenet
"""

from __future__ import annotations

import argparse
import datetime
import math
import os

# Reduce CUDA allocator fragmentation. Without this, a 30-epoch SOAP run
# on ViT-Base can leave 7-8 GB reserved-but-unallocated after training,
# which is enough to OOM the first validation batch on a 24 GB L4.
# User can override by setting the env var before invoking the script.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import sys
from typing import Optional
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    # Preferred (PyTorch >=2.3): unified torch.amp API that supports
    # both CUDA and CPU (even though we only use AMP on CUDA here).
    from torch.amp import GradScaler as _GradScalerNew, autocast as _autocast_new

    def autocast(enabled: bool):
        return _autocast_new("cuda", enabled=enabled)

    def GradScaler(enabled: bool):
        return _GradScalerNew("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # legacy PyTorch fallback
from torch.utils.tensorboard import SummaryWriter

import yaml

# ---------------------------------------------------------------------------
# Project imports -- scripts/ is a sibling of src/, so we add project root.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.vit import vit_base, vit_small
from src.data.imagenet import get_imagenet_loaders

_MODEL_FACTORIES = {"vit_base": vit_base, "vit_small": vit_small}

# Optional imports that may not exist yet -- guarded so the script still
# loads even when these modules are stubs.
try:
    from src.models.skipless_init import apply_skipless_init
except ImportError:
    apply_skipless_init = None

try:
    from src.optimizers.soap import SOAP
except ImportError:
    SOAP = None

# ---------------------------------------------------------------------------
# Augmentation helpers (timm-based)
# ---------------------------------------------------------------------------
try:
    from timm.data.mixup import Mixup
    from timm.data.auto_augment import rand_augment_transform
    from timm.data.random_erasing import RandomErasing
    from timm.loss import SoftTargetCrossEntropy
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """Compute top-k accuracy for the given predictions and targets."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size).item())
    return results


# ---------------------------------------------------------------------------
# Training augmentation transform (with RandAugment + RandomErasing)
# ---------------------------------------------------------------------------

def build_train_transform(args):
    """Build training transform with RandAugment and RandomErasing.

    Falls back to basic augmentation if timm is unavailable.
    """
    from torchvision import transforms
    from src.data.imagenet import IMAGENET_MEAN, IMAGENET_STD

    if HAS_TIMM:
        # RandAugment string config
        ra_config = f"rand-m{args.randaugment_magnitude}-n{args.randaugment_num_ops}-mstd0.5"
        primary_tfl = [
            transforms.RandomResizedCrop(args.img_size, interpolation=3),  # BICUBIC
            transforms.RandomHorizontalFlip(),
        ]
        secondary_tfl = [
            rand_augment_transform(ra_config, {}),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        if args.random_erase_prob > 0:
            secondary_tfl.append(
                RandomErasing(
                    probability=args.random_erase_prob,
                    mode="pixel",
                    device="cpu",
                )
            )
        return transforms.Compose(primary_tfl + secondary_tfl)
    else:
        # Fallback: basic augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    """Cosine learning rate schedule with linear warmup."""
    warmup_steps = args.warmup_epochs * total_steps_in_epoch
    current_step = epoch * total_steps_in_epoch + step_in_epoch
    total_steps = args.epochs * total_steps_in_epoch

    if current_step < warmup_steps:
        lr = args.lr * current_step / max(warmup_steps, 1)
    else:
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    mixup_fn,
    criterion,
    epoch,
    args,
    writer,
    global_step,
):
    model.train()
    total_steps_in_epoch = len(train_loader)

    running_loss = 0.0
    num_updates = 0
    optimizer.zero_grad()

    bad_batches = 0
    bad_steps = 0
    for step, (images, targets) in enumerate(train_loader):
        # LR schedule
        lr = adjust_learning_rate(optimizer, epoch, step, total_steps_in_epoch, args)

        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)

        # Mixup / CutMix
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        # Forward pass
        with autocast(enabled=args.amp):
            logits = model(images)
            loss = criterion(logits, targets)
            loss = loss / args.accum_steps  # scale for gradient accumulation

        # Skipless ViTs with standard init can produce Inf/NaN logits in the
        # first steps; dropping those batches is safer than poisoning the
        # optimizer state (and the scaler will lower the loss scale anyway).
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            bad_batches += 1
            continue

        # Backward pass
        scaler.scale(loss).backward()

        # Optimizer step (with gradient accumulation)
        if (step + 1) % args.accum_steps == 0 or (step + 1) == total_steps_in_epoch:
            try:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    # Checking grads *before* clipping lets us skip a step
                    # cleanly when a skipless-standard-init run produces
                    # NaN grads -- otherwise clip_grad_norm_ leaves NaN grads
                    # and scaler.step / optimizer.step would try to apply them.
                    grad_finite = True
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            grad_finite = False
                            break
                    if grad_finite:
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                            error_if_nonfinite=False,
                        )
                        scaler.step(optimizer)
                    else:
                        # Let scaler know we detected inf so it lowers scale.
                        scaler.update()
                        bad_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        continue
                else:
                    scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                num_updates += 1
            except Exception as exc:
                # Defensive: a single bad optimizer step should never kill the
                # whole run. Log, zero grads, try to lower the AMP scale, move on.
                print(f"  [warn] optimizer step failed at epoch={epoch+1} "
                      f"step={step}: {type(exc).__name__}: {exc}")
                try:
                    scaler.update()
                except Exception:
                    pass
                optimizer.zero_grad(set_to_none=True)
                bad_steps += 1
                global_step += 1
                continue

        # Logging
        step_loss = loss.item() * args.accum_steps  # un-scale for logging
        running_loss += step_loss
        global_step += 1

        if step % args.log_interval == 0:
            writer.add_scalar("train/loss", step_loss, global_step)
            writer.add_scalar("train/lr", lr, global_step)
            print(
                f"  Epoch [{epoch+1}/{args.epochs}] "
                f"Step [{step}/{total_steps_in_epoch}] "
                f"Loss: {step_loss:.4f}  LR: {lr:.6f}"
            )

    avg_loss = running_loss / total_steps_in_epoch
    if bad_batches or bad_steps:
        print(
            f"  [epoch {epoch+1}] bad_batches={bad_batches} "
            f"bad_steps={bad_steps} of {total_steps_in_epoch} "
            f"(non-finite loss/grad were skipped cleanly)"
        )
    return avg_loss, global_step


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, args):
    """Run a full-dataset validation pass.

    SOAP+ViT-Base at batch 256 on a 24 GB L4 can OOM in the attention
    softmax because of memory fragmentation after training, so we:
      * ``empty_cache`` before starting,
      * catch ``torch.cuda.OutOfMemoryError`` and halve the batch size,
      * retry up to three times (bs -> bs/2 -> bs/4 -> bs/8),
      * log every step so unhealthy runs are visible.
    """
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    current_loader = val_loader
    last_exc: Optional[Exception] = None
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            return _validate_once(model, current_loader, args)
        except torch.cuda.OutOfMemoryError as exc:
            last_exc = exc
            bs = getattr(current_loader, "batch_size", None)
            if attempt >= max_attempts or bs is None or bs <= 1:
                # No further retries possible / allowed; propagate.
                break
            new_bs = max(1, bs // 2)
            print(f"  [warn] validate() OOM at bs={bs} "
                  f"({type(exc).__name__}); retrying at bs={new_bs} "
                  f"(attempt {attempt + 1}/{max_attempts})")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current_loader = torch.utils.data.DataLoader(
                current_loader.dataset,
                batch_size=new_bs,
                shuffle=False,
                num_workers=current_loader.num_workers,
                pin_memory=current_loader.pin_memory,
            )
    raise last_exc  # type: ignore[misc]


def _validate_once(model, val_loader, args):
    total_correct_1 = 0.0
    total_correct_5 = 0.0
    total_samples = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    for images, targets in val_loader:
        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)

        with autocast(enabled=args.amp):
            logits = model(images)
            loss = criterion(logits, targets)

        # Replace non-finite logits so topk / accuracy don't blow up when a
        # skipless model produces Inf/NaN during validation.
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        num_classes = logits.size(1)
        k5 = min(5, num_classes)
        acc1, acck = accuracy(logits.float(), targets, topk=(1, k5))
        batch_size = targets.size(0)
        total_correct_1 += acc1 * batch_size / 100.0
        total_correct_5 += acck * batch_size / 100.0
        total_samples += batch_size
        loss_val = loss.item() if torch.isfinite(loss) else 0.0
        total_loss += loss_val * batch_size

    if total_samples == 0:
        return 0.0, 0.0, 0.0
    top1 = 100.0 * total_correct_1 / total_samples
    top5 = 100.0 * total_correct_5 / total_samples
    avg_loss = total_loss / total_samples
    return top1, top5, avg_loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"  Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scaler=None):
    print(f"  Resuming from checkpoint: {filepath}")
    ckpt = torch.load(filepath, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    global_step = ckpt.get("global_step", 0)
    best_acc = ckpt.get("best_acc", 0.0)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return start_epoch, global_step, best_acc


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised ViT-Base training on ImageNet-1k"
    )
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file")

    # Model
    parser.add_argument("--model", type=str, default="vit_base")
    parser.add_argument(
        "--mode", type=str, default="skip",
        choices=["skip", "skipless", "skipless_init"],
        help="skip=standard ViT, skipless=no residual, skipless_init=no residual + paper init",
    )
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)

    # Dataset
    parser.add_argument("--data_dir", type=str, default="/data/imagenet")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=0,
                        help="Batch size used at validation. Default 0 = "
                             "use a conservative size (half of train batch "
                             "for SOAP, same as train for AdamW) so a 24 GB "
                             "L4 doesn't OOM after a 30-epoch SOAP run. "
                             "Pass a positive integer to override.")
    parser.add_argument("--num_train_samples", type=int, default=0,
                        help="If >0, randomly subsample the ImageFolder "
                             "training set to this size (seeded by --seed).")
    parser.add_argument("--num_val_samples", type=int, default=0,
                        help="If >0, randomly subsample the ImageFolder "
                             "validation set to this size (seeded by --seed). "
                             "Useful for fast CPU smoke sweeps where a full "
                             "10k-image val set dominates wall-clock.")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--accum_steps", type=int, default=4)

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "soap"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.3)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--soap_precondition_frequency", type=int, default=10)

    # Schedule
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-5)

    # Augmentation
    parser.add_argument("--randaugment_num_ops", type=int, default=2)
    parser.add_argument("--randaugment_magnitude", type=int, default=9)
    parser.add_argument("--mixup_alpha", type=float, default=0.8)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--random_erase_prob", type=float, default=0.25)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # Regularization
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--attn_drop_rate", type=float, default=0.0)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)

    # Mixed precision
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")

    # Logging
    parser.add_argument("--output_dir", type=str, default="./output/supervised_vit_base")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' (default), 'cuda', 'cpu', or 'cuda:N'.")

    args = parser.parse_args()

    # Load YAML config and apply as defaults (CLI overrides YAML).
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        # Re-parse with YAML values as defaults
        parser.set_defaults(**cfg)
        args = parser.parse_args()

    # Handle --no_amp flag
    if args.no_amp:
        args.amp = False

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # --- Output dirs ---
    run_name = f"{args.model}_{args.mode}_{args.optimizer}_ep{args.epochs}"
    output_dir = os.path.join(args.output_dir, run_name)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))

    print("=" * 72)
    print("Supervised ViT-Base Training -- Cutting the Skip")
    print("=" * 72)
    print(f"  Mode       : {args.mode}")
    print(f"  Optimizer  : {args.optimizer}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size} x {args.accum_steps} accum = {args.batch_size * args.accum_steps} effective")
    print(f"  LR         : {args.lr}")
    print(f"  AMP        : {args.amp}")
    print(f"  Output     : {output_dir}")
    print("=" * 72)

    # --- Build model ---
    skipless = args.mode in ("skipless", "skipless_init")
    drop_path_rate = 0.0 if skipless else args.drop_path_rate

    model_factory = _MODEL_FACTORIES.get(args.model)
    if model_factory is None:
        raise ValueError(
            f"Unknown --model {args.model!r}; expected one of "
            f"{sorted(_MODEL_FACTORIES)}."
        )
    model = model_factory(
        skipless=skipless,
        num_classes=args.num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        drop_path_rate=drop_path_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_rate=args.drop_rate,
    )

    # Apply skipless initialization if requested
    if args.mode == "skipless_init":
        if apply_skipless_init is None:
            raise ImportError(
                "src.models.skipless_init is not available. "
                "Please implement apply_skipless_init() in src/models/skipless_init.py."
            )
        print("  Applying skipless initialization...")
        apply_skipless_init(model)

    # Pick device: honor --device if given, otherwise auto-detect. This also
    # makes the script runnable on CPU-only hosts (useful for CI/smoke tests).
    if getattr(args, "device", None) in (None, "", "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"  [warn] device={args.device!r} but no CUDA available; falling back to CPU")
        args.device = "cpu"
    # AMP only makes sense on CUDA.
    if args.amp and args.device == "cpu":
        print("  [warn] AMP disabled because device=cpu")
        args.amp = False
    model = model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Device     : {args.device}")
    print(f"  Model parameters: {num_params:.1f}M")

    # --- Build optimizer ---
    # Separate weight-decay groups: no decay for biases and layer-norm params.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=tuple(args.betas),
        )
    elif args.optimizer == "soap":
        if SOAP is None:
            raise ImportError(
                "src.optimizers.soap is not available. "
                "Please implement the SOAP optimizer in src/optimizers/soap.py."
            )
        soap_betas = (0.9, 0.95)  # paper-specified betas for SOAP
        optimizer = SOAP(
            param_groups,
            lr=args.lr,
            betas=soap_betas,
            precondition_frequency=args.soap_precondition_frequency,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # --- Mixed precision scaler ---
    scaler = GradScaler(enabled=args.amp)

    # --- Build data loaders ---
    # We replace the default train transform with our augmented version.
    from torchvision import datasets
    from src.data.imagenet import get_val_transform, IMAGENET_MEAN, IMAGENET_STD

    train_transform = build_train_transform(args)
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"ImageNet training directory not found at {train_dir}. "
            "Please set --data_dir to the ImageNet root."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    # Optionally subsample the training set (deterministic given --seed) for
    # fast sanity sweeps on CPU without touching the dataset on disk.
    if getattr(args, "num_train_samples", 0) and args.num_train_samples > 0 \
            and args.num_train_samples < len(train_dataset):
        import numpy as _np
        _rng = _np.random.RandomState(args.seed)
        _idx = _rng.choice(
            len(train_dataset), size=args.num_train_samples, replace=False
        ).tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, _idx)
        print(f"  Subsampling training set to {len(train_dataset)} images "
              f"(seed={args.seed})")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if os.path.isdir(val_dir):
        val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transform())
        if getattr(args, "num_val_samples", 0) and args.num_val_samples > 0 \
                and args.num_val_samples < len(val_dataset):
            import numpy as _np
            _rng_v = _np.random.RandomState(args.seed + 1)
            _vidx = _rng_v.choice(
                len(val_dataset), size=args.num_val_samples, replace=False
            ).tolist()
            val_dataset = torch.utils.data.Subset(val_dataset, _vidx)
            print(f"  Subsampling val set to {len(val_dataset)} images "
                  f"(seed={args.seed + 1})")
        # Pick a conservative val batch size. SOAP on ViT-Base with 30
        # training epochs leaves ~8 GB of reserved-but-unallocated CUDA
        # memory plus ~3 GB of persistent optimizer state (L / R / Q_L /
        # Q_R / moments), so even a bs=256 val pass can request a 7 GB
        # contiguous attention tensor and OOM on a 24 GB L4. Default
        # batch_size // 4 for SOAP (i.e. bs=64 at the paper's 256 train
        # batch) is safe in practice; the validate() OOM-retry halves
        # again on the rare case that's still too tight.
        if args.val_batch_size and args.val_batch_size > 0:
            val_bs = args.val_batch_size
        elif args.optimizer == "soap":
            val_bs = max(1, args.batch_size // 4)
        else:
            val_bs = args.batch_size
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_bs,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # --- Mixup / CutMix ---
    mixup_fn = None
    if HAS_TIMM and (args.mixup_alpha > 0 or args.cutmix_alpha > 0):
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            label_smoothing=args.label_smoothing,
            num_classes=args.num_classes,
        )

    # --- Loss function ---
    if mixup_fn is not None:
        # SoftTargetCrossEntropy for mixup soft labels
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # --- Resume from checkpoint ---
    start_epoch = 0
    global_step = 0
    best_acc = 0.0

    if args.resume and os.path.isfile(args.resume):
        start_epoch, global_step, best_acc = load_checkpoint(
            args.resume, model, optimizer, scaler
        )

    # --- Training loop ---
    print(f"\nStarting training from epoch {start_epoch + 1}...")
    total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        avg_loss, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            mixup_fn=mixup_fn,
            criterion=criterion,
            epoch=epoch,
            args=args,
            writer=writer,
            global_step=global_step,
        )

        epoch_time = time.time() - epoch_start
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        print(
            f"Epoch [{epoch+1}/{args.epochs}] completed in {epoch_time:.1f}s  "
            f"Avg Loss: {avg_loss:.4f}"
        )

        # --- Validation ---
        # Explicitly drop the training-time activation cache before
        # validate(). validate() has its own empty_cache, but doing it
        # here too hides the free from the caller's view and lets the
        # allocator release reserved-but-unallocated segments back to the
        # OS before we rebuild the val graph. Important for SOAP, which
        # leaves ~3 GB of persistent optimizer state resident.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if val_loader is not None:
            top1, top5, val_loss = validate(model, val_loader, args)
            writer.add_scalar("val/top1", top1, epoch)
            writer.add_scalar("val/top5", top5, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            print(
                f"  Val -- Top-1: {top1:.2f}%  Top-5: {top5:.2f}%  Loss: {val_loss:.4f}"
            )

            # Append a line to metrics.jsonl so we have the results on disk
            # even if training crashes before the final save_checkpoint.
            try:
                import json as _json
                with open(os.path.join(output_dir, "metrics.jsonl"), "a") as _fh:
                    _fh.write(_json.dumps({
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "val_top1": top1,
                        "val_top5": top5,
                        "val_loss": val_loss,
                        "best_top1_so_far": max(best_acc, top1),
                        "mode": args.mode,
                        "optimizer": args.optimizer,
                    }) + "\n")
            except Exception:
                pass

            if top1 > best_acc:
                best_acc = top1
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_acc": best_acc,
                        "args": vars(args),
                    },
                    os.path.join(ckpt_dir, "best.pth"),
                )

        # --- Periodic checkpoint ---
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_acc": best_acc,
                    "args": vars(args),
                },
                os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pth"),
            )

    # Final metrics for the JSON summary. Reuse the last per-epoch
    # validation result -- running validate() a third time on the full val
    # set doubles wall-clock for no new info on 1-epoch sweeps.
    last_top1 = last_top5 = last_val_loss = None
    if val_loader is not None:
        # The last-epoch validate ran inside the loop above; top1/top5/val_loss
        # from that final iteration are still available in locals.
        try:
            last_top1 = top1
            last_top5 = top5
            last_val_loss = val_loss
        except NameError:
            # No epoch ran at all (resume + target epoch already reached);
            # do one validate to emit useful metrics.
            last_top1, last_top5, last_val_loss = validate(model, val_loader, args)
            best_acc = max(best_acc, last_top1)

    # --- Final summary ---
    total_time = time.time() - total_start
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("\n" + "=" * 72)
    print(f"Training completed in {total_time_str}")
    if last_top1 is not None:
        print(
            f"Final Top-1: {last_top1:.2f}%  "
            f"Final Top-5: {last_top5:.2f}%  "
            f"Final Val Loss: {last_val_loss:.4f}"
        )
    print(f"Best Top-1 Accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print("=" * 72)

    # --- Always persist a machine-readable summary ---
    try:
        import json as _json
        summary = {
            "mode": args.mode,
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "best_top1": best_acc if val_loader is not None else None,
            "final_top1": last_top1,
            "final_top5": last_top5,
            "final_val_loss": last_val_loss,
            "total_time_sec": total_time,
            "ran_validation": val_loader is not None,
        }
        with open(os.path.join(output_dir, "final_metrics.json"), "w") as fh:
            _json.dump(summary, fh, indent=2)
    except Exception as exc:  # pragma: no cover
        print(f"(could not write final_metrics.json: {exc})")

    writer.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Explicitly dump the traceback to stderr AND a file inside the run's
        # output_dir so run.py's log captures it (log.exception can swallow it
        # when the script is executed via runpy.run_path).
        traceback.print_exc()
        try:
            out_dir = None
            for i, arg in enumerate(sys.argv):
                if arg == "--output_dir" and i + 1 < len(sys.argv):
                    out_dir = sys.argv[i + 1]
                elif arg.startswith("--output_dir="):
                    out_dir = arg.split("=", 1)[1]
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "crash.log"), "a") as fh:
                    fh.write(f"\n\n===== {datetime.datetime.now()} =====\n")
                    fh.write("argv: " + " ".join(sys.argv) + "\n\n")
                    traceback.print_exc(file=fh)
        except Exception:
            pass
        raise

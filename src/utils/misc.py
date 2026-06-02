"""Miscellaneous utilities for training and evaluation."""

import os
import math
import numpy as np
import torch
import torch.nn as nn


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: dict, filepath: str):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Drop ``prefix`` from keys that start with it. Non-matching keys are
    dropped (so callers only get the sub-module they asked for)."""
    plen = len(prefix)
    return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer=None,
    source: str = "auto",
    strict: bool = False,
):
    """Load a checkpoint into ``model`` (and optionally ``optimizer``).

    Supports several checkpoint formats:
      * supervised: ``{"model_state_dict": ..., ...}``
      * DINO (train_dino.py): ``{"student": ..., "teacher": ..., "args": ...}``
        where each entry is a ``nn.Sequential(backbone, DINOHead)`` state dict.
      * bare state_dict tensors (no wrapping).

    Parameters
    ----------
    source : {"auto", "teacher", "student", "model"}
        Which sub-tree to load into ``model``. ``"auto"`` picks the first one
        that exists, preferring ``teacher`` (EMA) over ``student`` for DINO
        checkpoints. If the state dict comes from a ``Sequential(backbone, head)``,
        the ``"0."`` prefix is stripped and only the backbone weights are
        loaded.
    strict : bool
        Forwarded to ``model.load_state_dict``. Default False so we tolerate
        missing head / positional-embedding-grid mismatches.
    """
    ckpt = torch.load(filepath, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        # Bare state dict on disk.
        model.load_state_dict(ckpt, strict=strict)
        return ckpt

    # Pick which sub-tree holds the state dict.
    candidates = []
    if source == "auto":
        candidates = ["teacher", "student", "model", "state_dict", "model_state_dict"]
    else:
        candidates = [source]

    sd = None
    picked_key = None
    for key in candidates:
        if key in ckpt and isinstance(ckpt[key], dict):
            sd = ckpt[key]
            picked_key = key
            break
    if sd is None:
        # Treat the whole ckpt as a flat state dict.
        sd = ckpt
        picked_key = "<root>"

    # DINO saves Sequential(backbone, DINOHead); the backbone keys live under
    # "0." and the head under "1.". For segmentation / tokencut we only want
    # the backbone, so strip the leading "0." and drop the head keys.
    if any(k.startswith("0.") for k in sd.keys()) and \
       any(k.startswith("1.") for k in sd.keys()):
        sd = _strip_prefix(sd, "0.")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"load_checkpoint(strict=True) had missing={missing} "
            f"unexpected={unexpected} (source='{picked_key}')"
        )
    print(
        f"[load_checkpoint] {filepath} "
        f"source='{picked_key}'  missing={len(missing)}  unexpected={len(unexpected)}"
    )

    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            print(f"[load_checkpoint] optimizer state not restored: {exc}")
    return ckpt


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    """Cosine schedule with optional linear warmup.

    Returns a numpy array of length ``epochs * niter_per_ep``.
    """
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = epochs * niter_per_ep

    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    cos_iters = total_iters - warmup_iters
    cos_schedule = np.array([
        final_value + 0.5 * (base_value - final_value) *
        (1 + math.cos(math.pi * i / cos_iters))
        for i in range(cos_iters)
    ])

    return np.concatenate([warmup_schedule, cos_schedule])


def get_params_groups(model: nn.Module, weight_decay: float = 0.0):
    """Separate parameters into decay / no-decay groups.

    Biases and LayerNorm parameters do not get weight decay.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Compute mean Intersection over Union.

    Parameters
    ----------
    pred : (N,) int tensor of predicted class indices.
    target : (N,) int tensor of ground-truth class indices.
    num_classes : number of classes.

    Returns
    -------
    mIoU as a float.
    """
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

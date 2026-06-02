#!/usr/bin/env python3
"""
DINO self-supervised pretraining of ViT-Small on ImageNet.

Reproduces Table 2 and Section 6.2 from
"Cutting the Skip: Training Residual-Free Transformers".

Usage:
    python scripts/train_dino.py --mode skipless_init --data_dir /data/imagenet
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.vit import vit_small
from src.models.skipless_init import apply_skipless_init
from src.optimizers.soap import SOAP
from src.data.imagenet import get_imagenet_loaders
from src.dino.dino_loss import DINOLoss
from src.dino.dino_trainer import DINOHead
from src.utils.misc import cosine_scheduler, save_checkpoint, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="DINO ViT-Small pretraining")
    parser.add_argument("--config", type=str, default="")

    # Model
    parser.add_argument("--mode", type=str, default="skip",
                        choices=["skip", "skipless", "skipless_init"])
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--global_crop_size", type=int, default=224)
    parser.add_argument("--local_crop_size", type=int, default=96)

    # Dataset
    parser.add_argument("--data_dir", type=str, default="/data/imagenet")
    parser.add_argument("--num_workers", type=int, default=8)

    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--accum_steps", type=int, default=1)

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "soap"])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay_start", type=float, default=0.04)
    parser.add_argument("--weight_decay_end", type=float, default=0.4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # DINO-specific
    parser.add_argument("--out_dim", type=int, default=65536)
    parser.add_argument("--teacher_temp", type=float, default=0.07)
    parser.add_argument("--warmup_teacher_temp", type=float, default=0.04)
    parser.add_argument("--warmup_teacher_temp_epochs", type=int, default=30)
    parser.add_argument("--student_temp", type=float, default=0.1)
    parser.add_argument("--momentum_teacher_start", type=float, default=0.996)
    parser.add_argument("--momentum_teacher_end", type=float, default=1.0)
    parser.add_argument("--num_local_crops", type=int, default=8)

    # Skipless init hyperparameters
    parser.add_argument("--alpha", type=float, default=1.8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=3.0)

    # Misc
    parser.add_argument("--output_dir", type=str, default="./output/dino_vit_small")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' (default), 'cuda', 'cpu', or 'cuda:N'.")
    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
        args = parser.parse_args()
    if args.no_amp:
        args.amp = False
    return args


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device selection -- fall back cleanly to CPU if no CUDA is available
    # so this script can at least start on a laptop for smoke testing.
    if getattr(args, "device", None) in (None, "", "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] device={args.device!r} but no CUDA; falling back to CPU")
        args.device = "cpu"
    if args.amp and args.device == "cpu":
        print("[warn] AMP disabled because device=cpu")
        args.amp = False

    run_name = f"dino_{args.mode}_{args.optimizer}_ep{args.epochs}"
    output_dir = os.path.join(args.output_dir, run_name)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 72)
    print("DINO ViT-Small Pretraining -- Cutting the Skip")
    print("=" * 72)
    print(f"  Mode: {args.mode}, Optimizer: {args.optimizer}, Epochs: {args.epochs}")

    # --- Build student and teacher ---
    skipless = args.mode in ("skipless", "skipless_init")
    student_backbone = vit_small(
        skipless=skipless, num_classes=0, depth=args.depth,
        img_size=args.img_size, patch_size=args.patch_size,
    )
    teacher_backbone = vit_small(
        skipless=skipless, num_classes=0, depth=args.depth,
        img_size=args.img_size, patch_size=args.patch_size,
    )

    if args.mode == "skipless_init":
        apply_skipless_init(student_backbone, alpha=args.alpha, beta=args.beta, c=args.c)
        apply_skipless_init(teacher_backbone, alpha=args.alpha, beta=args.beta, c=args.c)

    embed_dim = student_backbone.embed_dim

    student = nn.Sequential(
        student_backbone,
        DINOHead(embed_dim, args.out_dim),
    ).to(args.device)

    teacher = nn.Sequential(
        teacher_backbone,
        DINOHead(embed_dim, args.out_dim),
    ).to(args.device)

    # Teacher starts as copy of student, no gradients
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # --- Loss ---
    num_crops = 2 + args.num_local_crops
    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        num_crops=num_crops,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        num_epochs=args.epochs,
        student_temp=args.student_temp,
    ).to(args.device)

    # --- Data ---
    train_loader, _ = get_imagenet_loaders(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, dino=True,
        num_local_crops=args.num_local_crops,
        global_crop_size=args.global_crop_size,
        local_crop_size=args.local_crop_size,
    )
    niter_per_ep = len(train_loader)

    # --- Schedules ---
    lr_schedule = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, niter_per_ep,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay_start, args.weight_decay_end,
        args.epochs, niter_per_ep,
    )
    momentum_schedule = cosine_scheduler(
        args.momentum_teacher_start, args.momentum_teacher_end,
        args.epochs, niter_per_ep,
    )

    # --- Optimizer ---
    # No weight decay on bias/norm
    decay, no_decay = [], []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": args.weight_decay_start},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    else:
        optimizer = SOAP(param_groups, lr=args.lr, betas=(0.9, 0.95))

    scaler = GradScaler(enabled=args.amp)

    # --- Resume ---
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["student"])
        teacher.load_state_dict(ckpt["teacher"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        dino_loss.load_state_dict(ckpt["dino_loss"])
        print(f"  Resumed from epoch {start_epoch}")

    # --- Training ---
    print(f"\nStarting DINO training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        loss_meter = AverageMeter()

        student.train()
        for step, (images, _) in enumerate(train_loader):
            global_step = epoch * niter_per_ep + step

            # Update LR and WD
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[global_step]
                if param_group.get("weight_decay", 0) > 0:
                    param_group["weight_decay"] = wd_schedule[global_step]

            # images is a list of crops from DINODataAugmentation
            images = [im.to(args.device, non_blocking=True) for im in images]

            # Forward student on all crops
            with autocast(enabled=args.amp):
                student_output = torch.cat([student(crop) for crop in images])
                # Forward teacher on global crops only (first 2)
                with torch.no_grad():
                    teacher_output = torch.cat([teacher(crop) for crop in images[:2]])
                loss = dino_loss(student_output, teacher_output, epoch)

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA update teacher
            with torch.no_grad():
                m = momentum_schedule[global_step]
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)

            loss_meter.update(loss.item())

            if step % args.log_interval == 0:
                print(f"  Epoch [{epoch+1}/{args.epochs}] Step [{step}/{niter_per_ep}] "
                      f"Loss: {loss.item():.4f} LR: {lr_schedule[global_step]:.6f}")

        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{args.epochs}] done in {elapsed:.0f}s, "
              f"avg loss: {loss_meter.avg:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            save_checkpoint({
                "epoch": epoch,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "dino_loss": dino_loss.state_dict(),
                "args": vars(args),
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pth"))

    print("\nDINO pretraining completed!")


if __name__ == "__main__":
    main()

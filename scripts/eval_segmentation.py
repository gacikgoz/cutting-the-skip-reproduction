#!/usr/bin/env python3
"""Linear probing evaluation for semantic segmentation.

Reproduces Table 2: train a 1x1 conv head on frozen ViT patch features and
report mIoU on VOC / ADE20K / COCO-Stuff. Designed to consume a DINO
checkpoint produced by ``scripts/train_dino.py`` directly -- it picks up the
backbone from the EMA teacher (or student) automatically.
"""

import argparse
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.vit import vit_small, vit_base
from src.models.skipless_init import apply_skipless_init
from src.eval.linear_probe import LinearProbeSegmentation
from src.utils.misc import load_checkpoint


def get_dataset(name, data_root, image_size, batch_size=16, num_workers=4,
                num_samples=None):
    if name == "voc":
        from src.data.voc import get_voc_loaders
        return get_voc_loaders(
            data_root, batch_size=batch_size, num_workers=num_workers,
            image_size=image_size,
        )
    elif name == "ade20k":
        from src.data.ade20k import get_ade20k_loaders
        return get_ade20k_loaders(
            data_root, batch_size=batch_size, num_workers=num_workers,
            image_size=image_size, num_samples=num_samples,
        )
    elif name == "coco_stuff":
        from src.data.coco_stuff import get_coco_stuff_loaders
        return get_coco_stuff_loaders(
            data_root, batch_size=batch_size, num_workers=num_workers,
            image_size=image_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


# coco_stuff = 182 here matches the post-remap label range (0..181) used by
# COCOStuffDataset. Some of those output channels correspond to unused COCO
# category IDs (gaps at 11, 25, 28, ...) and just stay zero.
NUM_CLASSES = {"voc": 21, "ade20k": 150, "coco_stuff": 182}


def _infer_model_kwargs_from_ckpt(ckpt_path: str):
    """Peek at the checkpoint's saved args to pick img_size/patch_size/mode.

    Falls back to ``None`` when the checkpoint wasn't produced by our training
    scripts (e.g. a bare state dict).
    """
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(blob, dict) or "args" not in blob:
        return {}
    a = blob["args"]
    return {
        "img_size": a.get("img_size"),
        "patch_size": a.get("patch_size"),
        "mode": a.get("mode"),
        "depth": a.get("depth"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="voc",
                        choices=["voc", "ade20k", "coco_stuff"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_small",
                        choices=["vit_small", "vit_base"])
    parser.add_argument("--skipless", action="store_true")
    parser.add_argument("--apply_skipless_init", action="store_true",
                        help="Also re-apply the paper's init before loading "
                             "the checkpoint (harmless; safe to leave off "
                             "because the checkpoint weights overwrite it).")
    parser.add_argument("--alpha", type=float, default=1.8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=3.0)
    # Backbone geometry. Leave as None -> infer from checkpoint.
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--eval_image_size", type=int, default=None,
                        help="Image size used for seg features. Defaults to "
                             "backbone img_size; override with e.g. 224 when "
                             "the DINO backbone was pretrained at 64.")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "teacher", "student", "model"],
                        help="Which sub-tree of the checkpoint to load from.")
    # Default to the last-4-block concat as in the original DINO paper (and
    # the same protocol "Cutting the Skip" uses for its Table 2 numbers).
    # block-11-only is fine for fully residual ViTs but degenerate for
    # residual-free + DINO at our pretraining scale, where the deepest
    # attention heads collapse (TokenCut block 11 = 0.2% on our
    # skipless_init checkpoint vs block 9 = 33%). Concatenating 8/9/10/11
    # gives the linear probe a chance to use the layers that did learn.
    parser.add_argument("--feature_blocks", type=int, nargs="+",
                        default=[8, 9, 10, 11])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Subsample training set size (paper uses 3000 "
                             "for ADE20K/COCO-Stuff).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Infer geometry from the checkpoint's saved args when not given.
    inferred = _infer_model_kwargs_from_ckpt(args.checkpoint)
    if args.img_size is None:
        args.img_size = inferred.get("img_size") or 224
    if args.patch_size is None:
        args.patch_size = inferred.get("patch_size") or 16
    if args.eval_image_size is None:
        # Match the backbone grid by default. When probing at higher res,
        # pos-embedding is bicubically interpolated.
        args.eval_image_size = args.img_size
    if not args.skipless and inferred.get("mode") in ("skipless", "skipless_init"):
        args.skipless = True
    if not args.apply_skipless_init and inferred.get("mode") == "skipless_init":
        args.apply_skipless_init = True

    print("=" * 60)
    print("Linear probe segmentation")
    print(f"  checkpoint     : {args.checkpoint}")
    print(f"  dataset        : {args.dataset} @ {args.eval_image_size}px")
    print(f"  model          : {args.model} (skipless={args.skipless})")
    print(f"  backbone grid  : img={args.img_size}, patch={args.patch_size}")
    print(f"  source         : {args.source}")
    print("=" * 60)

    # Build model
    build_fn = vit_small if args.model == "vit_small" else vit_base
    model = build_fn(
        skipless=args.skipless,
        num_classes=0,
        img_size=args.img_size,
        patch_size=args.patch_size,
    )
    if args.apply_skipless_init:
        apply_skipless_init(model, alpha=args.alpha, beta=args.beta, c=args.c)
    load_checkpoint(args.checkpoint, model, source=args.source)
    model.eval()

    # Get data
    train_loader, val_loader = get_dataset(
        args.dataset, args.data_dir,
        image_size=args.eval_image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
    )

    # Evaluate
    num_classes = NUM_CLASSES[args.dataset]
    probe = LinearProbeSegmentation(
        backbone=model,
        num_classes=num_classes,
        feature_blocks=args.feature_blocks,
        patch_size=args.patch_size,
        device=args.device,
    )

    print(f"\nLinear probe on {args.dataset} (blocks={args.feature_blocks})")
    miou = probe.train(train_loader, val_loader, epochs=args.epochs, lr=args.lr)
    print(f"\nFinal mIoU: {miou:.4f}")

    # Always persist a JSON beside the run so results are never lost.
    out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    result_path = os.path.join(
        out_dir, f"eval_seg_{args.dataset}.json"
    )
    try:
        import json
        with open(result_path, "w") as fh:
            json.dump({
                "dataset": args.dataset,
                "checkpoint": args.checkpoint,
                "miou_pct": float(miou) * 100.0,
                "feature_blocks": list(args.feature_blocks),
                "eval_image_size": args.eval_image_size,
                "img_size": args.img_size,
                "patch_size": args.patch_size,
                "epochs": args.epochs,
                "skipless": args.skipless,
            }, fh, indent=2)
        print(f"Wrote {result_path}")
    except Exception as exc:
        print(f"(could not write {result_path}: {exc})")
    return miou


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

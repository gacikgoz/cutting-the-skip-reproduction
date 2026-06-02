#!/usr/bin/env python3
"""TokenCut object discovery evaluation.

Computes CorLoc@0.5 on PASCAL VOC 2012 trainval using the TokenCut
algorithm (Wang et al., TPAMI 2023) on top of a ViT-Small backbone.
A correct prediction is one whose IoU with the single largest
non-difficult ground-truth bbox exceeds the threshold.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.vit import vit_small, vit_base
from src.eval.tokencut import TokenCutEvaluator
from src.utils.misc import load_checkpoint


def _infer_model_kwargs_from_ckpt(ckpt_path: str) -> dict:
    """Peek at a checkpoint's saved ``args`` to infer backbone geometry.

    Falls back to ``{}`` for bare state dicts (no ``args`` stored).
    """
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(blob, dict) or "args" not in blob:
        return {}
    a = blob["args"]
    if not isinstance(a, dict):
        return {}
    return {
        "img_size": a.get("img_size"),
        "patch_size": a.get("patch_size"),
        "mode": a.get("mode"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="voc", choices=["voc"],
                        help="Dataset with bounding box annotations.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing VOC2012 (or VOCdevkit).")
    parser.add_argument("--model", type=str, default="vit_small",
                        choices=["vit_small", "vit_base"])
    parser.add_argument("--skipless", action="store_true")
    parser.add_argument("--blocks", type=int, nargs="+", default=[9, 10, 11])
    parser.add_argument("--image_size", type=int, default=None,
                        help="Eval input resolution (pos_embed bicubically "
                             "interpolated if != backbone img_size).")
    parser.add_argument("--img_size", type=int, default=None,
                        help="Backbone training image size (for model build). "
                             "Inferred from checkpoint if omitted.")
    parser.add_argument("--patch_size", type=int, default=None,
                        help="Backbone patch size (for model build). "
                             "Inferred from checkpoint if omitted.")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=-1,
                        help="Limit to the first N images (for quick runs).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Infer backbone geometry from the checkpoint if not given.
    inferred = _infer_model_kwargs_from_ckpt(args.checkpoint)
    if args.img_size is None:
        args.img_size = inferred.get("img_size") or 224
    if args.patch_size is None:
        args.patch_size = inferred.get("patch_size") or 16
    if not args.skipless and inferred.get("mode") in ("skipless", "skipless_init"):
        args.skipless = True

    # image_size for inference -- default to backbone img_size so no
    # pos_embed interpolation is needed. Round up to a multiple of patch_size
    # when the user overrides it.
    if args.image_size is None:
        args.image_size = args.img_size
    if args.image_size % args.patch_size != 0:
        adj = args.patch_size * round(args.image_size / args.patch_size)
        print(f"[eval_tokencut] image_size={args.image_size} not divisible "
              f"by patch_size={args.patch_size}; snapping to {adj}.")
        args.image_size = adj

    print("=" * 60)
    print(f"TokenCut evaluation")
    print(f"  checkpoint   : {args.checkpoint}")
    print(f"  model        : {args.model}  skipless={args.skipless}")
    print(f"  backbone grid: img={args.img_size}  patch={args.patch_size}")
    print(f"  eval image   : {args.image_size}")
    print("=" * 60)

    build_fn = vit_small if args.model == "vit_small" else vit_base
    model = build_fn(
        skipless=args.skipless,
        num_classes=0,
        img_size=args.img_size,
        patch_size=args.patch_size,
    )
    load_checkpoint(args.checkpoint, model)

    # Build the bbox dataset once; reuse across all block evaluations.
    from src.data.voc import VOCBBox
    from torch.utils.data import Subset

    dataset = VOCBBox(
        data_root=args.data_dir,
        split="trainval",
        image_size=args.image_size,
    )
    if args.max_images > 0 and args.max_images < len(dataset):
        dataset = Subset(dataset, list(range(args.max_images)))

    print(f"\nTokenCut evaluation on {args.dataset} "
          f"({len(dataset)} images, IoU>={args.iou_threshold})")
    print(f"{'Block':>8} {'CorLoc':>10}")
    print("-" * 20)

    results = {}
    for block_idx in args.blocks:
        evaluator = TokenCutEvaluator(
            model,
            block_idx=block_idx,
            device=args.device,
            image_size=args.image_size,
        )
        corloc = evaluator.evaluate(dataset, iou_threshold=args.iou_threshold)
        pct = 100.0 * corloc
        results[block_idx] = pct
        print(f"  {block_idx:>4}  {pct:>8.2f}%")
    print()

    # Persist results alongside the checkpoint.
    try:
        import json
        out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        path = os.path.join(out_dir, f"eval_tokencut_{args.dataset}.json")
        with open(path, "w") as fh:
            json.dump({
                "dataset": args.dataset,
                "checkpoint": args.checkpoint,
                "iou_threshold": args.iou_threshold,
                "image_size": args.image_size,
                "max_images": args.max_images,
                "corloc_pct_by_block": results,
            }, fh, indent=2)
        print(f"Wrote {path}")
    except Exception as exc:
        print(f"(could not write TokenCut JSON: {exc})")


if __name__ == "__main__":
    main()

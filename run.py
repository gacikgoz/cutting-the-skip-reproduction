#!/usr/bin/env python3
"""
Single entry-point for reproducing
"Cutting the Skip: Training Residual-Free Transformers" (arXiv:2510.00345v1)
at Tiny-ImageNet scale.

`python run.py` runs the whole pipeline end-to-end on one GPU:

    fetch datasets  ->  train (supervised + DINO)  ->  evaluate (segmentation + TokenCut)

Individual stages can be run on their own; every stage forwards unknown
`--flag value` arguments to the underlying training/eval script, so any
config option can be overridden from the command line.

Stages
------
    setup         Install requirements.txt into the current interpreter.
    sanity        Build the three ViT variants, apply the paper's skipless
                  initialization, verify its properties and take one SOAP
                  step. Needs no dataset (use it as a smoke test).
    download      Fetch every auto-downloadable dataset (Tiny-ImageNet, VOC,
                  ADE20K, COCO, COCO-Stuff) into --data_root.
    supervised    Supervised ViT-Base training (Table 1).
    dino          DINO ViT-Small self-supervised pretraining (Tables 2-3).
    segmentation  Linear-probe segmentation eval of a DINO backbone (Table 2).
    tokencut      TokenCut object-discovery eval of a DINO backbone (Table 3).
    all (default) Everything above, in order.

Examples
--------
    python run.py                 # full reproduction (downloads + trains + evaluates)
    python run.py --quick         # 1-epoch smoke test of the whole pipeline
    python run.py setup           # just install dependencies
    python run.py sanity          # verify the method with no dataset
    python run.py download        # just fetch datasets
    python run.py supervised --mode skipless_init --optimizer soap
    python run.py dino --mode skipless_init
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import os
import runpy
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REQUIREMENTS_TXT = PROJECT_ROOT / "requirements.txt"

# Make ``src.*`` importable regardless of the working directory.
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(verbose: bool = True) -> logging.Logger:
    log = logging.getLogger("cutting-the-skip")
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)
    log.propagate = False
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG if verbose else logging.INFO)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s",
                                     datefmt="%H:%M:%S"))
    log.addHandler(h)
    return log


# ---------------------------------------------------------------------------
# Dependency auto-install
# ---------------------------------------------------------------------------

# Distribution name -> importable module name.
REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "numpy": "numpy",
    "scipy": "scipy",
    "pyyaml": "yaml",
    "tqdm": "tqdm",
    "pillow": "PIL",
    "tensorboard": "tensorboard",
}


def _missing_packages() -> List[str]:
    missing = []
    for dist, mod in REQUIRED_PACKAGES.items():
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(dist)
        except (ImportError, ValueError):
            missing.append(dist)
    return missing


def ensure_environment(log: logging.Logger, auto_install: bool = True,
                       force: bool = False) -> bool:
    """Install requirements.txt if any package is missing (or if ``force``)."""
    missing = _missing_packages()
    if not force and not missing:
        log.info("All required packages are available.")
        return True
    if missing:
        log.warning("Missing package(s): %s", ", ".join(missing))
    if not auto_install and not force:
        log.error("Auto-setup disabled. Run: pip install -r %s", REQUIREMENTS_TXT)
        return False
    if not REQUIREMENTS_TXT.is_file():
        log.error("requirements.txt not found at %s", REQUIREMENTS_TXT)
        return False
    log.info("Installing dependencies from %s ...", REQUIREMENTS_TXT)
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                         str(REQUIREMENTS_TXT)], check=False).returncode
    if rc != 0:
        log.error("pip install failed (exit %d). For GPU, install a CUDA-matched "
                  "torch wheel first from https://pytorch.org/get-started/locally/", rc)
        return False
    importlib.invalidate_caches()
    if _missing_packages():
        log.error("Still missing after install: %s", ", ".join(_missing_packages()))
        return False
    log.info("Environment ready.")
    return True


# ---------------------------------------------------------------------------
# Script dispatch
# ---------------------------------------------------------------------------

def run_script(script: str, argv: List[str], log: logging.Logger) -> int:
    """Run a ``scripts/<script>`` in-process with ``argv`` as its sys.argv."""
    path = SCRIPTS_DIR / script
    if not path.is_file():
        log.error("Script not found: %s", path)
        return 2
    log.info("-> %s %s", script, " ".join(argv))
    old_argv = sys.argv
    sys.argv = [str(path)] + list(argv)
    start = time.time()
    try:
        runpy.run_path(str(path), run_name="__main__")
        rc = 0
    except SystemExit as exc:
        rc = int(exc.code) if isinstance(exc.code, int) else (0 if not exc.code else 1)
    except KeyboardInterrupt:
        log.warning("Interrupted")
        rc = 130
    except Exception:
        log.exception("Stage raised an exception")
        rc = 1
    finally:
        sys.argv = old_argv
        log.info("   finished in %.1fs (rc=%d)", time.time() - start, rc)
    return rc


# ---------------------------------------------------------------------------
# Stage: sanity (no dataset)
# ---------------------------------------------------------------------------

def stage_sanity(extra: List[str], log: logging.Logger) -> int:
    import statistics
    import torch
    from src.models.vit import vit_small
    from src.models.skipless_init import apply_skipless_init, verify_init
    from src.optimizers.soap import SOAP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Sanity check on device=%s", device)

    skip = vit_small(skipless=False, num_classes=200, depth=12)
    skipless = vit_small(skipless=True, num_classes=200, depth=12)
    paper = vit_small(skipless=True, num_classes=200, depth=12)
    apply_skipless_init(paper, alpha=1.8, beta=1.0, c=3.0)

    res = verify_init(paper, verbose=False)
    log.info("  W_V*W_O cond     mean=%.3f max=%.3f (target ~1.0)",
             statistics.mean(res["wv_wo_cond_numbers"]), max(res["wv_wo_cond_numbers"]))
    log.info("  W_Q*W_K^T domin. mean=%.3f min=%.3f (higher better)",
             statistics.mean(res["wqk_diag_dominance"]), min(res["wqk_diag_dominance"]))
    log.info("  MLP orth. dev    mean=%.6f max=%.6f (target ~0)",
             statistics.mean(res["mlp_orthogonality"]), max(res["mlp_orthogonality"]))

    x = torch.randn(2, 3, 64, 64, device=device)
    y = torch.randint(0, 200, (2,), device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    for name, model in [("skip", skip), ("skipless", skipless), ("skipless_init", paper)]:
        model = model.to(device)
        opt = SOAP(model.parameters(), lr=1e-4)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        log.info("  [%s] logits=%s loss=%.4f", name, tuple(logits.shape), loss.item())
    log.info("Sanity check passed.")
    return 0


# ---------------------------------------------------------------------------
# Stage: download
# ---------------------------------------------------------------------------

def stage_download(data_root: Path, log: logging.Logger,
                   which: Optional[List[str]] = None) -> dict:
    from src.data.download import download_all
    logging.getLogger("cutting-the-skip.download").addHandler(log.handlers[0])
    logging.getLogger("cutting-the-skip.download").setLevel(logging.INFO)
    log.info("Fetching datasets into %s ...", data_root)
    results = download_all(data_root, which=which)
    for name, ok in results.items():
        log.info("  dataset %-14s %s", name, "OK" if ok else "FAILED")
    return results


# ---------------------------------------------------------------------------
# Thin stage wrappers (forward all flags to the underlying script)
# ---------------------------------------------------------------------------

def stage_supervised(extra: List[str], log: logging.Logger) -> int:
    if not any(a.startswith("--config") for a in extra):
        extra = ["--config", str(CONFIGS_DIR / "supervised_vit_base.yaml")] + extra
    return run_script("train_supervised.py", extra, log)


def stage_dino(extra: List[str], log: logging.Logger) -> int:
    if not any(a.startswith("--config") for a in extra):
        extra = ["--config", str(CONFIGS_DIR / "dino_vit_small.yaml")] + extra
    return run_script("train_dino.py", extra, log)


def stage_segmentation(extra: List[str], log: logging.Logger) -> int:
    return run_script("eval_segmentation.py", extra, log)


def stage_tokencut(extra: List[str], log: logging.Logger) -> int:
    return run_script("eval_tokencut.py", extra, log)


# ---------------------------------------------------------------------------
# Stage: all  (download -> sanity -> supervised -> dino -> segmentation -> tokencut)
# ---------------------------------------------------------------------------

# Table 1: 3 modes x 2 optimizers. paper_top1 are full-ImageNet-1k references.
TABLE1_RUNS = [
    ("skip",          "adamw"), ("skip",          "soap"),
    ("skipless",      "adamw"), ("skipless",      "soap"),
    ("skipless_init", "adamw"), ("skipless_init", "soap"),
]

# Tables 2-3 DINO backbones (mode -> whether the architecture is residual-free).
DINO_BACKBONES = [("skip", False), ("skipless_init", True)]

# Table 2 linear-probe segmentation datasets (name -> dir under data_root).
SEG_DATASETS = [
    ("voc",        "VOCdevkit/VOC2012"),
    ("ade20k",     "ade20k/ADEChallengeData2016"),
    ("coco_stuff", "coco_stuff"),
]


def _tiny_imagenet_dir(data_root: Path) -> Optional[Path]:
    for name in ("imagenet", "tiny-imagenet-200"):
        d = data_root / name
        if (d / "train").is_dir() and (d / "val").is_dir():
            return d
    return None


def stage_all(extra: List[str], log: logging.Logger) -> int:
    p = argparse.ArgumentParser(prog="run.py all", add_help=True)
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--output_root", default=str(PROJECT_ROOT / "output"))
    p.add_argument("--only", nargs="*", default=[],
                   choices=["download", "sanity", "supervised", "dino",
                            "segmentation", "tokencut"])
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["download", "sanity", "supervised", "dino",
                            "segmentation", "tokencut"])
    p.add_argument("--quick", action="store_true",
                   help="1-epoch smoke test of every training/eval loop.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    args, passthrough = p.parse_known_args(extra)

    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    def want(name: str) -> bool:
        return name in args.only if args.only else name not in args.skip

    results: List[tuple] = []
    def record(name: str, status: str, detail: str = "") -> None:
        results.append((name, status, detail))
        log.info("[all] %-28s %-10s %s", name, status, detail)

    log.info("=" * 70)
    log.info("Full reproduction | data_root=%s output_root=%s quick=%s",
             data_root, out_root, args.quick)
    log.info("=" * 70)

    # 0. Datasets ---------------------------------------------------------
    if want("download"):
        dl = stage_download(data_root, log)
        for name, ok in dl.items():
            record(f"download[{name}]", "OK" if ok else "FAIL")
    else:
        record("download", "SKIPPED")

    # 1. Sanity -----------------------------------------------------------
    if want("sanity"):
        record("sanity", "OK" if stage_sanity([], log) == 0 else "FAIL")
    else:
        record("sanity", "SKIPPED")

    imagenet_dir = _tiny_imagenet_dir(data_root)

    # 2. Table 1 -- supervised -------------------------------------------
    if want("supervised"):
        if imagenet_dir is None:
            record("supervised", "SKIPPED", "Tiny-ImageNet missing")
        else:
            for mode, opt in TABLE1_RUNS:
                cli = ["--config", str(CONFIGS_DIR / "supervised_vit_base.yaml"),
                       "--mode", mode, "--optimizer", opt,
                       "--data_dir", str(imagenet_dir),
                       "--output_dir", str(out_root / "table1" / f"{mode}_{opt}"),
                       "--num_workers", str(args.num_workers), "--seed", str(args.seed)]
                if args.quick:
                    cli += ["--epochs", "1", "--warmup_epochs", "0", "--save_interval", "1"]
                rc = run_script("train_supervised.py", cli + passthrough, log)
                record(f"table1[{mode}/{opt}]", "OK" if rc == 0 else "FAIL")
    else:
        record("supervised", "SKIPPED")

    # 3. DINO pretraining (prerequisite for Tables 2-3) ------------------
    dino_ckpt = {}
    if want("dino"):
        if imagenet_dir is None:
            record("dino", "SKIPPED", "Tiny-ImageNet missing")
        else:
            for mode, _ in DINO_BACKBONES:
                run_out = out_root / "dino" / mode
                run_out.mkdir(parents=True, exist_ok=True)
                cli = ["--config", str(CONFIGS_DIR / "dino_vit_small.yaml"),
                       "--mode", mode, "--data_dir", str(imagenet_dir),
                       "--output_dir", str(run_out),
                       "--num_workers", str(args.num_workers), "--seed", str(args.seed)]
                if args.quick:
                    cli += ["--epochs", "1", "--warmup_epochs", "0",
                            "--warmup_teacher_temp_epochs", "1", "--save_interval", "1"]
                rc = run_script("train_dino.py", cli + passthrough, log)
                record(f"dino[{mode}]", "OK" if rc == 0 else "FAIL")
                ckpts = sorted(run_out.rglob("epoch_*.pth")) + sorted(run_out.rglob("checkpoint*.pth"))
                if ckpts:
                    dino_ckpt[mode] = ckpts[-1]
    else:
        record("dino", "SKIPPED")

    # Locate any pre-existing DINO checkpoints if we skipped training.
    for mode, _ in DINO_BACKBONES:
        if mode not in dino_ckpt:
            ckpts = sorted((out_root / "dino" / mode).rglob("*.pth"))
            if ckpts:
                dino_ckpt[mode] = ckpts[-1]

    # 4. Table 2 -- linear-probe segmentation ----------------------------
    if want("segmentation"):
        for mode, skipless in DINO_BACKBONES:
            ckpt = dino_ckpt.get(mode)
            if ckpt is None:
                record(f"table2[{mode}]", "SKIPPED", "no DINO ckpt")
                continue
            for ds_name, subpath in SEG_DATASETS:
                ds_dir = data_root / subpath
                if not ds_dir.is_dir():
                    record(f"table2[{mode}/{ds_name}]", "SKIPPED", "dataset missing")
                    continue
                cli = ["--checkpoint", str(ckpt), "--dataset", ds_name,
                       "--data_dir", str(ds_dir), "--model", "vit_small",
                       "--epochs", "1" if args.quick else "20"]
                if skipless:
                    cli += ["--skipless"]
                rc = run_script("eval_segmentation.py", cli + passthrough, log)
                record(f"table2[{mode}/{ds_name}]", "OK" if rc == 0 else "FAIL")
    else:
        record("segmentation", "SKIPPED")

    # 5. Table 3 -- TokenCut object discovery (VOC) ----------------------
    if want("tokencut"):
        voc_dir = data_root / "VOCdevkit" / "VOC2012"
        for mode, skipless in DINO_BACKBONES:
            ckpt = dino_ckpt.get(mode)
            if ckpt is None:
                record(f"table3[{mode}]", "SKIPPED", "no DINO ckpt")
                continue
            if not voc_dir.is_dir():
                record(f"table3[{mode}]", "SKIPPED", "VOC missing")
                continue
            cli = ["--checkpoint", str(ckpt), "--dataset", "voc",
                   "--data_dir", str(voc_dir), "--model", "vit_small",
                   "--blocks", "9", "10", "11"]
            if skipless:
                cli += ["--skipless"]
            rc = run_script("eval_tokencut.py", cli + passthrough, log)
            record(f"table3[{mode}]", "OK" if rc == 0 else "FAIL")
    else:
        record("tokencut", "SKIPPED")

    # Summary -------------------------------------------------------------
    log.info("=" * 70)
    log.info("Summary")
    log.info("=" * 70)
    ok = sum(1 for _, s, _ in results if s == "OK")
    fail = sum(1 for _, s, _ in results if s == "FAIL")
    skipped = len(results) - ok - fail
    for name, status, detail in results:
        log.info("  %-28s %-10s %s", name, status, detail)
    log.info("Totals: %d OK, %d FAIL, %d SKIPPED | output: %s", ok, fail, skipped, out_root)
    return 0 if fail == 0 else 1


STAGES = {
    "sanity": stage_sanity,
    "supervised": stage_supervised,
    "dino": stage_dino,
    "segmentation": stage_segmentation,
    "tokencut": stage_tokencut,
    "all": stage_all,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # First token is a stage name only if it matches a known stage (or the
    # special 'setup'/'download' stages); otherwise default to 'all'.
    known = set(STAGES) | {"setup", "download"}
    if not argv or argv[0].startswith("-") or argv[0] not in known:
        argv = ["all"] + argv

    top = argparse.ArgumentParser(prog="run.py", add_help=True,
                                  description="Reproduce 'Cutting the Skip' end-to-end.")
    top.add_argument("stage", nargs="?", default="all", choices=sorted(known))
    top.add_argument("--quiet", action="store_true")
    top.add_argument("--no_auto_setup", action="store_true",
                     help="Do not auto-install missing packages.")
    # NB: --data_root is intentionally NOT defined here so it falls through to
    # the chosen stage (stage_all / download both parse their own copy).
    top_args, stage_args = top.parse_known_args(argv)

    log = setup_logger(verbose=not top_args.quiet)
    log.info("Stage: %s", top_args.stage)

    # 'setup' just installs dependencies and exits.
    if top_args.stage == "setup":
        return 0 if ensure_environment(log, auto_install=True, force=True) else 1

    # Every other stage needs the third-party packages.
    if not ensure_environment(log, auto_install=not top_args.no_auto_setup):
        log.error("Environment not ready. Run: python run.py setup")
        return 1

    if top_args.stage == "download":
        dp = argparse.ArgumentParser(prog="run.py download")
        dp.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
        dl_args, _ = dp.parse_known_args(stage_args)
        data_root = Path(dl_args.data_root).expanduser().resolve()
        results = stage_download(data_root, log)
        return 0 if all(results.values()) else 1

    rc = STAGES[top_args.stage](stage_args, log)
    log.info("Stage '%s' %s (rc=%d)", top_args.stage,
             "succeeded" if rc == 0 else "failed", rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

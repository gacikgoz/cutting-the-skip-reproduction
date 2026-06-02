# Cutting the Skip — Training Residual-Free Transformers (CENG502 reproduction)

Minimal, self-contained reproduction of
**"Cutting the Skip: Training Residual-Free Transformers"** (arXiv:2510.00345v1)
at **Tiny-ImageNet** scale on **a single NVIDIA A100 80 GB GPU**.

The paper proposes (i) a closed-form initialization for residual-free ViTs
(`W_V W_O` scaled-orthonormal, `W_Q W_K^T` near-identity, orthogonal MLP
weights) and (ii) the **SOAP** optimizer. Its central claim: a residual-free
ViT with the proposed initialization (`skipless_init`) trains and stays
competitive with a standard residual ViT (`skip`), while a residual-free ViT
*without* the init (`skipless`) does not train.

One command fetches the data, trains, and evaluates:

```bash
python run.py
```

## What reproduces

Judged on the paper's central **within-column ordering** criterion (relative
ordering of `skip` / `skipless` / `skipless_init`, not absolute magnitudes —
we train on a ~39× smaller token-view budget than the paper's ImageNet-1k):

| Paper claim                  | Result |
|------------------------------|--------|
| Initialization correctness   | ✓ (`W_V W_O` cond ≈ 1.000, `W_Q W_K^T` dominance 7.1, MLP orth. dev ~1e-6) |
| Table 1 — SOAP supervised    | ✓ `skipless_init` − `skipless` = +14.16 pp (paper +8.4) |
| Table 1 — AdamW supervised   | ✓ init-vs-no-init gap +12.84 pp (paper +12.8) |
| Tables 2–3 — DINO transfer   | ✓ `skip` ≥ `skipless_init` on every segmentation row and TokenCut best block |

The full 3-page write-up (with the per-block diagnostics and the dataset-substitution
argument) is bundled here: [`final_report.pdf`](final_report.pdf).

## Repository layout

```
.
├── run.py                 # single entry-point: fetch data → train → evaluate
├── requirements.txt       # dependencies (or run: python run.py setup)
├── README.md
├── final_report.pdf       # the 3-page reproduction write-up
├── configs/
│   ├── supervised_vit_base.yaml   # Table 1 (ViT-Base, Tiny-ImageNet)
│   ├── dino_vit_small.yaml        # Tables 2–3 (DINO ViT-Small pretraining)
│   └── eval_segmentation.yaml     # linear-probe segmentation defaults
├── scripts/               # training / evaluation entry scripts
│   ├── train_supervised.py        # supervised ViT (Table 1)
│   ├── train_dino.py              # DINO self-supervised pretraining
│   ├── eval_segmentation.py       # linear-probe segmentation (Table 2)
│   └── eval_tokencut.py           # TokenCut object discovery (Table 3)
└── src/                   # library code
    ├── models/            # ViT + paper's skipless initialization
    ├── optimizers/        # SOAP
    ├── dino/              # DINO loss + head
    ├── data/              # dataset loaders + auto-download
    ├── eval/              # linear probe + TokenCut
    └── utils/             # schedulers, checkpoint IO, logging
```

`data/` and `output/` are created on first run and are git-ignored.

## Setup

```bash
# Install a CUDA-matched PyTorch wheel first (recommended for GPU):
#   see https://pytorch.org/get-started/locally/
pip install -r requirements.txt        # or: python run.py setup
```

`run.py` also auto-installs anything missing on first use (disable with
`--no_auto_setup`).

## Usage

```bash
python run.py                 # full pipeline: download + train + evaluate
python run.py --quick         # 1-epoch smoke test of the whole pipeline
python run.py setup           # install dependencies only
python run.py sanity          # verify the init + SOAP step (no dataset needed)
python run.py download        # fetch datasets only
```

Run a single stage (any config key can be overridden on the command line):

```bash
python run.py supervised --mode skipless_init --optimizer soap
python run.py dino --mode skipless_init
python run.py segmentation --checkpoint output/dino/skipless_init/<ckpt>.pth \
    --dataset voc --data_dir data/VOCdevkit/VOC2012 --skipless
python run.py tokencut --checkpoint output/dino/skipless_init/<ckpt>.pth \
    --dataset voc --data_dir data/VOCdevkit/VOC2012 --skipless
```

Scope the full run with `--only` / `--skip`:

```bash
python run.py --only supervised            # just Table 1
python run.py --skip dino segmentation tokencut
```

### Stages

| Stage          | What it does                                              | Dataset |
|----------------|-----------------------------------------------------------|---------|
| `setup`        | `pip install -r requirements.txt`                         | —       |
| `sanity`       | Build ViTs, apply init, verify properties, one SOAP step  | none    |
| `download`     | Fetch Tiny-ImageNet, VOC, ADE20K, COCO, COCO-Stuff        | —       |
| `supervised`   | Supervised ViT-Base training (Table 1)                    | Tiny-ImageNet |
| `dino`         | DINO ViT-Small pretraining (Tables 2–3 prerequisite)      | Tiny-ImageNet |
| `segmentation` | Linear-probe segmentation of a DINO backbone (Table 2)    | VOC / ADE20K / COCO-Stuff |
| `tokencut`     | TokenCut object discovery (Table 3)                       | VOC     |

## Datasets

`python run.py download` (and the default `all` run) auto-fetch every
redistributable dataset into `./data`:

- **Tiny-ImageNet-200** (200 classes, 100k images @ 64×64) — the ImageNet-1k
  proxy used throughout, also aliased as `data/imagenet`.
- **PASCAL VOC 2012**, **ADE20K**, **COCO 2017 + COCO-Stuff** — transfer-eval
  targets for Tables 2–3.

Full **ImageNet-1k** requires manual registration at https://image-net.org/.
To use it instead, point `--data_dir` at it and override
`--num_classes 1000 --img_size 224 --patch_size 16`.

## Reproducing the report's exact numbers

The default configs train at a sensible Tiny-ImageNet budget (supervised 60 ep,
DINO 100 ep) and reproduce the **SOAP** ordering directly. To match the report's
headline figures:

```bash
# Table 1 — AdamW (milder recipe; the paper recipe stalls at Tiny-ImageNet scale)
python run.py supervised --mode skipless_init --optimizer adamw \
    --epochs 150 --lr 1e-4 --weight_decay 0.05 --warmup_epochs 15 \
    --mixup_alpha 0 --cutmix_alpha 0 --label_smoothing 0

# DINO — the working lr=1.5e-4 / warmup=20 recipe is already the config default;
# just lengthen to the paper's 300-epoch schedule (Tables 2-3 magnitudes)
python run.py dino --mode skipless_init --epochs 300
```

## Notes

- Trained and evaluated on **a single NVIDIA A100 80 GB GPU**. AMP is on by
  default; reduce `--batch_size` for smaller cards.
- Absolute mIoU / CorLoc are below the paper's ImageNet-1k figures by
  construction (smaller corpus); the reproduction is judged on within-column
  ordering. See the report for the full argument and per-block diagnostics.

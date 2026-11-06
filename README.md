# CourtVisionNet

Badminton court detection from non-standard camera angles using a hybrid segmentation-keypoint deep learning architecture.

## Overview

CourtVisionNet detects badminton courts in video frames captured from arbitrary camera positions (ground-level, phone recordings, non-broadcast angles). It predicts 30 court keypoints (all line intersections on a standard doubles court) and a segmentation mask, then estimates a homography to map pixel coordinates to real-world court positions — enabling player position tracking and tactical analysis.

### Architecture

- **Backbone**: ResNet-50 + FPN with 7-channel input (RGB + grayscale + CLAHE + Canny edges + green court-colour mask). The model normalizes its own [0,1] input (ImageNet stats for RGB).
- **Segmentation head**: multi-scale feature fusion → court-line mask (BCE + per-image Dice).
- **Keypoint head**: 30 heatmaps (stride 4, or stride 2 with `heatmap_size = image_size // 2`), pixel-centre soft-argmax for training, argmax + local/DARK refinement at inference, and a per-keypoint visibility head.
- **Loss**: Jensen–Shannon heatmap loss (DSNT-style) + coordinate loss + differentiable-homography reprojection loss, optionally symmetric over the court's 180° rotation.
- **Inference**: letterboxed input, RANSAC/MAGSAC homography from confident keypoints with sanity checks, and reprojection of the full template.

### 30 Keypoints

All line intersections on a standard doubles badminton court, organized as a 6×5 grid: row r = 0..5 along the court length, column c = 0..4 across the width, index `K = 5r + c`.

| Row | Keypoints | Line |
|-----|-----------|------|
| 0 | K0–K4 | Baseline |
| 1 | K5–K9 | Doubles long service line |
| 2 | K10–K14 | Short service line |
| 3 | K15–K19 | Short service line (other half) |
| 4 | K20–K24 | Doubles long service line (other half) |
| 5 | K25–K29 | Baseline (other end) |

Columns: doubles sideline, singles sideline, centre line, singles sideline, doubles sideline.

**Labeling convention.** A court looks identical when mirrored or rotated 180°, so labels must follow a view-dependent rule; `src/court_geometry.py:canonicalize` enforces it everywhere (annotators, converters, dataset build, augmentation):

1. the template→image homography has positive orientation (never mirrored), and
2. the image direction from row 0 to row 5 points up and/or right.

For an end-on view, K0 is the **near-left** baseline corner; for a side view, row 0 is the **left** baseline.

## Project Structure

```
src/
├── court_geometry.py          # Court dimensions, keypoint template, homography utils
├── models/
│   ├── backbone.py            # ResNet-50 + FPN (7-channel input)
│   ├── segmentation_head.py   # Binary segmentation decoder
│   ├── keypoint_head.py       # Heatmap + offset + visibility heads
│   ├── courtvisionnet.py      # Full model assembly
│   └── losses.py              # Combined loss (seg + heatmap + offset + visibility + geometric)
├── preprocessing/
│   ├── channels.py            # 7-channel preprocessing pipeline
│   └── augmentation.py        # Albumentations augmentation with flip-pair handling
├── training/
│   ├── dataset.py             # PyTorch dataset (annotations → heatmaps + masks)
│   ├── train.py               # Training loop with early stopping, backbone freeze
│   └── config.py              # Hyperparameter configuration
├── evaluation/
│   ├── metrics.py             # PCK, MRE, Court IoU, Segmentation IoU
│   └── evaluate.py            # Standalone evaluation CLI
├── inference/
│   ├── predict.py             # Single-image inference + homography estimation
│   └── visualize.py           # Court overlay visualization
└── tools/
    ├── extract_frames.py      # Extract frames from video files
    ├── annotator.py           # Desktop annotation tool (Tkinter)
    ├── annotator.html         # Browser-based annotation tool
    ├── blender_court.py       # Court geometry builder (Blender)
    ├── blender_camera.py      # Camera placement strategies (Blender)
    ├── blender_lighting.py    # Lighting presets (Blender)
    ├── blender_occluders.py   # On-court occluders (Blender)
    ├── blender_environment.py # Venue environment (Blender)
    ├── blender_render.py      # Batch renderer orchestrator (Blender)
    └── blender_to_cvn.py      # Blender metadata → CVN format converter
tests/                         # Unit tests covering all modules
```

## Setup

```bash
pip install -r requirements.txt
```

## Data preparation

All annotation sources (hand labels in `data/annotations`, the Roboflow export in `data/cvn_dataset`, Blender renders in `data/blender/raw`) are merged into one leak-free dataset:

```bash
python scripts/prepare_colab.py --out data/colab_data_v2
```

This canonicalizes every label, drops annotations whose homography residual exceeds `--max-residual` px, splits **by group** (whole videos / capture sessions, merging visually near-identical groups) into `train`, `val`, `test_real` and `test_roboflow`, checks that no near-duplicate images cross splits, and writes `manifest.json` (file list with SHA-256 hashes) plus `summary.txt`. Rebuild an existing split exactly with `--manifest data/colab_data_v2/manifest.json`. Blender data is train-only.

Quality checks: `python scripts/qa_annotations.py data/annotations --images data/images`.

## Annotation

Start the local annotation server (binds to 127.0.0.1 only) and use the browser annotator:

```bash
python src/tools/serve_annotator.py
```

1. Click a keypoint on the court diagram (right panel), then click where it appears on the image.
2. With at least 4 points that support a homography, **Suggest (G)** fills the rest; accept with `Y`.
3. **Canonicalize (A)** relabels into the canonical convention (also done automatically on save).
4. **Save (S)** writes `data/annotations/<image>.json`; each point records whether it was clicked or suggested.

Scroll to zoom, drag to pan, right-click to remove a point, `N`/`P` to navigate, `Z`/`Y` undo/redo.

Desktop alternative (Tkinter): `python -m src.tools.annotator --input-dir data/frames/ --output-dir data/annotations/`.

## Synthetic Data (Blender)

Requires [Blender 5.2+](https://www.blender.org/download/).

```bash
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --factory-startup --python src/tools/blender_render.py -- --count 500 --engine BLENDER_EEVEE --samples 32 --seed 42
```

Or `python scripts/generate.py render --count 500` (wraps the same command). Each image is seeded from `(seed, index)`, so batches with different `--start` are distinct and reproducible. Camera strategies cover broadcast, end-low, sideline, corner, overhead, spectator and adjacent-court views with roll, 13–85 mm lenses, partial framing and 16:9 / 9:16 / 4:3 / 1:1 aspect ratios (`--aspects`, `--strategies`, `--res-min/--res-max` short side). Metadata records intrinsics, extrinsics, all 30 projected keypoints, and ray-cast occlusion per keypoint.

```bash
python src/tools/blender_to_cvn.py                         # raw metadata -> canonical CVN annotations
python scripts/generate.py augment --out data/blender_aug  # lens distortion, blur, noise, JPEG
```

Pass augmented sets to the dataset build with `--extra-synthetic DIR`.

## Training

```bash
python -m src.training.train --config configs/baseline.json --set seed=0
python -m src.training.train --config configs/baseline.json --resume-from checkpoints/baseline_s0/last.pt
```

`--set key=value` overrides any `TrainConfig` field (`src/training/config.py`). Checkpoints (`best_model.pt`, `last.pt`) carry the config, git SHA, history and RNG state; resuming reproduces an uninterrupted run. Model selection and early stopping use pooled PCK@10 on the `val` split (real images only). Ablation configs and the seed protocol are in [configs/README.md](configs/README.md); Kaggle/Colab notebooks are in `notebooks/`.

## Inference

```python
import cv2
from src.inference.predict import CourtPredictor

predictor = CourtPredictor("checkpoints/baseline_s0/best_model.pt")
det = predictor.predict(cv2.imread("frame.jpg"))   # BGR image
det.keypoints       # (30, 2) normalized, homography-refined
det.raw_keypoints   # (30, 2) normalized, straight from the decoder
det.visibility      # (30,) probabilities
det.homography      # 3x3 court metres -> image pixels, or None
```

## Evaluation

```bash
python -m src.evaluation.evaluate --checkpoint checkpoints/baseline_s0/best_model.pt     --manifest data/colab_data_v2/manifest.json --split test_real --out results/test_real.json
python scripts/nn_baseline.py --manifest data/colab_data_v2/manifest.json     --train-split train --eval-split test_real --out results/nn_test_real.json
```

Errors are measured in original-image pixels. Reported: raw keypoint error (pooled and per image), PCK at fixed pixels and normalized by court diagonal, KLA@0.05/0.1 of the image diagonal, PCK-curve AUC, error in court metres, RANSAC homography reprojection error, court IoU, PCR, visibility precision/recall/F1, seg IoU, bootstrap 95% CIs, and per-source/group/keypoint breakdowns. Always report `test_real` and `test_roboflow` separately, alongside the nearest-neighbour baseline. Definitions are at the top of `src/evaluation/metrics.py`.

## Tests

```bash
python -m pytest tests/ -q              # Blender smoke test is skipped outside Blender
python -m pytest tests/ -q -m "not slow"
```

## License

MIT

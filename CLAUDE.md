# CourtVisionNet

Badminton court keypoint detection from non-standard camera angles. Research project targeting a publishable paper.

## Project structure

- `src/models/` - CourtVisionNet architecture (ResNet-50 + FPN backbone, segmentation head, keypoint head), losses, differentiable DLT (`homography.py`)
- `src/preprocessing/` - 7-channel input pipeline (RGB + grayscale + CLAHE + Canny + green court-colour mask), `letterbox.py`, augmentation
- `src/training/` - Dataset, training loop (CLI), `TrainConfig`
- `src/evaluation/` - Metrics (definitions at top of `metrics.py`) and the evaluation CLI
- `src/inference/` - `CourtPredictor`, heatmap decoding (`decode.py`), visualization
- `src/data_manifest.py` - Grouping, splitting, canonicalization and QA helpers used by `scripts/prepare_colab.py`
- `src/court_geometry.py` - Court dimensions, 30 keypoints in 6x5 grid, symmetry permutations, `canonicalize`, homography utils
- `src/tools/blender_*.py` - Blender synthetic data pipeline (runs inside Blender's Python, not project Python); shared constants in `blender_common.py`
- `src/tools/camera_framing.py` - Per-image auto-FOV for Blender cameras (pure numpy, unit-tested): frames the court or a region of it, then picks a realistic phone / broadcast focal length
- `src/tools/blender_to_cvn.py` - Blender metadata -> canonical CVN annotations (project Python)
- `src/tools/synthetic_augment.py` - Lens distortion / blur / noise / JPEG augmentation of CVN datasets
- `src/tools/annotator.html` + `serve_annotator.py` - Browser annotator (127.0.0.1:8000); `annotator.py` - Tkinter annotator
- `src/tools/serve_review.py` - Inference review UI (127.0.0.1:8001)
- `src/tools/scrape_courts.py`, `extract_frames.py`, `convert_dataset.py`, `coco_to_cvn.py`, `run_inference.py` (Roboflow API)
- `scripts/` - `prepare_colab.py` (leak-free dataset build), `qa_annotations.py`, `canonicalize_annotations.py`, `nn_baseline.py`, `summarize_runs.py`, `generate.py`, `prepare_roboflow.py`; `runpod/run_experiments.sh` (GPU training driver, see `docs/runpod.md`)
- `docs/improvement-plan.md` - Prioritized research plan (evaluation, baselines, geometry, model, data)
- `configs/` - Training configs: `baseline.json` plus ablations (see `configs/README.md`)
- `notebooks/` - Kaggle/Colab training notebooks
- `tests/` - Unit tests (`pytest`; Blender smoke test is skipped outside Blender)

## Key conventions

- 30 keypoints K = 5*row + col: 6 rows along the court length x 5 columns across the width. BWF court 13.4m x 6.1m.
- **Canonical labeling** (enforced by `court_geometry.canonicalize` everywhere labels are created or transformed): positive template->image orientation (never mirrored) and the row-0 -> row-5 direction points up/right in the image. End-on view: K0 = near-left baseline corner. Never use a fixed index swap after a geometric transform; re-canonicalize instead.
- CVN annotation format: `{"image_path" (basename), "image_size": [h, w], "keypoints" (normalized 0-1, [-1,-1] if unlabeled), "visibility", "bounding_box"}` plus optional `source`, `group`, `occluded`, `keypoint_origin`, `canonical_perm`
- `image_size` uses `[height, width]` convention
- Images are letterboxed (aspect-preserving) to the model size; augmentation runs on BGR before the 7 channels are computed; the model normalizes its own [0,1] input
- Pixel-centre convention: heatmap index i <-> normalized (i + 0.5) / W
- Splits are by group (video / capture session), never by image. Report `test_real` and `test_roboflow` separately, next to the nearest-neighbour baseline.

## Data layout

- `data/images/` + `data/annotations/` - Hand-labeled real images (116 annotations; mostly one hall, several sessions)
- `data/blender/raw/` - Blender renders + metadata
- `data/cvn_dataset/` - Roboflow COCO export converted to CVN (its own random split is ignored)
- `data/colab_data_v2/` - Leak-free merged dataset: `train`, `val`, `test_real`, `test_roboflow` + `manifest.json` (hashes) + `summary.txt`
- `data/colab_data/` - Old merged dataset with frame-level leakage; do not use for reported results

## Blender pipeline

Requires Blender 5.2+ and must be launched with `--factory-startup` (otherwise the default cube ends up in the scene).

- Engine name is `BLENDER_EEVEE` (not `EEVEE`) in Blender 5.2
- Scene is built procedurally; geometry helpers in `blender_common.py` (no z-scaled planes)
- Per-image seed = f(global seed, index); metadata records intrinsics, extrinsics, all 30 projected keypoints and ray-cast `keypoint_occluded`
- Camera FOV is chosen per image (`camera_framing.py`); focal lengths are 35 mm-equivalent (43.27 mm sensor diagonal). Defaults are tuned so court coverage matches the real images; re-check with real data statistics if you change them
- Raw Blender keypoints are world-fixed and mirrored relative to the canonical convention; `blender_to_cvn.py` canonicalizes them

## Training

- `TrainConfig` in `src/training/config.py`; run configs in `configs/`. Override any field with `--set key=value`.
- Loss: BCE + per-image Dice (seg) + Jensen-Shannon heatmap + coordinate + BCE visibility + homography reprojection; optional symmetric (rot180) min
- Model selection / early stopping on pooled PCK@10 on the real-image `val` split
- AdamW with backbone / heads LR groups, warmup + cosine; backbone (ResNet only, BN frozen) frozen for the first epochs
- Checkpoints are format v2 (config, git SHA, history, RNG state); resume reproduces an uninterrupted run
- Legacy Run-7 checkpoint (`checkpoints/best_model.pt`) loads in `CourtPredictor` non-strictly (visibility head re-initialized) and was trained on leaked splits

## Commands

```bash
# Run tests
python -m pytest tests/ -q

# Build the leak-free dataset
python scripts/prepare_colab.py --out data/colab_data_v2

# Train / resume
python -m src.training.train --config configs/baseline.json --set seed=0
python -m src.training.train --config configs/baseline.json --resume-from checkpoints/baseline_s0/last.pt

# Evaluate (run for test_real and test_roboflow)
python -m src.evaluation.evaluate --checkpoint checkpoints/baseline_s0/best_model.pt --manifest data/colab_data_v2/manifest.json --split test_real --out results/test_real.json

# GPU runs on RunPod (inside the pod, from the repo root; see docs/runpod.md)
CONFIGS="baseline" SEEDS="0 1 2" AUTO_STOP=1 bash scripts/runpod/run_experiments.sh

# Nearest-neighbour baseline
python scripts/nn_baseline.py --manifest data/colab_data_v2/manifest.json --train-split train --eval-split test_real --out results/nn_test_real.json

# Annotation QA
python scripts/qa_annotations.py data/annotations --images data/images

# Generate synthetic data
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --factory-startup --python src/tools/blender_render.py -- --count 500 --engine BLENDER_EEVEE --samples 32 --seed 42

# Convert / augment synthetic data
python src/tools/blender_to_cvn.py
python scripts/generate.py augment --out data/blender_aug

# Prepare for Roboflow upload (from the leak-free splits)
python scripts/prepare_roboflow.py --from-split-dir data/colab_data_v2

# Launch browser annotator / inference review server
python src/tools/serve_annotator.py
python src/tools/serve_review.py
```

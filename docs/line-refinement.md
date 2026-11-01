# Line-aware homography refinement (C1) and multi-hypothesis verification (C2)

Date: 2026-11-01. Code: `src/inference/line_refine.py`, wired into `CourtPredictor`
(`src/inference/predict.py`) and `src/evaluation/evaluate.py`. Validation:
`scripts/validate_line_refine.py`. Tests: `tests/test_line_refine.py`.

## What it does

The segmentation head was unused at inference. It now feeds two steps after
keypoint decoding. Both run on the model's square canvas, where `seg_mask` lives.

- **C2, multi-hypothesis.** MAGSAC runs on two keypoint subsets (all candidates
  and the more confident half) at three inlier thresholds (0.5 / 1 / 2 % of the
  long side). Each fit is scored by the F1 agreement between the projected
  painted court lines and `seg_mask`, with a 3 px tolerance. The default fit
  (all candidates, 1 %) is kept unless another beats it by more than 0.05 F1.
  - When the default fit fails, another hypothesis can still succeed, so
    completeness improves.
  - Court-half subsets (`rows0-2` / `rows3-5`) are available but off by
    default: they extrapolate badly and sometimes won on F1 by a small margin
    while being 100+ px wrong.
- **C1, refinement.** Robust Levenberg–Marquardt over the 8 DoF of H. The cost
  is `(1-α)·mean ρ(keypoint reprojection)` over MAGSAC inliers, plus
  `α·mean ρ(point–line distance)` over seg line pixels.
  - Defaults: Huber loss with δ = 2 canvas px, α = 0.5, analytic Jacobian, up
    to 3000 subsampled line pixels.
  - Each line pixel is assigned to the nearest projected painted segment (the
    virtual net segment K12–K17 is excluded). Pixels beyond the gate are
    dropped. The assignment is redone 3 times with gates of 12 → 6 → 4 px.
  - The result is reverted to the MAGSAC H if handedness flips, an inlier goes
    behind the camera, the outer corners turn non-convex, or the median inlier
    moves more than 12 px.
- **Reject flag.** `CourtDetection.line_score` (the F1 of the final H) is always
  reported. With `reject_threshold` set, H is dropped below that score.
  - In that case the keypoints fall back to the raw decoder output,
    `rejected=True`, and `homography_status="rejected (...)"`.
  - Evaluation now reports `completeness` (share of images with a kept model H),
    `model_court_iou_mean` (court IoU of the predictor's own H; failures count
    as 0) and `n_rejected`.

Defaults: `CourtPredictor(line_refine=True, multi_hypothesis=True, reject_threshold=None)`.
The evaluation CLI accepts `--line-refine/--no-line-refine` (C1+C2),
`--multi-hypothesis/--no-multi-hypothesis` (C2 alone), `--reject-threshold`
and `--line-alpha`. Runtime on CPU is about 130 ms per image for C1+C2, small
next to the forward pass.

## Measured effect (no trained model)

`python scripts/validate_line_refine.py --data data/colab_data_v2 --splits test_real test_roboflow --synthetic 200`

Keypoints are GT plus Gaussian noise (σ = 3 canvas px) plus 15 % outliers
(displaced 15–60 px). Errors are in 640-px canvas pixels, the evaluation's
reference scale. Errors are compared on images that every method fitted. "fail"
and IoU (where failure counts as 0) cover all images.

- **GT-derived** (87 test annotations): the seg is the GT line mask, i.e. the
  seg training target. It is corrupted with occlusions, parallel distractor
  lines, random strokes, blur and speckle.
- **Synthetic** (200 images with a known H): rendered courts with neighbours,
  players and noise, segmented by a classical top-hat line detector. Nothing
  here is derived from the labels.

| input | method | fail % | H err mean / median | H PCK@5 % | court IoU % |
|---|---|---|---|---|---|
| GT-derived | MAGSAC | 8.0 | 5.70 / 2.23 | 87.2 | 84.4 |
| | C1 keypoints only (α = 0) | 8.0 | 5.43 / 2.16 | 88.5 | 84.9 |
| | C2 | 2.3 | 3.65 / 2.00 | 91.0 | 89.9 |
| | C1 | 8.0 | 3.94 / 1.88 | 91.6 | 85.1 |
| | **C1 + C2** | **2.3** | **3.20 / 1.76** | **95.0** | **90.5** |
| Synthetic | MAGSAC | 0.0 | 3.16 / 2.18 | 87.8 | 96.9 |
| | C1 keypoints only (α = 0) | 0.0 | 2.85 / 1.97 | 90.4 | 97.1 |
| | C2 | 0.0 | 2.35 / 1.79 | 95.8 | 97.8 |
| | C1 | 0.0 | 1.84 / 1.27 | 96.7 | 98.5 |
| | **C1 + C2** | **0.0** | **1.44 / 1.08** | **98.6** | **98.9** |

- **The line term carries the gain.** The α = 0 control, the same optimiser
  without lines, gains little.
- **C1 + C2 roughly halves the homography error**: 44 % lower on GT-derived
  input, 54 % lower on synthetic. It also cuts fit failures on GT-derived input
  from 8 % to 2 %.
- **Choice of α.** These line masks are geometrically exact, so a larger α is
  always better here (synthetic H error 1.84 → 1.06 → 0.57 px at α = 0.5 / 0.7 /
  0.9). A trained seg head will have systematic errors, so α stays at
  PnLCalib's 0.5 until it can be tuned on the trained model's val split.
- **The default `fill="in_frame"` hides most of the gain.** Inlier keypoints
  keep their decoded positions, and only outliers and unseen points take the
  reprojection. Output keypoint error moves only 3.93 → 3.19 px (synthetic),
  while `fill="all"` reaches 1.44 px. Re-decide whether line refinement should
  replace inliers too, using the trained model, where raw keypoint noise may be
  much smaller than σ = 3 px.

**Reject flag.** This shows how well the line score separates correct from
wrong fits. Values are the share below each threshold.

| input | fit | median F1 | < 0.2 | < 0.3 | < 0.4 |
|---|---|---|---|---|---|
| GT-derived | correct (MAGSAC) | 0.66 | 2 | 4 | 5 |
| | labels shifted one row | 0.50 | 8 | 15 | 25 |
| Synthetic | correct (MAGSAC) | 0.77 | 0 | 0 | 1 |
| | labels shifted one row | 0.51 | 3 | 8 | 16 |
| | another random court | 0.14 | 85 | 97 | 100 |

A threshold around 0.3 rejects almost every grossly wrong court and almost no
correct ones. It catches only a minority of one-row label shifts: those keep
the sidelines aligned and overlap other transverse lines. Courts are also
symmetric under rotation by 180°, so the line score cannot detect a 180°
relabeling at all. On GT-derived input, "another court" was the previous
annotation, often the same session and camera, so that row is omitted.
Rejection therefore stays off by default. Calibrate it on the trained model's
val split.

## Smoke test with the legacy Run-7 checkpoint

This is not a fair test: old labels, squashed input, and an unreliable
visibility head. Both runs used `--n-boot 200`.

| split | | PCK@10 px | sym PCK_norm@0.02 | court IoU (refit) | completeness | model-H court IoU |
|---|---|---|---|---|---|---|
| test_real (41) | `--no-line-refine` | 25.9 | 51.7 | 32.6 | 53.7 | 22.1 |
| | `--line-refine` | 27.2 | 53.6 | 38.6 | 78.0 | 38.4 |
| test_roboflow (46) | `--no-line-refine` | 24.0 | 38.7 | 96.3 | 30.4 | 28.0 |
| | `--line-refine` | 24.0 | 38.3 | 95.7 | 32.6 | 29.6 |

On test_roboflow, most images never reach 6 MAGSAC inliers with this
checkpoint, so there is little to refine.

## Still to do

- Rerun the A/B with the new-pipeline checkpoint:
  `python -m src.evaluation.evaluate ... --line-refine` vs `--no-line-refine`.
  No `checkpoints/*/best_model.pt` from the new pipeline existed when this note
  was written.
- Using the trained model, tune α, the reject threshold and the fill policy on
  val, then report on test.
- C4 (radial distortion k1) can be added as a ninth parameter of the same
  optimiser.

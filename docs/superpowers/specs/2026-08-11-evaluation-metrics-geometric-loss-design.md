# Evaluation Metrics & Geometric Consistency Loss — Design Spec

**Goal:** Add quantitative evaluation metrics (PCK, reprojection error, court IoU, segmentation IoU) and differentiable geometric consistency losses to CourtVisionNet, and update the keypoint layout from 31 to 30 (6x5 grid, no net-center intersection).

**Architecture:** Pure-function metrics module + three new loss terms integrated into the existing CourtVisionLoss. Metrics run during validation and via a standalone CLI evaluation script.

## Global Constraints

- Python 3.10+, PyTorch, NumPy, OpenCV, Shapely (new dependency)
- All metric functions are stateless pure functions (numpy in, scalars/arrays out)
- Geometric losses must be differentiable (PyTorch tensors, no detach/numpy conversion in the loss path)
- 30 keypoints in a 6-row x 5-column grid (K0-K29)
- No net line keypoints — the net is a physical object, not a painted line
- Center line is painted from baseline to short service on each half; K12-K17 across the net is virtual (not painted but inferrable)

---

## 1. Keypoint Layout Update (31 → 30)

### 30-Keypoint Template (6 rows x 5 columns)

Court dimensions: 13.4m x 6.1m. Origin at K0 (top-left corner).

Columns (y-axis, court width):
- Doubles sideline top: y = 0.0
- Singles sideline top: y = 0.46m
- Center line: y = 3.05m
- Singles sideline bottom: y = 5.64m
- Doubles sideline bottom: y = 6.1m

Rows (x-axis, court length):
- Baseline L: x = 0.0
- Long service L: x = 0.76m
- Short service L: x = 4.72m (NET_POSITION - 1.98m)
- Short service R: x = 8.68m (NET_POSITION + 1.98m)
- Long service R: x = 12.64m
- Baseline R: x = 13.4m

| Row | Dbl-top (y=0) | Sgl-top (y=0.46) | Center (y=3.05) | Sgl-bot (y=5.64) | Dbl-bot (y=6.1) |
|-----|:---:|:---:|:---:|:---:|:---:|
| Baseline L (x=0) | K0 | K1 | K2 | K3 | K4 |
| Long Svc L (x=0.76) | K5 | K6 | K7 | K8 | K9 |
| Short Svc L (x=4.72) | K10 | K11 | K12 | K13 | K14 |
| Short Svc R (x=8.68) | K15 | K16 | K17 | K18 | K19 |
| Long Svc R (x=12.64) | K20 | K21 | K22 | K23 | K24 |
| Baseline R (x=13.4) | K25 | K26 | K27 | K28 | K29 |

### 9 Intersection Types

| Type | Horizontal | Vertical | Keypoints |
|------|-----------|----------|-----------|
| 1 | Baseline | Doubles sideline | K0, K4, K25, K29 |
| 2 | Baseline | Singles sideline | K1, K3, K26, K28 |
| 3 | Baseline | Center line | K2, K27 |
| 4 | Long service | Doubles sideline | K5, K9, K20, K24 |
| 5 | Long service | Singles sideline | K6, K8, K21, K23 |
| 6 | Long service | Center line | K7, K22 |
| 7 | Short service | Doubles sideline | K10, K14, K15, K19 |
| 8 | Short service | Singles sideline | K11, K13, K16, K18 |
| 9 | Short service | Center line | K12, K17 |

### Outer corners: K0, K4, K25, K29

### FLIP_PAIRS (horizontal flip, left↔right across center column)

Pairs: (0,4), (1,3), (5,9), (6,8), (10,14), (11,13), (15,19), (16,18), (20,24), (21,23), (25,29), (26,28) — 12 pairs.

Unpaired (center column): K2, K7, K12, K17, K22, K27.

### Court Lines

Painted segments:
- Doubles sideline top (y=0): K0→K5→K10→K15→K20→K25
- Doubles sideline bot (y=6.1): K4→K9→K14→K19→K24→K29
- Singles sideline top (y=0.46): K1→K6→K11→K16→K21→K26
- Singles sideline bot (y=5.64): K3→K8→K13→K18→K23→K28
- Center line L half: K2→K7→K12 (painted)
- Center line R half: K17→K22→K27 (painted)
- Center line across net: K12→K17 (virtual — not painted, used for homography)
- Baseline L: K0→K1→K2→K3→K4
- Baseline R: K25→K26→K27→K28→K29
- Long service L: K5→K6→K7→K8→K9
- Long service R: K20→K21→K22→K23→K24
- Short service L: K10→K11→K12→K13→K14
- Short service R: K15→K16→K17→K18→K19

### Cascade Updates

All files referencing 31 keypoints must update to 30:
- `src/court_geometry.py` — template, flip pairs, corner indices, court lines, keypoint names
- `src/models/courtvisionnet.py` — default num_keypoints=30
- `src/models/keypoint_head.py` — default num_keypoints=30
- `src/training/config.py` — num_keypoints=30
- `src/training/dataset.py` — import NUM_KEYPOINTS
- `src/inference/predict.py` — import NUM_KEYPOINTS
- `src/inference/visualize.py` — colors and names for 30
- `src/tools/annotator.html` — NK=30, template, court diagram
- `src/tools/annotator.py` — corner indices, colors
- All test files — shape assertions, mock data

---

## 2. Evaluation Metrics

### File: `src/evaluation/metrics.py`

Four pure functions:

#### `pck_at_k(pred_kps, gt_kps, visibility, k, image_size=640)`
- pred_kps, gt_kps: (30, 2) normalized [0,1]
- visibility: (30,) ground truth visibility flags
- k: pixel threshold
- Returns: (per_keypoint: (30,) bool array, mean: float)
- Distance computed in pixel space (multiply by image_size before comparing to k)

#### `mean_reprojection_error(pred_kps, gt_kps, visibility, image_w, image_h)`
- Estimate homography from visible predicted keypoints to court template
- Project all 30 template points through homography
- Return mean Euclidean pixel distance for visible points
- Returns None if homography estimation fails

#### `court_iou(pred_kps, gt_kps, pred_vis, gt_vis)`
- Extract outer corners (K0, K4, K25, K29) from both prediction and ground truth
- Compute polygon IoU using Shapely
- Returns 0.0 if fewer than 4 corners visible on either side

#### `segmentation_iou(pred_mask, gt_mask, threshold=0.5)`
- Binarize predicted mask at threshold
- Compute intersection / union
- Returns float in [0, 1]

### File: `src/evaluation/evaluate.py`

Standalone CLI:
```
python -m src.evaluation.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --annotations data/annotations/test \
    --images data/frames/ \
    --image-size 640
```

Outputs:
- Summary table: PCK@5/10/20, MRE (mean ± std), Court IoU (mean ± std), Seg IoU (mean ± std)
- Per-keypoint PCK@10 breakdown
- Per-type PCK@10 breakdown (9 intersection types)

### Training Integration

`validate()` in `src/training/train.py` computes PCK@10 and MRE alongside loss components after each epoch. Logged but not used for model selection (best model still selected on validation loss).

---

## 3. Geometric Consistency Loss

### Added to: `src/models/losses.py`

Three new differentiable loss terms:

#### Collinearity Loss
- For each court line (row or column), take groups of 3+ visible keypoints
- Compute cross-product area of consecutive triplets: |(p2-p1) × (p3-p1)|
- Should be 0 for perfectly collinear points
- Loss = mean of squared cross-product areas across all line groups
- Line groups defined by `get_court_lines()` segments, grouped into full lines

#### Distance Ratio Loss
- For visible keypoint pairs on the same line, compute distance ratios
- Compare to known template ratios (e.g., K0→K1 / K0→K4 = 0.46/6.1)
- Loss = MSE between predicted and expected ratios
- Scale-invariant (ratios, not absolute distances)
- Only computed when both endpoints of a ratio pair are visible

#### Convexity Loss
- Outer corners K0, K4, K25, K29 must form a convex quadrilateral
- Compute cross products of consecutive edge vectors in cyclic order
- All cross products should have the same sign
- Loss = sum of ReLU penalties on sign-violating cross products
- Only active when all 4 corners are visible

### Integration into CourtVisionLoss

```python
total = (seg_weight * seg_loss
       + heatmap_weight * heatmap_loss
       + offset_weight * offset_loss
       + vis_weight * visibility_loss
       + collinear_weight * collinear_loss
       + ratio_weight * ratio_loss
       + convex_weight * convex_loss)
```

### New Config Fields in TrainConfig

- `collinear_weight: float = 0.1`
- `ratio_weight: float = 0.1`
- `convex_weight: float = 0.1`

Starting values are conservative (0.1) — tune after initial training runs.

---

## 4. New Dependency

Add `shapely>=2.0` to `requirements.txt` for polygon IoU computation in `court_iou()`.

---

## 5. What This Does NOT Include

Separate design cycles:
- Multi-court instance detection (architectural change)
- Temporal smoothing / Kalman filter (post-processing layer)
- Camera intrinsics estimation from court geometry
- Annotation batch auto-sort script

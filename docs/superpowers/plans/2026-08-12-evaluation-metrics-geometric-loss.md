# Evaluation Metrics & Geometric Consistency Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update CourtVisionNet from 31 to 30 keypoints (6×5 grid), add four evaluation metrics (PCK, MRE, Court IoU, Seg IoU), add three geometric consistency loss terms, and integrate metrics into training validation + standalone evaluation CLI.

**Architecture:** Keypoint layout change cascades through all modules first. Pure-function metrics in `src/evaluation/metrics.py`. Three differentiable geometric losses added to existing `CourtVisionLoss`. Standalone evaluation CLI in `src/evaluation/evaluate.py`.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, OpenCV, Shapely (new)

## Global Constraints

- 30 keypoints in a 6-row × 5-column grid (K0–K29), see spec for exact template coordinates
- All metric functions are stateless pure functions (numpy in, scalars/arrays out)
- Geometric losses must be differentiable (PyTorch tensors only, no `.detach()` or `.numpy()` in the loss path)
- Center line painted from baseline to short service on each half; K12→K17 across net is virtual (not painted but used for homography)
- Outer corners: K0, K4, K25, K29 (cyclic order for convexity: K0, K25, K29, K4)
- FLIP_PAIRS: 12 pairs — (0,4), (1,3), (5,9), (6,8), (10,14), (11,13), (15,19), (16,18), (20,24), (21,23), (25,29), (26,28)
- Unpaired center-column keypoints: K2, K7, K12, K17, K22, K27
- `shapely>=2.0` added to `requirements.txt`

---

### Task 1: Update court geometry to 30 keypoints

**Files:**
- Modify: `src/court_geometry.py`
- Modify: `tests/test_court_geometry.py`

**Interfaces:**
- Produces: `NUM_KEYPOINTS = 30`, `COURT_KEYPOINTS_TEMPLATE` shape (30, 2), `FLIP_PAIRS` (12 pairs), `CORNER_INDICES = [0, 25, 29, 4]` (cyclic TL→TR→BR→BL), `KEYPOINT_NAMES` (30 entries), `get_court_lines()` returning new segment pairs, `get_collinear_groups()` returning list of lists of keypoint indices on the same line

- [ ] **Step 1: Write failing tests for 30-keypoint layout**

```python
# tests/test_court_geometry.py — replace existing test_template_has_31_keypoints
# and test_template_dimensions_match_spec, add new tests

def test_template_has_30_keypoints():
    assert COURT_KEYPOINTS_TEMPLATE.shape == (30, 2)

def test_template_dimensions_match_spec():
    k0 = COURT_KEYPOINTS_TEMPLATE[0]   # (0, 0)
    k29 = COURT_KEYPOINTS_TEMPLATE[29]  # (13.4, 6.1)
    assert abs(k29[0] - k0[0] - 13.4) < 0.01
    assert abs(k29[1] - k0[1] - 6.1) < 0.01

def test_template_6x5_grid_structure():
    """Rows share the same x, columns share the same y."""
    tpl = COURT_KEYPOINTS_TEMPLATE
    # Row 0 (Baseline L): K0-K4 all at x=0
    for i in range(5):
        assert abs(tpl[i, 0] - 0.0) < 1e-6
    # Row 5 (Baseline R): K25-K29 all at x=13.4
    for i in range(25, 30):
        assert abs(tpl[i, 0] - 13.4) < 1e-6
    # Column 0 (Dbl-top): K0, K5, K10, K15, K20, K25 all at y=0
    for i in [0, 5, 10, 15, 20, 25]:
        assert abs(tpl[i, 1] - 0.0) < 1e-6

def test_num_keypoints_constant():
    from src.court_geometry import NUM_KEYPOINTS
    assert NUM_KEYPOINTS == 30

def test_flip_pairs_count():
    from src.court_geometry import FLIP_PAIRS
    assert len(FLIP_PAIRS) == 12
    paired = {i for pair in FLIP_PAIRS for i in pair}
    assert len(paired) == 24  # 12 pairs × 2
    # Center column excluded
    for center in [2, 7, 12, 17, 22, 27]:
        assert center not in paired

def test_corner_indices():
    from src.court_geometry import CORNER_INDICES
    assert set(CORNER_INDICES) == {0, 4, 25, 29}

def test_court_lines_indices_in_range():
    lines = get_court_lines()
    for start_idx, end_idx in lines:
        assert 0 <= start_idx < 30
        assert 0 <= end_idx < 30

def test_get_collinear_groups():
    from src.court_geometry import get_collinear_groups
    groups = get_collinear_groups()
    # 6 rows + 5 columns = 11 collinear groups
    assert len(groups) == 11
    # Each group has at least 3 points (smallest is a column with 6 rows)
    for group in groups:
        assert len(group) >= 2
    # Row 0 (Baseline L) should be [0, 1, 2, 3, 4]
    assert [0, 1, 2, 3, 4] in groups
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_court_geometry.py -v`
Expected: Multiple FAIL (shape mismatch, missing imports)

- [ ] **Step 3: Implement 30-keypoint geometry**

Replace the contents of `src/court_geometry.py`:

```python
import numpy as np
import cv2

COURT_LENGTH = 13.4       # meters
COURT_WIDTH_DOUBLES = 6.1  # meters
COURT_WIDTH_SINGLES = 5.18 # meters
SHORT_SERVICE_LINE = 1.98  # meters from net
NET_POSITION = 6.7         # meters from each end (center)
LONG_SERVICE_LINE_DOUBLES = 0.76  # meters from back boundary

SINGLES_SIDELINE_OFFSET = (COURT_WIDTH_DOUBLES - COURT_WIDTH_SINGLES) / 2  # 0.46m

NUM_KEYPOINTS = 30

_SS = NET_POSITION - SHORT_SERVICE_LINE   # 4.72m
_RS = NET_POSITION + SHORT_SERVICE_LINE   # 8.68m
_LS = LONG_SERVICE_LINE_DOUBLES           # 0.76m
_RLS = COURT_LENGTH - _LS                 # 12.64m
_SO = SINGLES_SIDELINE_OFFSET             # 0.46m
_CY = COURT_WIDTH_DOUBLES / 2            # 3.05m
_SB = COURT_WIDTH_DOUBLES - _SO          # 5.64m

# Horizontal-flip pairs: left↔right across center column (y=0 ↔ y=6.1).
# Center-column keypoints (K2, K7, K12, K17, K22, K27) have no pair.
FLIP_PAIRS = [
    (0, 4),    # Baseline-L dbl-top ↔ dbl-bot
    (1, 3),    # Baseline-L sgl-top ↔ sgl-bot
    (5, 9),    # LongSvc-L dbl-top ↔ dbl-bot
    (6, 8),    # LongSvc-L sgl-top ↔ sgl-bot
    (10, 14),  # ShortSvc-L dbl-top ↔ dbl-bot
    (11, 13),  # ShortSvc-L sgl-top ↔ sgl-bot
    (15, 19),  # ShortSvc-R dbl-top ↔ dbl-bot
    (16, 18),  # ShortSvc-R sgl-top ↔ sgl-bot
    (20, 24),  # LongSvc-R dbl-top ↔ dbl-bot
    (21, 23),  # LongSvc-R sgl-top ↔ sgl-bot
    (25, 29),  # Baseline-R dbl-top ↔ dbl-bot
    (26, 28),  # Baseline-R sgl-top ↔ sgl-bot
]

# Outer corners in cyclic order: TL → TR → BR → BL
CORNER_INDICES = [0, 25, 29, 4]

KEYPOINT_NAMES = [
    # Row 0: Baseline L (x=0)
    "Baseline-L / Dbl-top",    # K0
    "Baseline-L / Sgl-top",    # K1
    "Baseline-L / Center",     # K2
    "Baseline-L / Sgl-bot",    # K3
    "Baseline-L / Dbl-bot",    # K4
    # Row 1: Long service L (x=0.76)
    "LongSvc-L / Dbl-top",    # K5
    "LongSvc-L / Sgl-top",    # K6
    "LongSvc-L / Center",     # K7
    "LongSvc-L / Sgl-bot",    # K8
    "LongSvc-L / Dbl-bot",    # K9
    # Row 2: Short service L (x=4.72)
    "ShortSvc-L / Dbl-top",   # K10
    "ShortSvc-L / Sgl-top",   # K11
    "ShortSvc-L / Center",    # K12
    "ShortSvc-L / Sgl-bot",   # K13
    "ShortSvc-L / Dbl-bot",   # K14
    # Row 3: Short service R (x=8.68)
    "ShortSvc-R / Dbl-top",   # K15
    "ShortSvc-R / Sgl-top",   # K16
    "ShortSvc-R / Center",    # K17
    "ShortSvc-R / Sgl-bot",   # K18
    "ShortSvc-R / Dbl-bot",   # K19
    # Row 4: Long service R (x=12.64)
    "LongSvc-R / Dbl-top",    # K20
    "LongSvc-R / Sgl-top",    # K21
    "LongSvc-R / Center",     # K22
    "LongSvc-R / Sgl-bot",    # K23
    "LongSvc-R / Dbl-bot",    # K24
    # Row 5: Baseline R (x=13.4)
    "Baseline-R / Dbl-top",    # K25
    "Baseline-R / Sgl-top",    # K26
    "Baseline-R / Center",     # K27
    "Baseline-R / Sgl-bot",    # K28
    "Baseline-R / Dbl-bot",    # K29
]

# 6 rows × 5 columns = 30 keypoints.
# Origin at K0 (top-left corner), x along length, y along width.
COURT_KEYPOINTS_TEMPLATE = np.array([
    # Row 0: Baseline L (x=0)
    [0.0, 0.0],                    # K0
    [0.0, _SO],                    # K1
    [0.0, _CY],                    # K2
    [0.0, _SB],                    # K3
    [0.0, COURT_WIDTH_DOUBLES],    # K4
    # Row 1: Long service L (x=0.76)
    [_LS, 0.0],                    # K5
    [_LS, _SO],                    # K6
    [_LS, _CY],                    # K7
    [_LS, _SB],                    # K8
    [_LS, COURT_WIDTH_DOUBLES],    # K9
    # Row 2: Short service L (x=4.72)
    [_SS, 0.0],                    # K10
    [_SS, _SO],                    # K11
    [_SS, _CY],                    # K12
    [_SS, _SB],                    # K13
    [_SS, COURT_WIDTH_DOUBLES],    # K14
    # Row 3: Short service R (x=8.68)
    [_RS, 0.0],                    # K15
    [_RS, _SO],                    # K16
    [_RS, _CY],                    # K17
    [_RS, _SB],                    # K18
    [_RS, COURT_WIDTH_DOUBLES],    # K19
    # Row 4: Long service R (x=12.64)
    [_RLS, 0.0],                   # K20
    [_RLS, _SO],                   # K21
    [_RLS, _CY],                   # K22
    [_RLS, _SB],                   # K23
    [_RLS, COURT_WIDTH_DOUBLES],   # K24
    # Row 5: Baseline R (x=13.4)
    [COURT_LENGTH, 0.0],                   # K25
    [COURT_LENGTH, _SO],                   # K26
    [COURT_LENGTH, _CY],                   # K27
    [COURT_LENGTH, _SB],                   # K28
    [COURT_LENGTH, COURT_WIDTH_DOUBLES],   # K29
], dtype=np.float64)


def get_court_lines():
    """Return (start_idx, end_idx) pairs for all court line segments."""
    return [
        # Doubles sideline top (y=0): K0→K5→K10→K15→K20→K25
        (0, 5), (5, 10), (10, 15), (15, 20), (20, 25),
        # Doubles sideline bot (y=6.1): K4→K9→K14→K19→K24→K29
        (4, 9), (9, 14), (14, 19), (19, 24), (24, 29),
        # Singles sideline top (y=0.46): K1→K6→K11→K16→K21→K26
        (1, 6), (6, 11), (11, 16), (16, 21), (21, 26),
        # Singles sideline bot (y=5.64): K3→K8→K13→K18→K23→K28
        (3, 8), (8, 13), (13, 18), (18, 23), (23, 28),
        # Center line L half (painted): K2→K7→K12
        (2, 7), (7, 12),
        # Center line R half (painted): K17→K22→K27
        (17, 22), (22, 27),
        # Center line across net (virtual, for homography): K12→K17
        (12, 17),
        # Baseline L: K0→K1→K2→K3→K4
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Baseline R: K25→K26→K27→K28→K29
        (25, 26), (26, 27), (27, 28), (28, 29),
        # Long service L: K5→K6→K7→K8→K9
        (5, 6), (6, 7), (7, 8), (8, 9),
        # Long service R: K20→K21→K22→K23→K24
        (20, 21), (21, 22), (22, 23), (23, 24),
        # Short service L: K10→K11→K12→K13→K14
        (10, 11), (11, 12), (12, 13), (13, 14),
        # Short service R: K15→K16→K17→K18→K19
        (15, 16), (16, 17), (17, 18), (18, 19),
    ]


def get_collinear_groups():
    """Return groups of keypoint indices that must be collinear.

    Each group is a list of indices lying on the same court line.
    6 rows + 5 columns = 11 groups.
    """
    return [
        # 6 rows (horizontal lines, same x)
        [0, 1, 2, 3, 4],           # Baseline L
        [5, 6, 7, 8, 9],           # Long service L
        [10, 11, 12, 13, 14],      # Short service L
        [15, 16, 17, 18, 19],      # Short service R
        [20, 21, 22, 23, 24],      # Long service R
        [25, 26, 27, 28, 29],      # Baseline R
        # 5 columns (vertical lines, same y)
        [0, 5, 10, 15, 20, 25],    # Doubles sideline top
        [1, 6, 11, 16, 21, 26],    # Singles sideline top
        [2, 7, 12, 17, 22, 27],    # Center line
        [3, 8, 13, 18, 23, 28],    # Singles sideline bot
        [4, 9, 14, 19, 24, 29],    # Doubles sideline bot
    ]
```

Keep `compute_homography`, `project_points`, `validate_quadrilateral`, and `generate_line_mask` unchanged — they are index-agnostic.

- [ ] **Step 4: Update all other test assertions from 31 → 30**

In `tests/test_court_geometry.py`, update `test_generate_line_mask` to use 30-keypoint arrays:

```python
def test_generate_line_mask():
    kps = [[0.1, 0.1], [0.1, 0.15], [0.1, 0.5], [0.1, 0.85], [0.1, 0.9]]  # K0-K4
    kps += [[-1, -1]] * 25  # K5-K29
    keypoints = np.array(kps, dtype=np.float64)
    visibility = [1, 1, 1, 1, 1] + [0] * 25
    mask = generate_line_mask(keypoints, visibility, width=640, height=640, line_thickness=3)
    assert mask.shape == (640, 640)
    assert mask.dtype == np.uint8
    assert mask.max() == 255
    assert mask.min() == 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_court_geometry.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/court_geometry.py tests/test_court_geometry.py
git commit -m "feat: update court geometry to 30 keypoints (6×5 grid)"
```

---

### Task 2: Cascade 31→30 keypoint update across codebase

**Files:**
- Modify: `src/models/courtvisionnet.py` — default `num_keypoints=30`
- Modify: `src/models/keypoint_head.py` — default `num_keypoints=30`
- Modify: `src/training/config.py` — `num_keypoints=30`
- Modify: `src/training/dataset.py` — already imports `NUM_KEYPOINTS`, no code change needed
- Modify: `src/inference/predict.py` — update docstring comments from "14" to "30"
- Modify: `src/inference/visualize.py` — 30-element `KEYPOINT_COLORS`, update `KEYPOINT_NAMES`
- Modify: `src/tools/annotator.py` — `CORNER_INDICES = [0, 25, 29, 4]`, 30-element `KEYPOINT_COLORS`
- Modify: `src/tools/annotator.html` — `NK=30`, new `TPL`, `LINES`, `CORNERS` arrays, updated court diagram
- Modify: `tests/test_heads.py` — `num_keypoints=30`
- Modify: `tests/test_courtvisionnet.py` — shapes to 30
- Modify: `tests/test_dataset.py` — shapes to 30, mock data to 30
- Modify: `tests/test_inference.py` — shapes to 30, visibility indices for corners
- Modify: `tests/test_training.py` — mock data to 30
- Modify: `tests/test_annotator_logic.py` — corner indices K0/K4/K25/K29, shapes to 30
- Modify: `tests/test_augmentation.py` — new FLIP_PAIRS, center-column checks K2/K7/K12/K17/K22/K27
- Modify: `requirements.txt` — add `shapely>=2.0`
- Modify: `README.md` — update keypoint table

**Interfaces:**
- Consumes: `NUM_KEYPOINTS=30`, `COURT_KEYPOINTS_TEMPLATE` (30,2), `FLIP_PAIRS` (12 pairs), `CORNER_INDICES`, `get_court_lines()` from Task 1
- Produces: All modules and tests passing with 30 keypoints

- [ ] **Step 1: Update model defaults**

In `src/models/courtvisionnet.py` line 19: change `num_keypoints=31` to `num_keypoints=30`.

In `src/models/keypoint_head.py` line 15: change `num_keypoints=31` to `num_keypoints=30`.

In `src/training/config.py` line 22: change `num_keypoints: int = 31` to `num_keypoints: int = 30`.

- [ ] **Step 2: Update predict.py docstrings**

In `src/inference/predict.py`, update the `CourtDetection` docstring comment from `(14, 2)` to `(30, 2)` and the `estimate_homography_and_fill` docstring from "14" to "30".

- [ ] **Step 3: Update visualize.py**

Replace `KEYPOINT_COLORS` and `KEYPOINT_NAMES`:

```python
KEYPOINT_COLORS = (
    [(255, 0, 0)] * 5         # K0-K4:   Baseline L (red)
    + [(255, 136, 0)] * 5     # K5-K9:   Long Svc L (orange)
    + [(34, 204, 34)] * 5     # K10-K14: Short Svc L (green)
    + [(51, 136, 255)] * 5    # K15-K19: Short Svc R (blue)
    + [(204, 68, 255)] * 5    # K20-K24: Long Svc R (purple)
    + [(255, 68, 136)] * 5    # K25-K29: Baseline R (pink)
)

KEYPOINT_NAMES = [f"K{i}" for i in range(30)]
```

- [ ] **Step 4: Update annotator.py**

Change `CORNER_INDICES = [0, 27, 30, 3]` to `CORNER_INDICES = [0, 25, 29, 4]`.

Update `KEYPOINT_COLORS` to 30 elements (6 groups of 5):

```python
KEYPOINT_COLORS = (
    ["#FF0000"] * 5     # K0-K4:   Baseline L (red)
    + ["#FF8800"] * 5   # K5-K9:   Long Svc L (orange)
    + ["#22CC22"] * 5   # K10-K14: Short Svc L (green)
    + ["#3388FF"] * 5   # K15-K19: Short Svc R (blue)
    + ["#CC44FF"] * 5   # K20-K24: Long Svc R (purple)
    + ["#FF4488"] * 5   # K25-K29: Baseline R (pink)
)
```

Update `auto_sort_corners` to map to K0(TL), K25(TR), K29(BR), K4(BL).

- [ ] **Step 5: Update annotator.html**

Change `NK=31` to `NK=30`. Replace `TPL`, `LINES`, `CORNERS` arrays to match the new 30-keypoint layout from `src/court_geometry.py`. Update the SVG court diagram to show 6 rows × 5 columns. Update color groups to 6 groups of 5.

- [ ] **Step 6: Update all test files**

Apply the same pattern as the 14→31 migration:

- `tests/test_heads.py`: `num_keypoints=30` in constructors and shape assertions
- `tests/test_courtvisionnet.py`: all tensor shapes to 30
- `tests/test_dataset.py`: mock annotations 30 keypoints, `(30, 160, 160)` heatmap shape, `(30, 2)` keypoint shape
- `tests/test_inference.py`: shapes to 30, homography test visibility `[[0, 4, 25, 29]] = 1.0`, predictor shapes `(30, 2)` and `(30,)`
- `tests/test_training.py`: mock data 30 keypoints (4 visible + 26 invisible)
- `tests/test_annotator_logic.py`: `len(state.keypoints) == 30`, corner tests use K0/K4/K25/K29, homography test uses K0/K4/K25/K29, shape assertions (30, 2)
- `tests/test_augmentation.py`: new 12-pair FLIP_PAIRS set, center-column exclusion checks K2/K7/K12/K17/K22/K27, visibility swap test uses K0↔K4 pair

- [ ] **Step 7: Update requirements.txt and README.md**

Add `shapely>=2.0` to `requirements.txt`.

Update `README.md`: change "31 court keypoints" to "30 court keypoints", replace the keypoint table with the new 6×5 grid layout.

- [ ] **Step 8: Run full test suite**

Run: `pytest tests/ -v`
Expected: All 90 tests PASS

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: cascade 31→30 keypoint update across all modules and tests"
```

---

### Task 3: Evaluation metrics module

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `COURT_KEYPOINTS_TEMPLATE` (30,2), `CORNER_INDICES`, `compute_homography`, `project_points` from `src/court_geometry`
- Produces:
  - `pck_at_k(pred_kps: np.ndarray, gt_kps: np.ndarray, visibility: np.ndarray, k: float, image_size: int = 640) -> tuple[np.ndarray, float]`
  - `mean_reprojection_error(pred_kps: np.ndarray, gt_kps: np.ndarray, visibility: np.ndarray, image_w: int, image_h: int) -> float | None`
  - `court_iou(pred_kps: np.ndarray, gt_kps: np.ndarray, pred_vis: np.ndarray, gt_vis: np.ndarray) -> float`
  - `segmentation_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
import numpy as np
import pytest
from src.evaluation.metrics import (
    pck_at_k,
    mean_reprojection_error,
    court_iou,
    segmentation_iou,
)
from src.court_geometry import NUM_KEYPOINTS, COURT_KEYPOINTS_TEMPLATE, CORNER_INDICES


class TestPckAtK:
    def test_perfect_prediction(self):
        kps = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        vis = np.ones(NUM_KEYPOINTS, dtype=np.float32)
        per_kp, mean_acc = pck_at_k(kps, kps, vis, k=5, image_size=640)
        assert mean_acc == 1.0
        assert per_kp.all()

    def test_all_invisible_returns_zero(self):
        pred = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        gt = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        per_kp, mean_acc = pck_at_k(pred, gt, vis, k=5, image_size=640)
        assert mean_acc == 0.0

    def test_threshold_discriminates(self):
        gt = np.full((NUM_KEYPOINTS, 2), 0.5, dtype=np.float32)
        pred = gt.copy()
        pred[0] = [0.5 + 15.0 / 640, 0.5]  # 15px off
        vis = np.ones(NUM_KEYPOINTS, dtype=np.float32)
        _, acc_at_10 = pck_at_k(pred, gt, vis, k=10, image_size=640)
        _, acc_at_20 = pck_at_k(pred, gt, vis, k=20, image_size=640)
        assert acc_at_10 < 1.0  # K0 outside 10px
        assert acc_at_20 == 1.0  # K0 inside 20px


class TestMeanReprojectionError:
    def test_perfect_homography_gives_zero_error(self):
        image_w, image_h = 200, 100
        sx, sy = image_w / 13.4, image_h / 6.1
        H = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
        from src.court_geometry import project_points
        projected_px = project_points(H, COURT_KEYPOINTS_TEMPLATE)
        kps_norm = projected_px / np.array([image_w, image_h])
        vis = np.ones(NUM_KEYPOINTS, dtype=np.float32)
        mre = mean_reprojection_error(kps_norm, kps_norm, vis, image_w, image_h)
        assert mre is not None
        assert mre < 1.0  # sub-pixel

    def test_too_few_visible_returns_none(self):
        pred = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        gt = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        vis[0] = 1.0  # only 1 visible
        mre = mean_reprojection_error(pred, gt, vis, 640, 640)
        assert mre is None


class TestCourtIou:
    def test_identical_corners_gives_one(self):
        kps = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        kps[0] = [0.1, 0.1]   # K0 TL
        kps[4] = [0.1, 0.9]   # K4 BL
        kps[25] = [0.9, 0.1]  # K25 TR
        kps[29] = [0.9, 0.9]  # K29 BR
        vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        for i in CORNER_INDICES:
            vis[i] = 1.0
        iou = court_iou(kps, kps, vis, vis)
        assert abs(iou - 1.0) < 1e-6

    def test_missing_corner_gives_zero(self):
        kps = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        vis[0] = 1.0  # only 1 corner
        iou = court_iou(kps, kps, vis, vis)
        assert iou == 0.0

    def test_partial_overlap(self):
        kps_a = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        kps_a[0] = [0.0, 0.0]; kps_a[4] = [0.0, 1.0]
        kps_a[25] = [1.0, 0.0]; kps_a[29] = [1.0, 1.0]
        kps_b = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        kps_b[0] = [0.5, 0.0]; kps_b[4] = [0.5, 1.0]
        kps_b[25] = [1.5, 0.0]; kps_b[29] = [1.5, 1.0]
        vis = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        for i in CORNER_INDICES:
            vis[i] = 1.0
        iou = court_iou(kps_a, kps_b, vis, vis)
        assert 0.3 < iou < 0.4  # 50% overlap on unit square → IoU = 1/3


class TestSegmentationIou:
    def test_identical_masks(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[10:50, 10:50] = 1.0
        assert segmentation_iou(mask, mask) == 1.0

    def test_no_overlap(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        pred[:32, :] = 1.0
        gt = np.zeros((64, 64), dtype=np.float32)
        gt[32:, :] = 1.0
        assert segmentation_iou(pred, gt) == 0.0

    def test_empty_masks(self):
        empty = np.zeros((64, 64), dtype=np.float32)
        assert segmentation_iou(empty, empty) == 0.0

    def test_threshold_applied(self):
        pred = np.full((64, 64), 0.4, dtype=np.float32)  # all below 0.5
        gt = np.ones((64, 64), dtype=np.float32)
        assert segmentation_iou(pred, gt) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement metrics**

Create `src/evaluation/__init__.py` (empty file).

Create `src/evaluation/metrics.py`:

```python
"""Evaluation metrics for CourtVisionNet."""

import numpy as np
from shapely.geometry import Polygon

from src.court_geometry import (
    CORNER_INDICES,
    COURT_KEYPOINTS_TEMPLATE,
    NUM_KEYPOINTS,
    compute_homography,
    project_points,
)

_MIN_POINTS = 4


def pck_at_k(pred_kps, gt_kps, visibility, k, image_size=640):
    """Percentage of Correct Keypoints at threshold k pixels.

    Args:
        pred_kps: (30, 2) normalized [0,1] predicted keypoints
        gt_kps: (30, 2) normalized [0,1] ground truth keypoints
        visibility: (30,) ground truth visibility (1.0=visible)
        k: pixel distance threshold
        image_size: image dimension for converting to pixels

    Returns:
        (per_keypoint, mean_accuracy) where per_keypoint is (30,) bool
        and mean_accuracy is float in [0, 1].
    """
    pred_px = np.asarray(pred_kps, dtype=np.float64) * image_size
    gt_px = np.asarray(gt_kps, dtype=np.float64) * image_size
    vis = np.asarray(visibility, dtype=np.float64)

    dists = np.linalg.norm(pred_px - gt_px, axis=1)
    correct = (dists < k) & (vis > 0.5)
    visible_mask = vis > 0.5

    n_visible = visible_mask.sum()
    if n_visible == 0:
        return np.zeros(NUM_KEYPOINTS, dtype=bool), 0.0

    mean_acc = float(correct.sum() / n_visible)
    return correct, mean_acc


def mean_reprojection_error(pred_kps, gt_kps, visibility, image_w, image_h):
    """Mean reprojection error in pixels.

    Estimates homography from visible predicted keypoints, projects
    the full template, and measures pixel distance to ground truth
    for visible points.

    Returns None if homography estimation fails.
    """
    pred = np.asarray(pred_kps, dtype=np.float64)
    gt = np.asarray(gt_kps, dtype=np.float64)
    vis = np.asarray(visibility, dtype=np.float64)
    visible_mask = vis > 0.5

    if visible_mask.sum() < _MIN_POINTS:
        return None

    src_pts = COURT_KEYPOINTS_TEMPLATE[visible_mask]
    dst_pts_px = pred[visible_mask] * np.array([image_w, image_h])

    try:
        H = compute_homography(src_pts, dst_pts_px)
    except ValueError:
        return None

    projected_px = project_points(H, COURT_KEYPOINTS_TEMPLATE)
    gt_px = gt * np.array([image_w, image_h])

    errors = np.linalg.norm(projected_px[visible_mask] - gt_px[visible_mask], axis=1)
    return float(errors.mean())


def court_iou(pred_kps, gt_kps, pred_vis, gt_vis):
    """IoU of the quadrilateral formed by the 4 outer court corners.

    Returns 0.0 if fewer than 4 corners are visible on either side.
    """
    pred = np.asarray(pred_kps, dtype=np.float64)
    gt = np.asarray(gt_kps, dtype=np.float64)
    p_vis = np.asarray(pred_vis, dtype=np.float64)
    g_vis = np.asarray(gt_vis, dtype=np.float64)

    for vis in (p_vis, g_vis):
        if not all(vis[i] > 0.5 for i in CORNER_INDICES):
            return 0.0

    pred_corners = pred[CORNER_INDICES]
    gt_corners = gt[CORNER_INDICES]

    try:
        poly_pred = Polygon(pred_corners)
        poly_gt = Polygon(gt_corners)
        if not poly_pred.is_valid or not poly_gt.is_valid:
            return 0.0
        intersection = poly_pred.intersection(poly_gt).area
        union = poly_pred.union(poly_gt).area
        if union == 0:
            return 0.0
        return float(intersection / union)
    except Exception:
        return 0.0


def segmentation_iou(pred_mask, gt_mask, threshold=0.5):
    """Binary IoU between predicted and ground truth segmentation masks.

    Args:
        pred_mask: (H, W) float array (probabilities or raw values)
        gt_mask: (H, W) float array (binary ground truth)
        threshold: binarization threshold for pred_mask

    Returns:
        float in [0, 1].
    """
    pred_bin = (np.asarray(pred_mask) >= threshold).astype(bool)
    gt_bin = (np.asarray(gt_mask) >= threshold).astype(bool)

    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_metrics.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/__init__.py src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add evaluation metrics (PCK, MRE, court IoU, seg IoU)"
```

---

### Task 4: Geometric consistency losses

**Files:**
- Modify: `src/models/losses.py`
- Modify: `src/training/config.py`
- Create: `tests/test_geometric_losses.py`

**Interfaces:**
- Consumes: `get_collinear_groups()`, `CORNER_INDICES`, `COURT_KEYPOINTS_TEMPLATE` from `src/court_geometry`
- Produces: Updated `CourtVisionLoss` with `collinear_weight`, `ratio_weight`, `convex_weight` parameters; loss components dict now includes `"collinear_loss"`, `"ratio_loss"`, `"convex_loss"` keys

- [ ] **Step 1: Write failing tests**

```python
# tests/test_geometric_losses.py
import torch
import pytest
from src.models.losses import CourtVisionLoss
from src.court_geometry import NUM_KEYPOINTS, CORNER_INDICES


def _dummy_pred_targets(batch_size=2, hm_size=16, img_size=64):
    """Create minimal pred/target dicts for loss computation."""
    K = NUM_KEYPOINTS
    pred = {
        "seg_logits": torch.randn(batch_size, 1, img_size, img_size),
        "heatmaps": torch.randn(batch_size, K, hm_size, hm_size),
        "offsets": torch.randn(batch_size, K, 2),
        "visibility": torch.randn(batch_size, K),
    }
    targets = {
        "mask": torch.zeros(batch_size, 1, img_size, img_size),
        "heatmaps": torch.zeros(batch_size, K, hm_size, hm_size),
        "keypoints": torch.rand(batch_size, K, 2),
        "visibility": torch.ones(batch_size, K),
    }
    return pred, targets


class TestCollinearityLoss:
    def test_perfectly_collinear_gives_zero(self):
        """Points in a straight line → collinearity loss = 0."""
        loss_fn = CourtVisionLoss(collinear_weight=1.0)
        pred, targets = _dummy_pred_targets()
        # Make all predicted offsets map to template-like positions (collinear rows)
        # by using keypoints that lie on straight lines
        kps = torch.zeros(2, NUM_KEYPOINTS, 2)
        for row_start in range(0, 30, 5):
            for j in range(5):
                kps[:, row_start + j, 0] = row_start / 30.0
                kps[:, row_start + j, 1] = j / 5.0
        targets["keypoints"] = kps
        pred["offsets"] = kps  # perfect prediction
        loss, components = loss_fn(pred, targets)
        assert components["collinear_loss"] < 1e-6

    def test_noncollinear_gives_positive_loss(self):
        loss_fn = CourtVisionLoss(collinear_weight=1.0)
        pred, targets = _dummy_pred_targets()
        # Deliberately put K2 off-line from K0-K1-K3-K4
        kps = targets["keypoints"].clone()
        kps[:, 2, :] += 0.5  # push center point far off line
        pred["offsets"] = kps
        loss, components = loss_fn(pred, targets)
        assert components["collinear_loss"] > 0.0


class TestDistanceRatioLoss:
    def test_correct_ratios_give_zero(self):
        loss_fn = CourtVisionLoss(ratio_weight=1.0)
        pred, targets = _dummy_pred_targets()
        from src.court_geometry import COURT_KEYPOINTS_TEMPLATE
        import numpy as np
        tpl_norm = torch.from_numpy(
            COURT_KEYPOINTS_TEMPLATE / np.array([13.4, 6.1])
        ).float()
        tpl_batch = tpl_norm.unsqueeze(0).expand(2, -1, -1)
        targets["keypoints"] = tpl_batch
        pred["offsets"] = tpl_batch
        loss, components = loss_fn(pred, targets)
        assert components["ratio_loss"] < 1e-4

    def test_wrong_ratios_give_positive_loss(self):
        loss_fn = CourtVisionLoss(ratio_weight=1.0)
        pred, targets = _dummy_pred_targets()
        # Random keypoints will have wrong distance ratios
        pred["offsets"] = torch.rand(2, NUM_KEYPOINTS, 2)
        loss, components = loss_fn(pred, targets)
        assert components["ratio_loss"] > 0.0


class TestConvexityLoss:
    def test_convex_quad_gives_zero(self):
        loss_fn = CourtVisionLoss(convex_weight=1.0)
        pred, targets = _dummy_pred_targets()
        kps = torch.zeros(2, NUM_KEYPOINTS, 2)
        # K0=TL, K25=TR, K29=BR, K4=BL — convex
        kps[:, 0] = torch.tensor([0.1, 0.1])
        kps[:, 25] = torch.tensor([0.9, 0.1])
        kps[:, 29] = torch.tensor([0.9, 0.9])
        kps[:, 4] = torch.tensor([0.1, 0.9])
        pred["offsets"] = kps
        targets["visibility"] = torch.ones(2, NUM_KEYPOINTS)
        loss, components = loss_fn(pred, targets)
        assert components["convex_loss"] < 1e-6

    def test_bowtie_gives_positive_loss(self):
        loss_fn = CourtVisionLoss(convex_weight=1.0)
        pred, targets = _dummy_pred_targets()
        kps = torch.zeros(2, NUM_KEYPOINTS, 2)
        # Swap K25 and K4 to create a bowtie (non-convex)
        kps[:, 0] = torch.tensor([0.1, 0.1])
        kps[:, 25] = torch.tensor([0.1, 0.9])   # was TR, now BL
        kps[:, 29] = torch.tensor([0.9, 0.9])
        kps[:, 4] = torch.tensor([0.9, 0.1])    # was BL, now TR
        pred["offsets"] = kps
        targets["visibility"] = torch.ones(2, NUM_KEYPOINTS)
        loss, components = loss_fn(pred, targets)
        assert components["convex_loss"] > 0.0

    def test_missing_corner_skips_loss(self):
        loss_fn = CourtVisionLoss(convex_weight=1.0)
        pred, targets = _dummy_pred_targets()
        targets["visibility"] = torch.zeros(2, NUM_KEYPOINTS)  # no corners visible
        loss, components = loss_fn(pred, targets)
        assert components["convex_loss"] == 0.0


class TestGeometricLossIntegration:
    def test_total_loss_includes_geometric_terms(self):
        loss_fn = CourtVisionLoss(
            collinear_weight=0.1, ratio_weight=0.1, convex_weight=0.1
        )
        pred, targets = _dummy_pred_targets()
        loss, components = loss_fn(pred, targets)
        assert "collinear_loss" in components
        assert "ratio_loss" in components
        assert "convex_loss" in components
        assert loss.requires_grad

    def test_geometric_weights_zero_disables(self):
        loss_fn = CourtVisionLoss(
            collinear_weight=0.0, ratio_weight=0.0, convex_weight=0.0
        )
        pred, targets = _dummy_pred_targets()
        loss_with, _ = loss_fn(pred, targets)

        loss_fn_base = CourtVisionLoss()
        loss_base, _ = loss_fn_base(pred, targets)

        # With all geometric weights at 0, should equal the base loss
        assert abs(loss_with.item() - loss_base.item()) < 1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_geometric_losses.py -v`
Expected: FAIL (missing constructor args or missing component keys)

- [ ] **Step 3: Update TrainConfig**

In `src/training/config.py`, add after `vis_weight`:

```python
    collinear_weight: float = 0.1
    ratio_weight: float = 0.1
    convex_weight: float = 0.1
```

- [ ] **Step 4: Implement geometric losses in CourtVisionLoss**

Replace `src/models/losses.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.court_geometry import (
    CORNER_INDICES,
    COURT_KEYPOINTS_TEMPLATE,
    get_collinear_groups,
)
import numpy as np


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation logits."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = torch.sigmoid(pred).reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


# Precompute template distance ratios for the ratio loss.
_TPL = COURT_KEYPOINTS_TEMPLATE
_COLLINEAR_GROUPS = get_collinear_groups()

# For each collinear group, precompute consecutive distance ratios from template.
# ratio_i = dist(group[0], group[i+1]) / dist(group[0], group[-1])
_RATIO_SPECS = []  # list of (group_indices, expected_ratios)
for group in _COLLINEAR_GROUPS:
    if len(group) < 3:
        continue
    dists = [np.linalg.norm(_TPL[group[i]] - _TPL[group[0]]) for i in range(len(group))]
    total = dists[-1]
    if total < 1e-9:
        continue
    ratios = [d / total for d in dists[1:-1]]  # exclude 0 and 1
    _RATIO_SPECS.append((group, ratios))


class CourtVisionLoss(nn.Module):
    """Combined multi-task loss for CourtVisionNet.

    Components:
        seg_loss: BCE + Dice for segmentation
        heatmap_loss: MSE for keypoint heatmaps (visible only)
        offset_loss: L1 for keypoint offset regression (visible only)
        visibility_loss: BCE for visibility classification
        collinear_loss: cross-product penalty for non-collinear points
        ratio_loss: MSE on distance ratios vs template
        convex_loss: ReLU penalty on non-convex outer corners
    """

    def __init__(
        self,
        seg_weight=1.0,
        heatmap_weight=5.0,
        offset_weight=1.0,
        vis_weight=1.0,
        collinear_weight=0.0,
        ratio_weight=0.0,
        convex_weight=0.0,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.vis_weight = vis_weight
        self.collinear_weight = collinear_weight
        self.ratio_weight = ratio_weight
        self.convex_weight = convex_weight
        self.dice_loss = DiceLoss()

        self.register_buffer(
            "_corner_idx",
            torch.tensor(CORNER_INDICES, dtype=torch.long),
        )

    def forward(self, pred, targets):
        # Segmentation loss: BCE + Dice
        seg_bce = F.binary_cross_entropy_with_logits(
            pred["seg_logits"], targets["mask"]
        )
        seg_dice = self.dice_loss(pred["seg_logits"], targets["mask"])
        seg_loss = seg_bce + seg_dice

        vis = targets["visibility"]  # (B, K)
        num_visible = vis.sum().clamp(min=1.0)

        # Heatmap loss (MSE), masked to visible keypoints only.
        vis_mask_hm = vis.unsqueeze(-1).unsqueeze(-1)
        heatmap_diff = (pred["heatmaps"] - targets["heatmaps"]) ** 2
        heatmap_h, heatmap_w = pred["heatmaps"].shape[2], pred["heatmaps"].shape[3]
        heatmap_loss = (heatmap_diff * vis_mask_hm).sum() / (
            num_visible * heatmap_h * heatmap_w
        )

        # Offset loss (L1), masked to visible keypoints only.
        vis_mask_offset = vis.unsqueeze(-1)
        offset_diff = torch.abs(pred["offsets"] - targets["keypoints"]) * vis_mask_offset
        offset_loss = offset_diff.sum() / (num_visible * 2)

        # Visibility loss (BCE).
        visibility_loss = F.binary_cross_entropy_with_logits(
            pred["visibility"], targets["visibility"]
        )

        # Geometric consistency losses — computed on pred["offsets"] (B, K, 2).
        kps = pred["offsets"]  # (B, K, 2)

        collinear_loss = self._collinear_loss(kps, vis)
        ratio_loss = self._ratio_loss(kps, vis)
        convex_loss = self._convex_loss(kps, vis)

        total = (
            self.seg_weight * seg_loss
            + self.heatmap_weight * heatmap_loss
            + self.offset_weight * offset_loss
            + self.vis_weight * visibility_loss
            + self.collinear_weight * collinear_loss
            + self.ratio_weight * ratio_loss
            + self.convex_weight * convex_loss
        )

        components = {
            "seg_loss": seg_loss.item(),
            "heatmap_loss": heatmap_loss.item(),
            "offset_loss": offset_loss.item(),
            "visibility_loss": visibility_loss.item(),
            "collinear_loss": collinear_loss.item(),
            "ratio_loss": ratio_loss.item(),
            "convex_loss": convex_loss.item(),
        }
        return total, components

    @staticmethod
    def _collinear_loss(kps, vis):
        """Cross-product area penalty for non-collinear keypoints."""
        total = torch.tensor(0.0, device=kps.device, dtype=kps.dtype)
        count = 0
        for group in _COLLINEAR_GROUPS:
            for b in range(kps.shape[0]):
                visible_in_group = [i for i in group if vis[b, i] > 0.5]
                if len(visible_in_group) < 3:
                    continue
                for t in range(len(visible_in_group) - 2):
                    i, j, k = visible_in_group[t], visible_in_group[t + 1], visible_in_group[t + 2]
                    p1 = kps[b, i]
                    p2 = kps[b, j]
                    p3 = kps[b, k]
                    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                    total = total + cross ** 2
                    count += 1
        if count == 0:
            return total
        return total / count

    @staticmethod
    def _ratio_loss(kps, vis):
        """MSE on distance ratios along collinear groups vs template ratios."""
        total = torch.tensor(0.0, device=kps.device, dtype=kps.dtype)
        count = 0
        for group, expected_ratios in _RATIO_SPECS:
            for b in range(kps.shape[0]):
                if not all(vis[b, i] > 0.5 for i in group):
                    continue
                p_first = kps[b, group[0]]
                p_last = kps[b, group[-1]]
                total_dist = torch.norm(p_last - p_first).clamp(min=1e-8)
                for idx, expected in enumerate(expected_ratios):
                    p_mid = kps[b, group[idx + 1]]
                    dist = torch.norm(p_mid - p_first)
                    ratio = dist / total_dist
                    total = total + (ratio - expected) ** 2
                    count += 1
        if count == 0:
            return total
        return total / count

    @staticmethod
    def _convex_loss(kps, vis):
        """ReLU penalty on non-convex outer corner quadrilateral."""
        corner_idx = CORNER_INDICES  # [0, 25, 29, 4] cyclic
        total = torch.tensor(0.0, device=kps.device, dtype=kps.dtype)
        count = 0
        for b in range(kps.shape[0]):
            if not all(vis[b, i] > 0.5 for i in corner_idx):
                continue
            corners = torch.stack([kps[b, i] for i in corner_idx])  # (4, 2)
            for c in range(4):
                p1 = corners[c]
                p2 = corners[(c + 1) % 4]
                p3 = corners[(c + 2) % 4]
                cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
                # All cross products should be positive (CCW) — penalize negative
                total = total + F.relu(-cross)
            count += 1
        if count == 0:
            return total
        return total / count
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_geometric_losses.py tests/test_training.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/losses.py src/training/config.py tests/test_geometric_losses.py
git commit -m "feat: add geometric consistency losses (collinear, ratio, convex)"
```

---

### Task 5: Standalone evaluation script + training validation integration

**Files:**
- Create: `src/evaluation/evaluate.py`
- Modify: `src/training/train.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `pck_at_k`, `mean_reprojection_error`, `court_iou`, `segmentation_iou` from Task 3; `CourtPredictor` from `src/inference/predict`; `CourtDataset` from `src/training/dataset`
- Produces: `evaluate(config) -> dict` function; `validate()` now returns `(avg_loss, avg_components, avg_metrics)` with `avg_metrics` containing `"pck_at_10"` and `"mre"` keys

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluate.py
import json
import os

import cv2
import numpy as np
import pytest
import torch

from src.court_geometry import NUM_KEYPOINTS
from src.training.train import validate


def _create_test_dataset(tmp_path, n_images=2):
    """Create a minimal dataset for evaluation testing."""
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    for i in range(n_images):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img_path = str(img_dir / f"frame_{i:04d}.jpg")
        cv2.imwrite(img_path, img)

        kps = np.random.rand(NUM_KEYPOINTS, 2).tolist()
        vis = [1] * 4 + [0] * (NUM_KEYPOINTS - 4)

        ann = {
            "image_path": img_path,
            "keypoints": kps,
            "visibility": vis,
            "court_class": 1,
        }
        with open(ann_dir / f"frame_{i:04d}.json", "w") as f:
            json.dump(ann, f)

    return str(ann_dir), str(img_dir)


def test_validate_returns_metrics(tmp_path):
    """validate() should return loss, components, AND metrics."""
    from torch.utils.data import DataLoader
    from src.models.courtvisionnet import CourtVisionNet
    from src.models.losses import CourtVisionLoss
    from src.preprocessing.augmentation import get_val_transforms
    from src.training.dataset import CourtDataset

    ann_dir, img_dir = _create_test_dataset(tmp_path)
    ds = CourtDataset(ann_dir, img_dir, transform=get_val_transforms(64), image_size=64, heatmap_size=16)
    loader = DataLoader(ds, batch_size=2)

    model = CourtVisionNet(in_channels=7, image_size=64, heatmap_size=16, pretrained=False)
    loss_fn = CourtVisionLoss()

    avg_loss, avg_components, avg_metrics = validate(model, loader, loss_fn, device="cpu")

    assert isinstance(avg_loss, float)
    assert "seg_loss" in avg_components
    assert "pck_at_10" in avg_metrics
    assert "mre" in avg_metrics
    assert 0.0 <= avg_metrics["pck_at_10"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL (validate returns 2 values, not 3)

- [ ] **Step 3: Update validate() in train.py**

Modify `src/training/train.py`'s `validate()` to also compute PCK@10 and MRE:

```python
@torch.no_grad()
def validate(model, dataloader, loss_fn, device="cuda"):
    """Run validation. Returns (avg_loss, avg_components, avg_metrics)."""
    from src.evaluation.metrics import pck_at_k, mean_reprojection_error

    model.eval()
    total_loss = 0.0
    total_components = {}
    count = 0
    all_pck = []
    all_mre = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images = batch["image"].to(device)
        targets = {
            "mask": batch["mask"].to(device),
            "heatmaps": batch["heatmaps"].to(device),
            "keypoints": batch["keypoints"].to(device),
            "visibility": batch["visibility"].to(device),
        }

        pred = model(images)
        loss, components = loss_fn(pred, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size
        for k, v in components.items():
            total_components[k] = total_components.get(k, 0.0) + v * batch_size

        # Per-sample metrics
        pred_kps = pred["offsets"].cpu().numpy()
        gt_kps = targets["keypoints"].cpu().numpy()
        gt_vis = targets["visibility"].cpu().numpy()

        for i in range(batch_size):
            _, pck_mean = pck_at_k(pred_kps[i], gt_kps[i], gt_vis[i], k=10)
            all_pck.append(pck_mean)
            mre = mean_reprojection_error(pred_kps[i], gt_kps[i], gt_vis[i], 640, 640)
            if mre is not None:
                all_mre.append(mre)

    avg_loss = total_loss / max(count, 1)
    avg_components = {k: v / max(count, 1) for k, v in total_components.items()}
    avg_metrics = {
        "pck_at_10": float(np.mean(all_pck)) if all_pck else 0.0,
        "mre": float(np.mean(all_mre)) if all_mre else 0.0,
    }
    return avg_loss, avg_components, avg_metrics
```

Add `import numpy as np` to the top of `src/training/train.py`.

Update the `train()` function to unpack the new 3-tuple from `validate()`:

```python
# In train(), change:
#   val_loss, val_components = validate(...)
# to:
        val_loss, val_components, val_metrics = validate(model, val_loader, loss_fn, device)
```

And print the metrics:

```python
        print(f"  PCK@10: {val_metrics['pck_at_10']:.4f}")
        if val_metrics['mre'] > 0:
            print(f"  MRE:    {val_metrics['mre']:.2f} px")
```

- [ ] **Step 4: Create standalone evaluate.py**

```python
# src/evaluation/evaluate.py
"""Standalone evaluation CLI for CourtVisionNet."""

import argparse
import json
import os

import cv2
import numpy as np

from src.court_geometry import NUM_KEYPOINTS, KEYPOINT_NAMES, CORNER_INDICES
from src.evaluation.metrics import (
    court_iou,
    mean_reprojection_error,
    pck_at_k,
    segmentation_iou,
)
from src.inference.predict import CourtPredictor
from src.preprocessing.channels import generate_channels
from src.court_geometry import generate_line_mask


def evaluate(checkpoint_path, annotations_dir, images_dir, image_size=640, device="cpu"):
    """Run full evaluation on a test set.

    Returns a dict with summary metrics.
    """
    predictor = CourtPredictor(checkpoint_path, device=device, image_size=image_size)

    ann_paths = sorted(
        [os.path.join(annotations_dir, f) for f in os.listdir(annotations_dir) if f.endswith(".json")]
    )

    results = {
        "pck_5": [], "pck_10": [], "pck_20": [],
        "mre": [], "court_iou": [], "seg_iou": [],
        "per_keypoint_pck_10": [[] for _ in range(NUM_KEYPOINTS)],
    }

    for ann_path in ann_paths:
        with open(ann_path) as f:
            ann = json.load(f)

        img_path = ann["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(images_dir, os.path.basename(img_path))
        image = cv2.imread(img_path)
        if image is None:
            continue

        gt_kps = np.array(ann["keypoints"], dtype=np.float32)
        gt_vis = np.array(ann["visibility"], dtype=np.float32)

        detection = predictor.predict(image)
        pred_kps = detection.keypoints.astype(np.float32)
        pred_vis = detection.visibility.astype(np.float32)

        h, w = image.shape[:2]

        for label, k in [("pck_5", 5), ("pck_10", 10), ("pck_20", 20)]:
            per_kp, mean_acc = pck_at_k(pred_kps, gt_kps, gt_vis, k=k, image_size=image_size)
            results[label].append(mean_acc)
            if label == "pck_10":
                for i in range(NUM_KEYPOINTS):
                    if gt_vis[i] > 0.5:
                        results["per_keypoint_pck_10"][i].append(float(per_kp[i]))

        mre = mean_reprojection_error(pred_kps, gt_kps, gt_vis, w, h)
        if mre is not None:
            results["mre"].append(mre)

        iou = court_iou(pred_kps, gt_kps, pred_vis, gt_vis)
        results["court_iou"].append(iou)

        gt_mask = generate_line_mask(gt_kps, gt_vis.astype(int).tolist(), w, h).astype(np.float32) / 255.0
        pred_mask = cv2.resize(detection.seg_mask.astype(np.float32), (w, h))
        results["seg_iou"].append(segmentation_iou(pred_mask, gt_mask))

    summary = {
        "n_images": len(ann_paths),
        "pck_5_mean": float(np.mean(results["pck_5"])) if results["pck_5"] else 0.0,
        "pck_10_mean": float(np.mean(results["pck_10"])) if results["pck_10"] else 0.0,
        "pck_20_mean": float(np.mean(results["pck_20"])) if results["pck_20"] else 0.0,
        "mre_mean": float(np.mean(results["mre"])) if results["mre"] else 0.0,
        "mre_std": float(np.std(results["mre"])) if results["mre"] else 0.0,
        "court_iou_mean": float(np.mean(results["court_iou"])) if results["court_iou"] else 0.0,
        "seg_iou_mean": float(np.mean(results["seg_iou"])) if results["seg_iou"] else 0.0,
    }

    per_kp_pck = {}
    for i in range(NUM_KEYPOINTS):
        vals = results["per_keypoint_pck_10"][i]
        per_kp_pck[KEYPOINT_NAMES[i]] = float(np.mean(vals)) if vals else 0.0
    summary["per_keypoint_pck_10"] = per_kp_pck

    return summary


def print_summary(summary):
    """Pretty-print evaluation summary."""
    print(f"\n{'='*40}")
    print(f" CourtVisionNet Evaluation")
    print(f"{'='*40}")
    print(f" Images:     {summary['n_images']}")
    print(f" PCK@5:      {summary['pck_5_mean']:.1%}")
    print(f" PCK@10:     {summary['pck_10_mean']:.1%}")
    print(f" PCK@20:     {summary['pck_20_mean']:.1%}")
    print(f" MRE:        {summary['mre_mean']:.2f} px (±{summary['mre_std']:.2f})")
    print(f" Court IoU:  {summary['court_iou_mean']:.3f}")
    print(f" Seg IoU:    {summary['seg_iou_mean']:.3f}")
    print(f"\n Per-keypoint PCK@10:")
    for name, val in summary["per_keypoint_pck_10"].items():
        print(f"   {name:30s} {val:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CourtVisionNet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = evaluate(
        args.checkpoint, args.annotations, args.images,
        image_size=args.image_size, device=args.device,
    )
    print_summary(summary)
```

- [ ] **Step 5: Fix existing tests that unpack validate()**

The existing `test_training.py` calls `validate()` and unpacks 2 values. Update every call from:
```python
val_loss, val_components = validate(...)
```
to:
```python
val_loss, val_components, val_metrics = validate(...)
```

Search for all occurrences in `tests/test_training.py` and update them.

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/evaluation/evaluate.py src/training/train.py tests/test_evaluate.py tests/test_training.py
git commit -m "feat: standalone eval script + metrics in validation loop"
```

---

### Task 6: Final integration — update README, run full suite, push

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Everything from Tasks 1–5
- Produces: Updated README, all tests passing, pushed to GitHub

- [ ] **Step 1: Update README**

Replace the "31 Keypoints" section with:

```markdown
### 30 Keypoints

All line intersections on a standard doubles badminton court (6 rows × 5 columns):

| Row | Dbl-top | Sgl-top | Center | Sgl-bot | Dbl-bot |
|-----|:---:|:---:|:---:|:---:|:---:|
| Baseline L | K0 | K1 | K2 | K3 | K4 |
| Long Svc L | K5 | K6 | K7 | K8 | K9 |
| Short Svc L | K10 | K11 | K12 | K13 | K14 |
| Short Svc R | K15 | K16 | K17 | K18 | K19 |
| Long Svc R | K20 | K21 | K22 | K23 | K24 |
| Baseline R | K25 | K26 | K27 | K28 | K29 |

Outer corners: K0, K4, K25, K29.
```

Update the overview to say "30 court keypoints".

Add an "Evaluation" section:

```markdown
## Evaluation

```bash
python -m src.evaluation.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --annotations data/annotations/test \
    --images data/frames/
```

Metrics: PCK@5/10/20, Mean Reprojection Error, Court IoU, Segmentation IoU.
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit and push**

```bash
git add -A
git commit -m "docs: update README for 30 keypoints and evaluation"
git push
```

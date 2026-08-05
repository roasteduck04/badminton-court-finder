# CourtVisionNet Phase 1: Annotation Tool, Preprocessing, and Core Model

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the annotation tool to generate training data, the preprocessing pipeline, the dataset loader, and the core dual-head model (segmentation + keypoints) so training can begin on Colab.

**Architecture:** A Python project using PyTorch for the model, Tkinter+OpenCV for annotation, and a 7-channel preprocessing pipeline feeding a ResNet-50+FPN backbone with dual segmentation and keypoint heads. Training will run on Google Colab.

**Tech Stack:** Python 3.10+, PyTorch 2.x, torchvision, OpenCV, NumPy, Tkinter, Pillow, albumentations, matplotlib

## Global Constraints

- Python 3.10+ (Colab compatible)
- PyTorch 2.x with CUDA support
- All coordinates normalized to [0, 1] throughout the pipeline
- Input resolution: 640×640 for model
- 14 keypoints as defined in the spec (K0-K13)
- Invisible keypoints stored as [-1, -1] with visibility flag 0
- Annotation export format: JSON per frame (schema in spec)
- Segmentation masks: binary PNG, same resolution as input frame
- No hardcoded paths — use config/argparse for all paths
- Standard badminton court dimensions: 13.4m × 6.1m (doubles), 13.4m × 5.18m (singles)

---

### Task 1: Project Scaffolding and Court Geometry Constants

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/court_geometry.py`
- Create: `tests/__init__.py`
- Create: `tests/test_court_geometry.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: Nothing (first task)
- Produces:
  - `COURT_KEYPOINTS_TEMPLATE: np.ndarray` — shape (14, 2), real-world coordinates of all 14 keypoints in meters
  - `get_court_lines(keypoints: np.ndarray) -> list[tuple[int, int]]` — returns list of (start_kp_idx, end_kp_idx) pairs defining all court lines
  - `compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray` — compute 3×3 homography from ≥4 point correspondences
  - `project_points(H: np.ndarray, points: np.ndarray) -> np.ndarray` — apply homography to project points
  - `validate_quadrilateral(corners: np.ndarray) -> bool` — check if 4 corners form a valid convex quadrilateral

- [ ] **Step 1: Create .gitignore**

```
__pycache__/
*.pyc
*.pyo
.env
*.egg-info/
dist/
build/
data/frames/
data/masks/
*.pt
*.pth
*.onnx
.ipynb_checkpoints/
wandb/
```

- [ ] **Step 2: Create requirements.txt**

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
albumentations>=1.3.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

- [ ] **Step 3: Write failing tests for court geometry**

```python
# tests/test_court_geometry.py
import numpy as np
import pytest
from src.court_geometry import (
    COURT_KEYPOINTS_TEMPLATE,
    COURT_WIDTH_DOUBLES,
    COURT_LENGTH,
    get_court_lines,
    compute_homography,
    project_points,
    validate_quadrilateral,
    generate_line_mask,
)


def test_template_has_14_keypoints():
    assert COURT_KEYPOINTS_TEMPLATE.shape == (14, 2)


def test_template_dimensions_match_spec():
    # K0 is top-left (0,0), K2 is bottom-right (13.4, 6.1)
    k0 = COURT_KEYPOINTS_TEMPLATE[0]
    k2 = COURT_KEYPOINTS_TEMPLATE[2]
    length = abs(k2[0] - k0[0])
    width = abs(k2[1] - k0[1])
    assert abs(length - 13.4) < 0.01
    assert abs(width - 6.1) < 0.01


def test_court_lines_returns_pairs():
    lines = get_court_lines()
    assert len(lines) > 0
    for start_idx, end_idx in lines:
        assert 0 <= start_idx < 14
        assert 0 <= end_idx < 14


def test_compute_homography_identity():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    H = compute_homography(pts, pts)
    assert H.shape == (3, 3)
    # Should be close to identity
    np.testing.assert_allclose(H / H[2, 2], np.eye(3), atol=1e-6)


def test_project_points_roundtrip():
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    dst = np.array([[10, 10], [200, 20], [190, 180], [15, 170]], dtype=np.float64)
    H = compute_homography(src, dst)
    projected = project_points(H, src)
    np.testing.assert_allclose(projected, dst, atol=1.0)


def test_validate_quadrilateral_valid():
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    assert validate_quadrilateral(corners) is True


def test_validate_quadrilateral_invalid_crossed():
    corners = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=np.float64)
    assert validate_quadrilateral(corners) is False


def test_generate_line_mask():
    keypoints = np.array([
        [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
        [-1, -1], [-1, -1], [-1, -1], [-1, -1],
        [-1, -1], [-1, -1], [-1, -1], [-1, -1],
        [-1, -1], [-1, -1],
    ], dtype=np.float64)
    visibility = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = generate_line_mask(keypoints, visibility, width=640, height=640, line_thickness=3)
    assert mask.shape == (640, 640)
    assert mask.dtype == np.uint8
    assert mask.max() == 255
    assert mask.min() == 0
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_court_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.court_geometry'`

- [ ] **Step 5: Implement court geometry module**

```python
# src/__init__.py
# (empty)

# src/court_geometry.py
import numpy as np
import cv2

COURT_LENGTH = 13.4       # meters
COURT_WIDTH_DOUBLES = 6.1  # meters
COURT_WIDTH_SINGLES = 5.18 # meters
SHORT_SERVICE_LINE = 1.98  # meters from net
NET_POSITION = 6.7         # meters from each end (center)
LONG_SERVICE_LINE_DOUBLES = 0.76  # meters from back boundary

SINGLES_SIDELINE_OFFSET = (COURT_WIDTH_DOUBLES - COURT_WIDTH_SINGLES) / 2  # 0.46m

# 14 keypoints in real-world coordinates (meters)
# Origin at top-left corner (K0), x along length, y along width
COURT_KEYPOINTS_TEMPLATE = np.array([
    [0.0, 0.0],                                          # K0:  top-left outer
    [COURT_LENGTH, 0.0],                                 # K1:  top-right outer
    [COURT_LENGTH, COURT_WIDTH_DOUBLES],                  # K2:  bottom-right outer
    [0.0, COURT_WIDTH_DOUBLES],                           # K3:  bottom-left outer
    [NET_POSITION - SHORT_SERVICE_LINE, 0.0],             # K4:  left short service / top
    [NET_POSITION - SHORT_SERVICE_LINE, COURT_WIDTH_DOUBLES],  # K5:  left short service / bottom
    [NET_POSITION + SHORT_SERVICE_LINE, 0.0],             # K6:  right short service / top
    [NET_POSITION + SHORT_SERVICE_LINE, COURT_WIDTH_DOUBLES],  # K7:  right short service / bottom
    [NET_POSITION, 0.0],                                  # K8:  net / top
    [NET_POSITION, COURT_WIDTH_DOUBLES],                  # K9:  net / bottom
    [NET_POSITION - SHORT_SERVICE_LINE, COURT_WIDTH_DOUBLES / 2],  # K10: left short service / center
    [NET_POSITION + SHORT_SERVICE_LINE, COURT_WIDTH_DOUBLES / 2],  # K11: right short service / center
    [NET_POSITION, SINGLES_SIDELINE_OFFSET],              # K12: net / top singles
    [NET_POSITION, COURT_WIDTH_DOUBLES - SINGLES_SIDELINE_OFFSET],  # K13: net / bottom singles
], dtype=np.float64)


def get_court_lines():
    """Return list of (start_keypoint_idx, end_keypoint_idx) pairs defining all court lines."""
    return [
        # Outer boundary
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Short service lines
        (4, 5), (6, 7),
        # Net/center line
        (8, 9),
        # Center service line
        (10, 11),
        # Singles sidelines (partial — from short service to short service)
        (12, 12),  # These are points on the net line, connected via:
        # Long service lines for doubles (not full lines, but segments)
        # For simplicity, we define the major structural lines:
        # Left half center line
        (4, 10), (10, 5),
        # Right half center line
        (6, 11), (11, 7),
    ]


def compute_homography(src_points, dst_points):
    """Compute 3x3 homography from source to destination points.

    Args:
        src_points: (N, 2) array of source points, N >= 4
        dst_points: (N, 2) array of destination points, N >= 4

    Returns:
        3x3 homography matrix
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, method=0)
    return H


def project_points(H, points):
    """Apply homography to project points.

    Args:
        H: 3x3 homography matrix
        points: (N, 2) array of points

    Returns:
        (N, 2) array of projected points
    """
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])  # (N, 3)
    projected_h = (H @ pts_h.T).T   # (N, 3)
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    return projected


def validate_quadrilateral(corners):
    """Check if 4 corners form a valid convex quadrilateral.

    Args:
        corners: (4, 2) array of corner points in order

    Returns:
        True if valid convex quadrilateral
    """
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2):
        return False

    # Check convexity via cross products of consecutive edges
    n = len(corners)
    sign = None
    for i in range(n):
        p1 = corners[i]
        p2 = corners[(i + 1) % n]
        p3 = corners[(i + 2) % n]
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        if sign is None:
            sign = cross > 0
        elif (cross > 0) != sign:
            return False
    return True


def generate_line_mask(keypoints, visibility, width=640, height=640, line_thickness=3):
    """Generate a binary mask of court lines from keypoint annotations.

    Args:
        keypoints: (14, 2) array of normalized [0,1] keypoint coordinates
        visibility: list of 14 ints (1=visible, 0=not visible)
        width: mask width in pixels
        height: mask height in pixels
        line_thickness: line width in pixels

    Returns:
        (height, width) uint8 mask, 255 for line pixels, 0 elsewhere
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    kps = np.asarray(keypoints, dtype=np.float64).copy()

    # Convert normalized coords to pixel coords
    pixel_kps = np.zeros_like(kps)
    pixel_kps[:, 0] = kps[:, 0] * width
    pixel_kps[:, 1] = kps[:, 1] * height

    lines = get_court_lines()
    for start_idx, end_idx in lines:
        if start_idx == end_idx:
            continue
        if visibility[start_idx] and visibility[end_idx]:
            pt1 = tuple(pixel_kps[start_idx].astype(int))
            pt2 = tuple(pixel_kps[end_idx].astype(int))
            cv2.line(mask, pt1, pt2, 255, line_thickness)

    # Draw outer boundary if all 4 corners visible
    if all(visibility[i] for i in range(4)):
        for i in range(4):
            pt1 = tuple(pixel_kps[i].astype(int))
            pt2 = tuple(pixel_kps[(i + 1) % 4].astype(int))
            cv2.line(mask, pt1, pt2, 255, line_thickness)

    return mask
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_court_geometry.py -v`
Expected: All 8 tests PASS

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt src/ tests/
git commit -m "feat: project scaffolding and court geometry constants"
```

---

### Task 2: Video Frame Extraction Tool

**Files:**
- Create: `src/tools/extract_frames.py`
- Create: `tests/test_extract_frames.py`

**Interfaces:**
- Consumes: Nothing (standalone tool)
- Produces:
  - `extract_frames(video_path: str, output_dir: str, fps: float = 1.0, max_frames: int | None = None) -> list[str]` — extracts frames from video, returns list of saved frame paths

- [ ] **Step 1: Write failing tests**

```python
# tests/test_extract_frames.py
import os
import tempfile
import numpy as np
import cv2
import pytest
from src.tools.extract_frames import extract_frames


@pytest.fixture
def sample_video(tmp_path):
    """Create a tiny synthetic video for testing."""
    video_path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30, (320, 240))
    for i in range(90):  # 3 seconds at 30fps
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (i * 2, 100, 200)
        writer.write(frame)
    writer.release()
    return video_path


def test_extract_frames_basic(sample_video, tmp_path):
    output_dir = str(tmp_path / "frames")
    paths = extract_frames(sample_video, output_dir, fps=1.0)
    assert len(paths) == 3  # 3 seconds of video at 1 fps
    for p in paths:
        assert os.path.exists(p)
        img = cv2.imread(p)
        assert img is not None


def test_extract_frames_max_frames(sample_video, tmp_path):
    output_dir = str(tmp_path / "frames2")
    paths = extract_frames(sample_video, output_dir, fps=1.0, max_frames=2)
    assert len(paths) == 2


def test_extract_frames_higher_fps(sample_video, tmp_path):
    output_dir = str(tmp_path / "frames3")
    paths = extract_frames(sample_video, output_dir, fps=10.0)
    assert len(paths) >= 20  # ~30 frames at 10fps from 3s video
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extract_frames.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement frame extraction**

```python
# src/tools/__init__.py
# (empty)

# src/tools/extract_frames.py
import os
import cv2


def extract_frames(video_path, output_dir, fps=1.0, max_frames=None):
    """Extract frames from a video at the specified FPS.

    Args:
        video_path: path to the video file
        output_dir: directory to save extracted frames
        fps: frames per second to extract
        max_frames: maximum number of frames to extract (None = all)

    Returns:
        list of saved frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(video_fps / fps)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    saved_paths = []
    frame_idx = 0
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{video_name}_frame_{save_count:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_paths.append(filepath)
            save_count += 1

            if max_frames is not None and save_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("video_path", help="Path to video file or directory of videos")
    parser.add_argument("output_dir", help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per video")
    args = parser.parse_args()

    if os.path.isdir(args.video_path):
        for fname in sorted(os.listdir(args.video_path)):
            if fname.lower().endswith((".mov", ".mp4", ".avi", ".mkv")):
                vpath = os.path.join(args.video_path, fname)
                vname = os.path.splitext(fname)[0]
                out = os.path.join(args.output_dir, vname)
                paths = extract_frames(vpath, out, fps=args.fps, max_frames=args.max_frames)
                print(f"{fname}: extracted {len(paths)} frames")
    else:
        paths = extract_frames(args.video_path, args.output_dir, fps=args.fps, max_frames=args.max_frames)
        print(f"Extracted {len(paths)} frames")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extract_frames.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/ tests/test_extract_frames.py
git commit -m "feat: video frame extraction tool"
```

---

### Task 3: Annotation Tool

**Files:**
- Create: `src/tools/annotator.py`
- Create: `tests/test_annotator_logic.py`

**Interfaces:**
- Consumes:
  - `src/court_geometry.py`: `COURT_KEYPOINTS_TEMPLATE`, `get_court_lines()`, `compute_homography()`, `project_points()`, `validate_quadrilateral()`, `generate_line_mask()`
- Produces:
  - `AnnotationState` class — manages keypoints, visibility, undo/redo for a single frame
  - `save_annotation(state: AnnotationState, image_path: str, output_path: str) -> dict` — saves annotation as JSON
  - `load_annotation(path: str) -> AnnotationState` — loads annotation from JSON
  - GUI application launched via `python -m src.tools.annotator`

Note: We test the annotation logic (state management, save/load, geometry validation) but not the Tkinter GUI itself.

- [ ] **Step 1: Write failing tests for annotation state logic**

```python
# tests/test_annotator_logic.py
import json
import numpy as np
import pytest
from src.tools.annotator import AnnotationState, save_annotation, load_annotation


def test_annotation_state_init():
    state = AnnotationState()
    assert len(state.keypoints) == 14
    assert all(v == 0 for v in state.visibility)
    assert state.court_class == 1  # doubles default


def test_set_keypoint():
    state = AnnotationState()
    state.set_keypoint(0, 0.5, 0.3)
    assert state.keypoints[0] == [0.5, 0.3]
    assert state.visibility[0] == 1


def test_clear_keypoint():
    state = AnnotationState()
    state.set_keypoint(0, 0.5, 0.3)
    state.clear_keypoint(0)
    assert state.keypoints[0] == [-1, -1]
    assert state.visibility[0] == 0


def test_undo_redo():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.2)
    state.set_keypoint(1, 0.9, 0.2)
    assert state.visibility[1] == 1

    state.undo()
    assert state.visibility[1] == 0
    assert state.visibility[0] == 1

    state.redo()
    assert state.visibility[1] == 1


def test_visible_count():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(1, 0.9, 0.1)
    state.set_keypoint(2, 0.9, 0.9)
    assert state.visible_count() == 3


def test_save_and_load(tmp_path):
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(1, 0.9, 0.1)
    state.set_keypoint(2, 0.9, 0.9)
    state.set_keypoint(3, 0.1, 0.9)
    state.court_class = 0  # singles

    out_path = str(tmp_path / "annotation.json")
    result = save_annotation(state, "test_frame.jpg", out_path)

    assert result["image_path"] == "test_frame.jpg"
    assert result["court_class"] == 0
    assert len(result["keypoints"]) == 14
    assert len(result["visibility"]) == 14
    assert result["visibility"][0] == 1
    assert result["visibility"][4] == 0

    loaded = load_annotation(out_path)
    assert loaded.keypoints[0] == [0.1, 0.1]
    assert loaded.visibility[0] == 1
    assert loaded.court_class == 0


def test_get_bounding_box():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(1, 0.9, 0.1)
    state.set_keypoint(2, 0.9, 0.9)
    state.set_keypoint(3, 0.1, 0.9)
    cx, cy, w, h = state.get_bounding_box()
    assert abs(cx - 0.5) < 0.01
    assert abs(cy - 0.5) < 0.01
    assert abs(w - 0.8) < 0.01
    assert abs(h - 0.8) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_annotator_logic.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement AnnotationState and save/load logic**

```python
# src/tools/annotator.py
import copy
import json
import os
import numpy as np


class AnnotationState:
    """Manages keypoint annotations for a single frame."""

    def __init__(self):
        self.keypoints = [[-1.0, -1.0] for _ in range(14)]
        self.visibility = [0] * 14
        self.court_class = 1  # 0=singles, 1=doubles, 2=alternative
        self._history = []
        self._redo_stack = []

    def _save_snapshot(self):
        self._history.append({
            "keypoints": copy.deepcopy(self.keypoints),
            "visibility": list(self.visibility),
        })
        self._redo_stack.clear()

    def set_keypoint(self, idx, x, y):
        self._save_snapshot()
        self.keypoints[idx] = [float(x), float(y)]
        self.visibility[idx] = 1

    def clear_keypoint(self, idx):
        self._save_snapshot()
        self.keypoints[idx] = [-1.0, -1.0]
        self.visibility[idx] = 0

    def undo(self):
        if not self._history:
            return
        self._redo_stack.append({
            "keypoints": copy.deepcopy(self.keypoints),
            "visibility": list(self.visibility),
        })
        snapshot = self._history.pop()
        self.keypoints = snapshot["keypoints"]
        self.visibility = snapshot["visibility"]

    def redo(self):
        if not self._redo_stack:
            return
        self._history.append({
            "keypoints": copy.deepcopy(self.keypoints),
            "visibility": list(self.visibility),
        })
        snapshot = self._redo_stack.pop()
        self.keypoints = snapshot["keypoints"]
        self.visibility = snapshot["visibility"]

    def visible_count(self):
        return sum(self.visibility)

    def get_bounding_box(self):
        """Return (cx, cy, w, h) normalized bounding box of visible keypoints."""
        visible_pts = [
            self.keypoints[i] for i in range(14) if self.visibility[i]
        ]
        if not visible_pts:
            return (0.0, 0.0, 0.0, 0.0)
        pts = np.array(visible_pts)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        return (float(cx), float(cy), float(w), float(h))

    def to_dict(self, image_path=""):
        cx, cy, w, h = self.get_bounding_box()
        return {
            "image_path": image_path,
            "image_size": [640, 640],
            "court_class": self.court_class,
            "keypoints": [list(kp) for kp in self.keypoints],
            "visibility": list(self.visibility),
            "bounding_box": [cx, cy, w, h],
        }

    @classmethod
    def from_dict(cls, data):
        state = cls()
        state.keypoints = [list(kp) for kp in data["keypoints"]]
        state.visibility = list(data["visibility"])
        state.court_class = data.get("court_class", 1)
        return state


def save_annotation(state, image_path, output_path):
    data = state.to_dict(image_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def load_annotation(path):
    with open(path, "r") as f:
        data = json.load(f)
    return AnnotationState.from_dict(data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_annotator_logic.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Implement the Tkinter GUI**

Create the full GUI in the same file, below the logic code. The GUI allows:
- Loading a directory of frame images
- Clicking to place keypoints (K0-K13) — current keypoint selected via number keys or side panel
- Dragging to adjust existing keypoints
- Showing the projected court overlay when ≥4 keypoints are placed
- Court class selector (singles/doubles)
- Navigation between frames (prev/next)
- Save/load annotations as JSON
- Generate and save segmentation masks
- Keyboard shortcuts: 0-9 for keypoint selection, z=undo, y=redo, s=save, n=next, p=prev

This is a large GUI component — implement it as a `CourtAnnotator` class with Tkinter. The full implementation should be ~300-400 lines covering the window layout, canvas interaction, and file management.

- [ ] **Step 6: Manual test — launch annotator on extracted frames**

Run: `python -m src.tools.annotator --input-dir data/frames/IMG_3751`
Expected: Tkinter window opens showing the first frame, ready for annotation

- [ ] **Step 7: Commit**

```bash
git add src/tools/annotator.py tests/test_annotator_logic.py
git commit -m "feat: annotation tool with keypoint management and GUI"
```

---

### Task 4: Multi-Channel Preprocessing Pipeline

**Files:**
- Create: `src/preprocessing/channels.py`
- Create: `src/preprocessing/augmentation.py`
- Create: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: Nothing (standalone module)
- Produces:
  - `generate_channels(image: np.ndarray) -> np.ndarray` — takes BGR image (H,W,3), returns (H,W,7) multi-channel tensor
  - `get_train_transforms(image_size: int = 640) -> albumentations.Compose` — training augmentation pipeline
  - `get_val_transforms(image_size: int = 640) -> albumentations.Compose` — validation transforms (resize + normalize only)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preprocessing.py
import numpy as np
import cv2
import pytest
from src.preprocessing.channels import generate_channels
from src.preprocessing.augmentation import get_train_transforms, get_val_transforms


def test_generate_channels_shape():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = generate_channels(img)
    assert result.shape == (480, 640, 7)
    assert result.dtype == np.float32


def test_generate_channels_rgb_preserved():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = generate_channels(img)
    # First 3 channels should be RGB (converted from BGR), normalized to [0,1]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    np.testing.assert_allclose(result[:, :, :3], rgb, atol=1e-5)


def test_generate_channels_grayscale():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = generate_channels(img)
    gray_expected = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    np.testing.assert_allclose(result[:, :, 3], gray_expected, atol=1e-5)


def test_generate_channels_range():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = generate_channels(img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_train_transforms_output_shape():
    transform = get_train_transforms(image_size=640)
    img = np.random.randint(0, 255, (480, 640, 7), dtype=np.uint8)
    # albumentations expects keypoints in a specific format
    result = transform(image=img)
    assert result["image"].shape[:2] == (640, 640)


def test_val_transforms_deterministic():
    transform = get_val_transforms(image_size=640)
    img = np.random.randint(0, 255, (480, 640, 7), dtype=np.uint8)
    r1 = transform(image=img)["image"]
    r2 = transform(image=img)["image"]
    np.testing.assert_array_equal(r1, r2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_preprocessing.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement channel generation**

```python
# src/preprocessing/__init__.py
# (empty)

# src/preprocessing/channels.py
import cv2
import numpy as np


def generate_channels(image):
    """Generate 7-channel representation from a BGR image.

    Channels:
        0-2: RGB (normalized to [0,1])
        3:   Grayscale (normalized to [0,1])
        4:   CLAHE enhanced (normalized to [0,1])
        5:   Canny edge map (normalized to [0,1])
        6:   Court-color mask — HSV filter for green court surface (normalized to [0,1])

    Args:
        image: (H, W, 3) BGR uint8 image

    Returns:
        (H, W, 7) float32 array, all values in [0, 1]
    """
    h, w = image.shape[:2]
    result = np.zeros((h, w, 7), dtype=np.float32)

    # Ch 0-2: RGB normalized
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    result[:, :, :3] = rgb

    # Ch 3: Grayscale normalized
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result[:, :, 3] = gray.astype(np.float32) / 255.0

    # Ch 4: CLAHE enhanced
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    result[:, :, 4] = clahe_img.astype(np.float32) / 255.0

    # Ch 5: Canny edge map
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 50, 150)
    result[:, :, 5] = edges.astype(np.float32) / 255.0

    # Ch 6: Court-color mask (green surface in HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 60])
    upper_green = np.array([90, 255, 255])
    court_mask = cv2.inRange(hsv, lower_green, upper_green)
    court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    result[:, :, 6] = court_mask.astype(np.float32) / 255.0

    return result
```

- [ ] **Step 4: Implement augmentation pipeline**

```python
# src/preprocessing/augmentation.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transforms(image_size=640):
    """Training augmentation pipeline for 7-channel input with keypoints.

    Works with albumentations' keypoint format.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.5),
        A.Perspective(scale=(0.05, 0.15), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.CoarseDropout(
            max_holes=4, max_height=80, max_width=80,
            min_holes=1, min_height=20, min_width=20,
            fill_value=0, p=0.3
        ),
    ], keypoint_params=A.KeypointParams(
        format="xy", remove_invisible=False, angle_in_degrees=True
    ))


def get_val_transforms(image_size=640):
    """Validation/test transforms — deterministic resize only."""
    return A.Compose([
        A.Resize(image_size, image_size),
    ], keypoint_params=A.KeypointParams(
        format="xy", remove_invisible=False, angle_in_degrees=True
    ))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocessing.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Manual test — visualize channels on a real frame**

```bash
python -c "
import cv2, numpy as np, matplotlib.pyplot as plt
from src.preprocessing.channels import generate_channels
img = cv2.imread('data/frames/IMG_3751/IMG_3751_frame_00000.jpg')
if img is None:
    print('Extract frames first')
else:
    ch = generate_channels(img)
    names = ['R','G','B','Gray','CLAHE','Canny','CourtMask']
    fig, axes = plt.subplots(1, 7, figsize=(21, 3))
    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.imshow(ch[:,:,i], cmap='gray' if i >= 3 else None)
        ax.set_title(name); ax.axis('off')
    plt.tight_layout(); plt.savefig('channel_preview.png'); plt.show()
"
```
Expected: 7 sub-images showing each channel. Canny should highlight court lines, CourtMask should highlight the green surface.

- [ ] **Step 7: Commit**

```bash
git add src/preprocessing/ tests/test_preprocessing.py
git commit -m "feat: 7-channel preprocessing and augmentation pipeline"
```

---

### Task 5: PyTorch Dataset Class

**Files:**
- Create: `src/training/dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes:
  - `src/preprocessing/channels.py`: `generate_channels()`
  - `src/preprocessing/augmentation.py`: `get_train_transforms()`, `get_val_transforms()`
  - `src/court_geometry.py`: `generate_line_mask()`
  - Annotation JSON files (produced by Task 3)
- Produces:
  - `CourtDataset(annotations_dir, images_dir, transform, image_size)` — PyTorch Dataset
  - `__getitem__` returns dict with keys: `image` (7,H,W tensor), `heatmaps` (14,H/4,W/4 tensor), `keypoints` (14,2 tensor), `visibility` (14, tensor), `mask` (1,H,W tensor)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset.py
import json
import os
import numpy as np
import cv2
import torch
import pytest
from src.training.dataset import CourtDataset


@pytest.fixture
def mock_dataset(tmp_path):
    """Create a small mock dataset with 3 annotated frames."""
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    for i in range(3):
        # Create a fake image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = str(img_dir / f"frame_{i:03d}.jpg")
        cv2.imwrite(img_path, img)

        # Create annotation
        ann = {
            "image_path": img_path,
            "image_size": [640, 640],
            "court_class": 1,
            "keypoints": [
                [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
                [0.3, 0.1], [0.3, 0.9], [0.7, 0.1], [0.7, 0.9],
                [0.5, 0.1], [0.5, 0.9], [0.3, 0.5], [0.7, 0.5],
                [0.5, 0.2], [0.5, 0.8],
            ],
            "visibility": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "bounding_box": [0.5, 0.5, 0.8, 0.8],
        }
        with open(str(ann_dir / f"frame_{i:03d}.json"), "w") as f:
            json.dump(ann, f)

    return str(img_dir), str(ann_dir)


def test_dataset_length(mock_dataset):
    img_dir, ann_dir = mock_dataset
    ds = CourtDataset(ann_dir, img_dir, image_size=640)
    assert len(ds) == 3


def test_dataset_item_shapes(mock_dataset):
    img_dir, ann_dir = mock_dataset
    ds = CourtDataset(ann_dir, img_dir, image_size=640)
    sample = ds[0]

    assert sample["image"].shape == (7, 640, 640)
    assert sample["image"].dtype == torch.float32
    assert sample["heatmaps"].shape == (14, 160, 160)
    assert sample["keypoints"].shape == (14, 2)
    assert sample["visibility"].shape == (14,)
    assert sample["mask"].shape == (1, 640, 640)


def test_dataset_invisible_keypoints(tmp_path):
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "frame.jpg"), img)

    ann = {
        "image_path": str(img_dir / "frame.jpg"),
        "image_size": [640, 640],
        "court_class": 1,
        "keypoints": [
            [0.1, 0.1], [0.9, 0.1], [-1, -1], [-1, -1],
            [-1, -1], [-1, -1], [-1, -1], [-1, -1],
            [-1, -1], [-1, -1], [-1, -1], [-1, -1],
            [-1, -1], [-1, -1],
        ],
        "visibility": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "bounding_box": [0.5, 0.1, 0.8, 0.0],
    }
    with open(str(ann_dir / "frame.json"), "w") as f:
        json.dump(ann, f)

    ds = CourtDataset(str(ann_dir), str(img_dir), image_size=640)
    sample = ds[0]

    # Invisible keypoints should have zero heatmaps
    assert sample["visibility"][2] == 0
    assert sample["visibility"][0] == 1
    assert sample["heatmaps"][2].sum() == 0  # no heatmap for invisible
    assert sample["heatmaps"][0].sum() > 0   # heatmap present for visible
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement dataset class**

```python
# src/training/__init__.py
# (empty)

# src/training/dataset.py
import json
import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from src.preprocessing.channels import generate_channels
from src.court_geometry import generate_line_mask


class CourtDataset(Dataset):
    """PyTorch dataset for badminton court detection.

    Each sample returns:
        image: (7, H, W) float32 tensor — 7-channel preprocessed input
        heatmaps: (14, H/4, W/4) float32 tensor — Gaussian heatmaps per keypoint
        keypoints: (14, 2) float32 tensor — normalized [0,1] keypoint coordinates
        visibility: (14,) float32 tensor — 1.0 if visible, 0.0 if not
        mask: (1, H, W) float32 tensor — binary court line segmentation mask
    """

    def __init__(self, annotations_dir, images_dir, transform=None,
                 image_size=640, heatmap_size=160, sigma=3.0):
        self.images_dir = images_dir
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.transform = transform

        self.annotations = []
        for ann_path in sorted(glob.glob(os.path.join(annotations_dir, "*.json"))):
            with open(ann_path) as f:
                self.annotations.append(json.load(f))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # Load image
        img_path = ann["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.images_dir, os.path.basename(img_path))
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Resize to target
        image = cv2.resize(image, (self.image_size, self.image_size))

        keypoints = np.array(ann["keypoints"], dtype=np.float32)
        visibility = np.array(ann["visibility"], dtype=np.float32)

        # Generate 7-channel input
        channels = generate_channels(image)  # (H, W, 7)

        # Apply augmentation if provided
        if self.transform is not None:
            visible_kps = []
            for i in range(14):
                if visibility[i] > 0:
                    x_px = keypoints[i, 0] * self.image_size
                    y_px = keypoints[i, 1] * self.image_size
                    visible_kps.append((x_px, y_px))
                else:
                    visible_kps.append((0.0, 0.0))

            transformed = self.transform(
                image=(channels * 255).astype(np.uint8),
                keypoints=visible_kps,
            )
            channels = transformed["image"].astype(np.float32) / 255.0
            new_kps = transformed["keypoints"]
            for i in range(14):
                if visibility[i] > 0:
                    kx, ky = new_kps[i]
                    keypoints[i, 0] = kx / self.image_size
                    keypoints[i, 1] = ky / self.image_size
                    if not (0 <= keypoints[i, 0] <= 1 and 0 <= keypoints[i, 1] <= 1):
                        visibility[i] = 0
                        keypoints[i] = [-1, -1]

        # Generate heatmaps
        heatmaps = self._generate_heatmaps(keypoints, visibility)

        # Generate segmentation mask
        mask = generate_line_mask(
            keypoints, visibility.astype(int).tolist(),
            width=self.image_size, height=self.image_size, line_thickness=3
        )
        mask = mask.astype(np.float32) / 255.0

        # Convert to tensors (channels-first)
        image_tensor = torch.from_numpy(channels.transpose(2, 0, 1))  # (7, H, W)
        heatmap_tensor = torch.from_numpy(heatmaps)  # (14, hm_h, hm_w)
        keypoint_tensor = torch.from_numpy(keypoints)  # (14, 2)
        visibility_tensor = torch.from_numpy(visibility)  # (14,)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return {
            "image": image_tensor,
            "heatmaps": heatmap_tensor,
            "keypoints": keypoint_tensor,
            "visibility": visibility_tensor,
            "mask": mask_tensor,
        }

    def _generate_heatmaps(self, keypoints, visibility):
        """Generate Gaussian heatmaps for each keypoint."""
        hm_h, hm_w = self.heatmap_size, self.heatmap_size
        heatmaps = np.zeros((14, hm_h, hm_w), dtype=np.float32)

        for i in range(14):
            if visibility[i] < 1:
                continue
            cx = keypoints[i, 0] * hm_w
            cy = keypoints[i, 1] * hm_h

            x = np.arange(hm_w, dtype=np.float32)
            y = np.arange(hm_h, dtype=np.float32)
            xx, yy = np.meshgrid(x, y)

            heatmaps[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.sigma ** 2))

        return heatmaps
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/ tests/test_dataset.py
git commit -m "feat: PyTorch dataset class with heatmap and mask generation"
```

---

### Task 6: ResNet-50 + FPN Backbone (7-channel input)

**Files:**
- Create: `src/models/backbone.py`
- Create: `tests/test_backbone.py`

**Interfaces:**
- Consumes: Nothing (uses torchvision internally)
- Produces:
  - `CourtBackbone(in_channels: int = 7, pretrained: bool = True)` — nn.Module
  - Forward returns dict of feature maps: `{"p2": (B,256,H/4,W/4), "p3": (B,256,H/8,W/8), "p4": (B,256,H/16,W/16), "p5": (B,256,H/32,W/32)}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_backbone.py
import torch
import pytest
from src.models.backbone import CourtBackbone


def test_backbone_output_shapes():
    model = CourtBackbone(in_channels=7, pretrained=False)
    x = torch.randn(2, 7, 640, 640)
    features = model(x)

    assert "p2" in features
    assert "p3" in features
    assert "p4" in features
    assert "p5" in features

    assert features["p2"].shape == (2, 256, 160, 160)
    assert features["p3"].shape == (2, 256, 80, 80)
    assert features["p4"].shape == (2, 256, 40, 40)
    assert features["p5"].shape == (2, 256, 20, 20)


def test_backbone_3_channel_input():
    model = CourtBackbone(in_channels=3, pretrained=False)
    x = torch.randn(2, 3, 640, 640)
    features = model(x)
    assert features["p2"].shape == (2, 256, 160, 160)


def test_backbone_gradient_flow():
    model = CourtBackbone(in_channels=7, pretrained=False)
    x = torch.randn(1, 7, 256, 256, requires_grad=True)
    features = model(x)
    loss = sum(f.sum() for f in features.values())
    loss.backward()
    assert x.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backbone.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement backbone**

```python
# src/models/__init__.py
# (empty)

# src/models/backbone.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork


class CourtBackbone(nn.Module):
    """ResNet-50 + FPN backbone with configurable input channels.

    When in_channels > 3 and pretrained=True, the first conv layer is expanded:
    RGB channels get pretrained weights, extra channels get He-initialized weights.
    """

    def __init__(self, in_channels=7, pretrained=True):
        super().__init__()

        weights = "IMAGENET1K_V2" if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Modify first conv for multi-channel input
        original_conv = resnet.conv1
        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                # Copy pretrained weights for RGB channels
                with torch.no_grad():
                    new_conv.weight[:, :3] = original_conv.weight
                    # He-initialize extra channels
                    nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode="fan_out", nonlinearity="relu")
            resnet.conv1 = new_conv

        # Extract layer stages
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # stride 4,  256 channels
        self.layer2 = resnet.layer2  # stride 8,  512 channels
        self.layer3 = resnet.layer3  # stride 16, 1024 channels
        self.layer4 = resnet.layer4  # stride 32, 2048 channels

        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
        )

    def forward(self, x):
        c1 = self.stem(x)      # stride 4
        c2 = self.layer1(c1)   # stride 4,  256 ch
        c3 = self.layer2(c2)   # stride 8,  512 ch
        c4 = self.layer3(c3)   # stride 16, 1024 ch
        c5 = self.layer4(c4)   # stride 32, 2048 ch

        fpn_input = {
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
        }
        fpn_output = self.fpn(fpn_input)

        return {
            "p2": fpn_output["c2"],  # (B, 256, H/4, W/4)
            "p3": fpn_output["c3"],  # (B, 256, H/8, W/8)
            "p4": fpn_output["c4"],  # (B, 256, H/16, W/16)
            "p5": fpn_output["c5"],  # (B, 256, H/32, W/32)
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backbone.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/ tests/test_backbone.py
git commit -m "feat: ResNet-50 + FPN backbone with 7-channel input support"
```

---

### Task 7: Segmentation and Keypoint Heads

**Files:**
- Create: `src/models/segmentation_head.py`
- Create: `src/models/keypoint_head.py`
- Create: `tests/test_heads.py`

**Interfaces:**
- Consumes:
  - `CourtBackbone` output: dict with `p2`, `p3`, `p4`, `p5` feature maps
- Produces:
  - `SegmentationHead(in_channels=256, image_size=640)` — nn.Module, forward returns `(B, 1, 640, 640)` logits
  - `KeypointHead(in_channels=256, num_keypoints=14, heatmap_size=160)` — nn.Module, forward returns dict: `{"heatmaps": (B,14,160,160), "offsets": (B,14,2), "visibility": (B,14)}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_heads.py
import torch
import pytest
from src.models.segmentation_head import SegmentationHead
from src.models.keypoint_head import KeypointHead


def make_fake_features(batch_size=2):
    return {
        "p2": torch.randn(batch_size, 256, 160, 160),
        "p3": torch.randn(batch_size, 256, 80, 80),
        "p4": torch.randn(batch_size, 256, 40, 40),
        "p5": torch.randn(batch_size, 256, 20, 20),
    }


def test_segmentation_head_output_shape():
    head = SegmentationHead(in_channels=256, image_size=640)
    features = make_fake_features()
    out = head(features)
    assert out.shape == (2, 1, 640, 640)


def test_keypoint_head_output_shapes():
    head = KeypointHead(in_channels=256, num_keypoints=14, heatmap_size=160)
    features = make_fake_features()
    out = head(features)
    assert out["heatmaps"].shape == (2, 14, 160, 160)
    assert out["offsets"].shape == (2, 14, 2)
    assert out["visibility"].shape == (2, 14)


def test_segmentation_head_gradient():
    head = SegmentationHead(in_channels=256, image_size=640)
    features = make_fake_features()
    features["p2"].requires_grad_(True)
    out = head(features)
    out.sum().backward()
    assert features["p2"].grad is not None


def test_keypoint_head_gradient():
    head = KeypointHead(in_channels=256, num_keypoints=14, heatmap_size=160)
    features = make_fake_features()
    features["p2"].requires_grad_(True)
    out = head(features)
    loss = out["heatmaps"].sum() + out["offsets"].sum() + out["visibility"].sum()
    loss.backward()
    assert features["p2"].grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_heads.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement segmentation head**

```python
# src/models/segmentation_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """Lightweight FPN-based decoder for court line segmentation.

    Fuses multi-scale features and produces a full-resolution binary mask.
    """

    def __init__(self, in_channels=256, image_size=640):
        super().__init__()
        self.image_size = image_size

        # Per-level refinement
        self.refine_p2 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.refine_p3 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.refine_p4 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.refine_p5 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Fusion after concatenation (128 * 4 = 512 channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, features):
        p2 = self.refine_p2(features["p2"])  # (B, 128, H/4, W/4)

        target_size = p2.shape[2:]
        p3 = F.interpolate(self.refine_p3(features["p3"]), size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.refine_p4(features["p4"]), size=target_size, mode="bilinear", align_corners=False)
        p5 = F.interpolate(self.refine_p5(features["p5"]), size=target_size, mode="bilinear", align_corners=False)

        fused = torch.cat([p2, p3, p4, p5], dim=1)  # (B, 512, H/4, W/4)
        out = self.fuse(fused)  # (B, 1, H/4, W/4)

        out = F.interpolate(out, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return out  # (B, 1, H, W) logits
```

- [ ] **Step 4: Implement keypoint head**

```python
# src/models/keypoint_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointHead(nn.Module):
    """Keypoint detection head with heatmaps, offset regression, and visibility.

    Produces:
        heatmaps: (B, num_keypoints, heatmap_size, heatmap_size)
        offsets: (B, num_keypoints, 2) — sub-pixel refinement
        visibility: (B, num_keypoints) — probability each keypoint is visible
    """

    def __init__(self, in_channels=256, num_keypoints=14, heatmap_size=160):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

        # Heatmap branch — operates on P2 (highest resolution)
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1),
        )

        # Offset regression branch — global average pool then predict
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_keypoints * 2),
        )

        # Visibility branch — global features to per-keypoint visibility
        self.visibility_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_keypoints),
        )

    def forward(self, features):
        p2 = features["p2"]  # (B, 256, H/4, W/4)

        heatmaps = self.heatmap_conv(p2)  # (B, 14, H/4, W/4)

        # Ensure heatmap is at target resolution
        if heatmaps.shape[2] != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )

        offsets = self.offset_conv(p2)  # (B, 28)
        offsets = offsets.view(-1, self.num_keypoints, 2)  # (B, 14, 2)

        visibility = self.visibility_conv(p2)  # (B, 14)

        return {
            "heatmaps": heatmaps,
            "offsets": offsets,
            "visibility": visibility,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_heads.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/segmentation_head.py src/models/keypoint_head.py tests/test_heads.py
git commit -m "feat: segmentation and keypoint detection heads"
```

---

### Task 8: Full CourtVisionNet Model Assembly + Loss Functions

**Files:**
- Create: `src/models/courtvisionnet.py`
- Create: `src/models/losses.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes:
  - `CourtBackbone` from `src/models/backbone.py`
  - `SegmentationHead` from `src/models/segmentation_head.py`
  - `KeypointHead` from `src/models/keypoint_head.py`
- Produces:
  - `CourtVisionNet(in_channels=7, num_keypoints=14, pretrained=True)` — full model, forward returns dict with all head outputs
  - `CourtVisionLoss()` — combined loss function, forward takes model output + targets, returns scalar loss + loss components dict

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model.py
import torch
import pytest
from src.models.courtvisionnet import CourtVisionNet
from src.models.losses import CourtVisionLoss


def test_model_forward():
    model = CourtVisionNet(in_channels=7, pretrained=False)
    x = torch.randn(2, 7, 640, 640)
    out = model(x)

    assert "seg_logits" in out
    assert "heatmaps" in out
    assert "offsets" in out
    assert "visibility" in out

    assert out["seg_logits"].shape == (2, 1, 640, 640)
    assert out["heatmaps"].shape == (2, 14, 160, 160)
    assert out["offsets"].shape == (2, 14, 2)
    assert out["visibility"].shape == (2, 14)


def test_model_parameter_count():
    model = CourtVisionNet(in_channels=7, pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    # Should be roughly 28-32M params
    assert 20_000_000 < total < 40_000_000


def test_loss_function():
    loss_fn = CourtVisionLoss()

    pred = {
        "seg_logits": torch.randn(2, 1, 640, 640),
        "heatmaps": torch.randn(2, 14, 160, 160),
        "offsets": torch.randn(2, 14, 2),
        "visibility": torch.randn(2, 14),
    }
    targets = {
        "mask": torch.randint(0, 2, (2, 1, 640, 640)).float(),
        "heatmaps": torch.randn(2, 14, 160, 160),
        "keypoints": torch.rand(2, 14, 2),
        "visibility": torch.randint(0, 2, (2, 14)).float(),
    }

    total_loss, components = loss_fn(pred, targets)
    assert total_loss.requires_grad
    assert "seg_loss" in components
    assert "heatmap_loss" in components
    assert "offset_loss" in components
    assert "visibility_loss" in components


def test_loss_zero_visible():
    """Loss should handle the case where no keypoints are visible."""
    loss_fn = CourtVisionLoss()
    pred = {
        "seg_logits": torch.randn(1, 1, 640, 640),
        "heatmaps": torch.randn(1, 14, 160, 160),
        "offsets": torch.randn(1, 14, 2),
        "visibility": torch.randn(1, 14),
    }
    targets = {
        "mask": torch.zeros(1, 1, 640, 640),
        "heatmaps": torch.zeros(1, 14, 160, 160),
        "keypoints": torch.full((1, 14, 2), -1.0),
        "visibility": torch.zeros(1, 14),
    }
    total_loss, _ = loss_fn(pred, targets)
    assert not torch.isnan(total_loss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CourtVisionNet**

```python
# src/models/courtvisionnet.py
import torch.nn as nn
from src.models.backbone import CourtBackbone
from src.models.segmentation_head import SegmentationHead
from src.models.keypoint_head import KeypointHead


class CourtVisionNet(nn.Module):
    """Full CourtVisionNet: shared backbone + dual segmentation/keypoint heads."""

    def __init__(self, in_channels=7, num_keypoints=14, image_size=640,
                 heatmap_size=160, pretrained=True):
        super().__init__()
        self.backbone = CourtBackbone(in_channels=in_channels, pretrained=pretrained)
        self.seg_head = SegmentationHead(in_channels=256, image_size=image_size)
        self.kpt_head = KeypointHead(
            in_channels=256, num_keypoints=num_keypoints, heatmap_size=heatmap_size
        )

    def forward(self, x):
        features = self.backbone(x)
        seg_logits = self.seg_head(features)
        kpt_out = self.kpt_head(features)

        return {
            "seg_logits": seg_logits,
            "heatmaps": kpt_out["heatmaps"],
            "offsets": kpt_out["offsets"],
            "visibility": kpt_out["visibility"],
        }
```

- [ ] **Step 4: Implement loss functions**

```python
# src/models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = torch.sigmoid(pred).view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class CourtVisionLoss(nn.Module):
    """Combined loss for CourtVisionNet.

    Components:
        seg_loss: BCE + Dice for segmentation
        heatmap_loss: MSE for keypoint heatmaps (visible only)
        offset_loss: L1 for keypoint offset regression (visible only)
        visibility_loss: BCE for visibility classification
    """

    def __init__(self, seg_weight=1.0, heatmap_weight=1.0,
                 offset_weight=1.0, vis_weight=1.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.vis_weight = vis_weight
        self.dice_loss = DiceLoss()

    def forward(self, pred, targets):
        # Segmentation loss
        seg_bce = F.binary_cross_entropy_with_logits(
            pred["seg_logits"], targets["mask"]
        )
        seg_dice = self.dice_loss(pred["seg_logits"], targets["mask"])
        seg_loss = seg_bce + seg_dice

        # Visibility mask for keypoint losses
        vis = targets["visibility"]  # (B, 14)
        vis_mask = vis.unsqueeze(-1).unsqueeze(-1)  # (B, 14, 1, 1) for heatmaps

        # Heatmap loss (only for visible keypoints)
        heatmap_diff = (pred["heatmaps"] - targets["heatmaps"]) ** 2
        masked_heatmap = heatmap_diff * vis_mask
        num_visible = vis.sum().clamp(min=1)
        heatmap_loss = masked_heatmap.sum() / (num_visible * pred["heatmaps"].shape[2] * pred["heatmaps"].shape[3])

        # Offset loss (only for visible keypoints)
        vis_mask_2d = vis.unsqueeze(-1)  # (B, 14, 1)
        offset_diff = torch.abs(pred["offsets"] - targets["keypoints"]) * vis_mask_2d
        offset_loss = offset_diff.sum() / (num_visible * 2)

        # Visibility loss
        vis_loss = F.binary_cross_entropy_with_logits(
            pred["visibility"], targets["visibility"]
        )

        total = (
            self.seg_weight * seg_loss
            + self.heatmap_weight * heatmap_loss
            + self.offset_weight * offset_loss
            + self.vis_weight * vis_loss
        )

        components = {
            "seg_loss": seg_loss.item(),
            "heatmap_loss": heatmap_loss.item(),
            "offset_loss": offset_loss.item(),
            "visibility_loss": vis_loss.item(),
        }
        return total, components
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_model.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/courtvisionnet.py src/models/losses.py tests/test_model.py
git commit -m "feat: CourtVisionNet full model assembly and loss functions"
```

---

### Task 9: Training Loop and Colab Notebook

**Files:**
- Create: `src/training/train.py`
- Create: `src/training/config.py`
- Create: `tests/test_training.py`
- Create: `notebooks/train_colab.ipynb`

**Interfaces:**
- Consumes:
  - `CourtVisionNet` from `src/models/courtvisionnet.py`
  - `CourtVisionLoss` from `src/models/losses.py`
  - `CourtDataset` from `src/training/dataset.py`
  - `get_train_transforms()`, `get_val_transforms()` from `src/preprocessing/augmentation.py`
- Produces:
  - `train(config: TrainConfig) -> dict` — trains the model, saves checkpoints, returns metrics
  - `TrainConfig` dataclass — all hyperparameters
  - Colab notebook ready to upload and run

- [ ] **Step 1: Write failing tests**

```python
# tests/test_training.py
import json
import os
import numpy as np
import cv2
import torch
import pytest
from src.training.config import TrainConfig
from src.training.train import train_one_epoch, validate


@pytest.fixture
def mock_data(tmp_path):
    """Create minimal mock dataset for training tests."""
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    for i in range(4):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"f{i}.jpg"), img)
        ann = {
            "image_path": str(img_dir / f"f{i}.jpg"),
            "image_size": [128, 128],
            "court_class": 1,
            "keypoints": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]] + [[-1, -1]] * 10,
            "visibility": [1, 1, 1, 1] + [0] * 10,
            "bounding_box": [0.5, 0.5, 0.8, 0.8],
        }
        with open(str(ann_dir / f"f{i}.json"), "w") as f:
            json.dump(ann, f)
    return str(img_dir), str(ann_dir)


def test_train_one_epoch(mock_data):
    from src.training.dataset import CourtDataset
    from src.models.courtvisionnet import CourtVisionNet
    from src.models.losses import CourtVisionLoss
    from torch.utils.data import DataLoader

    img_dir, ann_dir = mock_data
    ds = CourtDataset(ann_dir, img_dir, image_size=128)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32, pretrained=False)
    loss_fn = CourtVisionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    avg_loss = train_one_epoch(model, loader, loss_fn, optimizer, device="cpu")
    assert isinstance(avg_loss, float)
    assert avg_loss > 0


def test_validate(mock_data):
    from src.training.dataset import CourtDataset
    from src.models.courtvisionnet import CourtVisionNet
    from src.models.losses import CourtVisionLoss
    from torch.utils.data import DataLoader

    img_dir, ann_dir = mock_data
    ds = CourtDataset(ann_dir, img_dir, image_size=128)
    loader = DataLoader(ds, batch_size=2)

    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32, pretrained=False)
    loss_fn = CourtVisionLoss()

    val_loss, metrics = validate(model, loader, loss_fn, device="cpu")
    assert isinstance(val_loss, float)
    assert "seg_loss" in metrics
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_training.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement config**

```python
# src/training/config.py
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data
    train_annotations: str = "data/annotations/train"
    val_annotations: str = "data/annotations/val"
    train_images: str = "data/frames"
    val_images: str = "data/frames"

    # Model
    in_channels: int = 7
    num_keypoints: int = 14
    image_size: int = 640
    heatmap_size: int = 160
    pretrained: bool = True

    # Training
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    freeze_backbone_epochs: int = 5

    # Loss weights
    seg_weight: float = 1.0
    heatmap_weight: float = 1.0
    offset_weight: float = 1.0
    vis_weight: float = 1.0

    # Output
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
```

- [ ] **Step 4: Implement training loop**

```python
# src/training/train.py
import os
import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, loss_fn, optimizer, device="cuda"):
    model.train()
    total_loss = 0.0
    count = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        targets = {
            "mask": batch["mask"].to(device),
            "heatmaps": batch["heatmaps"].to(device),
            "keypoints": batch["keypoints"].to(device),
            "visibility": batch["visibility"].to(device),
        }

        optimizer.zero_grad()
        pred = model(images)
        loss, _ = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        count += images.size(0)

    return total_loss / max(count, 1)


@torch.no_grad()
def validate(model, dataloader, loss_fn, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_components = {}
    count = 0

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

        total_loss += loss.item() * images.size(0)
        count += images.size(0)
        for k, v in components.items():
            total_components[k] = total_components.get(k, 0.0) + v * images.size(0)

    avg_loss = total_loss / max(count, 1)
    avg_components = {k: v / max(count, 1) for k, v in total_components.items()}
    return avg_loss, avg_components


def train(config):
    from src.training.dataset import CourtDataset
    from src.preprocessing.augmentation import get_train_transforms, get_val_transforms
    from src.models.courtvisionnet import CourtVisionNet
    from src.models.losses import CourtVisionLoss
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset
    train_transform = get_train_transforms(config.image_size)
    val_transform = get_val_transforms(config.image_size)

    train_ds = CourtDataset(
        config.train_annotations, config.train_images,
        transform=train_transform, image_size=config.image_size,
        heatmap_size=config.heatmap_size,
    )
    val_ds = CourtDataset(
        config.val_annotations, config.val_images,
        transform=val_transform, image_size=config.image_size,
        heatmap_size=config.heatmap_size,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = CourtVisionNet(
        in_channels=config.in_channels,
        num_keypoints=config.num_keypoints,
        image_size=config.image_size,
        heatmap_size=config.heatmap_size,
        pretrained=config.pretrained,
    ).to(device)

    loss_fn = CourtVisionLoss(
        seg_weight=config.seg_weight,
        heatmap_weight=config.heatmap_weight,
        offset_weight=config.offset_weight,
        vis_weight=config.vis_weight,
    )

    # Optimizer with backbone freeze
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.num_epochs):
        # Freeze/unfreeze backbone
        freeze = epoch < config.freeze_backbone_epochs
        for param in model.backbone.parameters():
            param.requires_grad = not freeze

        print(f"\nEpoch {epoch + 1}/{config.num_epochs} {'(backbone frozen)' if freeze else ''}")

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_components = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        for k, v in val_components.items():
            print(f"    {k}: {v:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(config.checkpoint_dir, "best_model.pt"))
            print("  Saved best model")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return {"best_val_loss": best_val_loss, "final_epoch": epoch + 1}


if __name__ == "__main__":
    from src.training.config import TrainConfig
    config = TrainConfig()
    train(config)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_training.py -v`
Expected: Both tests PASS

- [ ] **Step 6: Create Colab training notebook**

Create `notebooks/train_colab.ipynb` with cells for:
1. Mount Google Drive + clone repo
2. Install dependencies
3. Upload/link annotated data
4. Configure `TrainConfig`
5. Run training with `train(config)`
6. Visualize loss curves
7. Save best model to Drive

- [ ] **Step 7: Commit**

```bash
git add src/training/config.py src/training/train.py tests/test_training.py notebooks/
git commit -m "feat: training loop, config, and Colab notebook"
```

---

### Task 10: Inference and Visualization

**Files:**
- Create: `src/inference/predict.py`
- Create: `src/inference/visualize.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes:
  - `CourtVisionNet` from `src/models/courtvisionnet.py`
  - `generate_channels()` from `src/preprocessing/channels.py`
  - `COURT_KEYPOINTS_TEMPLATE`, `compute_homography()`, `project_points()` from `src/court_geometry.py`
- Produces:
  - `CourtPredictor(checkpoint_path: str, device: str = "cuda")` — loads model, provides `predict(image: np.ndarray) -> CourtDetection`
  - `CourtDetection` dataclass — `keypoints`, `visibility`, `confidence`, `homography`, `seg_mask`
  - `draw_court_overlay(image: np.ndarray, detection: CourtDetection) -> np.ndarray` — draws detected court on image

- [ ] **Step 1: Write failing tests**

```python
# tests/test_inference.py
import numpy as np
import torch
import pytest
from src.inference.predict import CourtPredictor, CourtDetection
from src.inference.visualize import draw_court_overlay


def test_court_detection_dataclass():
    det = CourtDetection(
        keypoints=np.zeros((14, 2)),
        visibility=np.zeros(14),
        confidence=0.9,
        homography=np.eye(3),
        seg_mask=np.zeros((640, 640)),
    )
    assert det.confidence == 0.9
    assert det.keypoints.shape == (14, 2)


def test_draw_court_overlay():
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    det = CourtDetection(
        keypoints=np.array([
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
            [0.3, 0.1], [0.3, 0.9], [0.7, 0.1], [0.7, 0.9],
            [0.5, 0.1], [0.5, 0.9], [0.3, 0.5], [0.7, 0.5],
            [0.5, 0.2], [0.5, 0.8],
        ]),
        visibility=np.ones(14),
        confidence=0.9,
        homography=np.eye(3),
        seg_mask=np.zeros((640, 640)),
    )
    result = draw_court_overlay(img, det)
    assert result.shape == img.shape
    assert result.dtype == np.uint8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement prediction**

```python
# src/inference/__init__.py
# (empty)

# src/inference/predict.py
from dataclasses import dataclass
import numpy as np
import cv2
import torch
from src.models.courtvisionnet import CourtVisionNet
from src.preprocessing.channels import generate_channels
from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, compute_homography, project_points


@dataclass
class CourtDetection:
    keypoints: np.ndarray     # (14, 2) normalized coordinates
    visibility: np.ndarray    # (14,) visibility scores
    confidence: float         # overall detection confidence
    homography: np.ndarray    # 3x3 homography matrix (or None)
    seg_mask: np.ndarray      # (H, W) segmentation mask


class CourtPredictor:
    def __init__(self, checkpoint_path, device="cuda", image_size=640):
        self.device = device
        self.image_size = image_size

        self.model = CourtVisionNet(in_channels=7, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        """Run inference on a BGR image.

        Args:
            image: (H, W, 3) BGR uint8 image

        Returns:
            CourtDetection with all outputs
        """
        h_orig, w_orig = image.shape[:2]
        resized = cv2.resize(image, (self.image_size, self.image_size))
        channels = generate_channels(resized)
        tensor = torch.from_numpy(channels.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        out = self.model(tensor)

        # Extract keypoints from heatmaps
        heatmaps = out["heatmaps"][0].cpu().numpy()  # (14, hm_h, hm_w)
        offsets = out["offsets"][0].cpu().numpy()      # (14, 2)
        vis_logits = out["visibility"][0].cpu().numpy()  # (14,)
        seg_logits = out["seg_logits"][0, 0].cpu().numpy()  # (H, W)

        vis_probs = 1 / (1 + np.exp(-vis_logits))  # sigmoid
        seg_mask = (1 / (1 + np.exp(-seg_logits))) > 0.5

        # Extract keypoints from heatmap peaks
        keypoints = np.zeros((14, 2), dtype=np.float64)
        for i in range(14):
            hm = heatmaps[i]
            peak_idx = np.unravel_index(hm.argmax(), hm.shape)
            ky, kx = peak_idx
            keypoints[i, 0] = kx / hm.shape[1] + offsets[i, 0]
            keypoints[i, 1] = ky / hm.shape[0] + offsets[i, 1]

        # Estimate homography from visible keypoints
        visible_mask = vis_probs > 0.5
        homography = None
        if visible_mask.sum() >= 4:
            src_pts = COURT_KEYPOINTS_TEMPLATE[visible_mask]
            dst_pts = keypoints[visible_mask]
            dst_pts_px = dst_pts * np.array([w_orig, h_orig])
            try:
                homography = compute_homography(src_pts, dst_pts_px)
                all_projected = project_points(homography, COURT_KEYPOINTS_TEMPLATE)
                all_projected[:, 0] /= w_orig
                all_projected[:, 1] /= h_orig
                for i in range(14):
                    if not visible_mask[i]:
                        keypoints[i] = all_projected[i]
            except Exception:
                pass

        confidence = float(vis_probs[visible_mask].mean()) if visible_mask.any() else 0.0

        return CourtDetection(
            keypoints=keypoints,
            visibility=vis_probs,
            confidence=confidence,
            homography=homography if homography is not None else np.eye(3),
            seg_mask=seg_mask.astype(np.uint8),
        )
```

- [ ] **Step 4: Implement visualization**

```python
# src/inference/visualize.py
import numpy as np
import cv2
from src.court_geometry import get_court_lines

KEYPOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),  # K0-K3 corners
    (255, 128, 0), (128, 255, 0), (0, 128, 255), (128, 0, 255),  # K4-K7
    (255, 0, 128), (0, 255, 128),  # K8-K9
    (200, 200, 0), (0, 200, 200),  # K10-K11
    (200, 0, 200), (100, 100, 100),  # K12-K13
]

KEYPOINT_NAMES = [
    "K0:TL", "K1:TR", "K2:BR", "K3:BL",
    "K4", "K5", "K6", "K7",
    "K8:NetT", "K9:NetB",
    "K10", "K11", "K12", "K13",
]


def draw_court_overlay(image, detection, alpha=0.3):
    """Draw detected court overlay on image.

    Args:
        image: (H, W, 3) BGR uint8 image
        detection: CourtDetection
        alpha: transparency for segmentation mask overlay

    Returns:
        (H, W, 3) BGR uint8 image with overlay
    """
    result = image.copy()
    h, w = image.shape[:2]

    # Draw segmentation mask overlay
    if detection.seg_mask is not None and detection.seg_mask.any():
        mask_resized = cv2.resize(
            detection.seg_mask.astype(np.uint8),
            (w, h), interpolation=cv2.INTER_NEAREST
        )
        overlay = result.copy()
        overlay[mask_resized > 0] = [0, 255, 255]  # yellow for court lines
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

    # Draw court lines between visible keypoints
    lines = get_court_lines()
    for start_idx, end_idx in lines:
        if start_idx == end_idx:
            continue
        if detection.visibility[start_idx] > 0.3 and detection.visibility[end_idx] > 0.3:
            pt1 = (int(detection.keypoints[start_idx, 0] * w),
                    int(detection.keypoints[start_idx, 1] * h))
            pt2 = (int(detection.keypoints[end_idx, 0] * w),
                    int(detection.keypoints[end_idx, 1] * h))
            cv2.line(result, pt1, pt2, (0, 255, 0), 2)

    # Draw outer boundary
    for i in range(4):
        j = (i + 1) % 4
        if detection.visibility[i] > 0.3 and detection.visibility[j] > 0.3:
            pt1 = (int(detection.keypoints[i, 0] * w), int(detection.keypoints[i, 1] * h))
            pt2 = (int(detection.keypoints[j, 0] * w), int(detection.keypoints[j, 1] * h))
            cv2.line(result, pt1, pt2, (0, 255, 0), 2)

    # Draw keypoints
    for i in range(14):
        x = int(detection.keypoints[i, 0] * w)
        y = int(detection.keypoints[i, 1] * h)
        vis = detection.visibility[i]
        color = KEYPOINT_COLORS[i]

        if vis > 0.5:
            cv2.circle(result, (x, y), 6, color, -1)
            cv2.circle(result, (x, y), 6, (255, 255, 255), 1)
        elif vis > 0.3:
            cv2.circle(result, (x, y), 6, color, 1)  # hollow = extrapolated

        cv2.putText(result, KEYPOINT_NAMES[i], (x + 8, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Confidence text
    cv2.putText(result, f"Conf: {detection.confidence:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference.py -v`
Expected: Both tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/inference/ tests/test_inference.py
git commit -m "feat: inference predictor and court overlay visualization"
```

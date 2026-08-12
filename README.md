# CourtVisionNet

Badminton court detection from non-standard camera angles using a hybrid segmentation-keypoint deep learning architecture.

## Overview

CourtVisionNet detects badminton courts in video frames captured from arbitrary camera positions (ground-level, phone recordings, non-broadcast angles). It predicts 30 court keypoints (all line intersections on a standard doubles court) and a segmentation mask, then estimates a homography to map pixel coordinates to real-world court positions — enabling player position tracking and tactical analysis.

### Architecture

- **Backbone**: ResNet-50 + FPN with 7-channel input (RGB + grayscale + CLAHE + Canny edges + court prior mask)
- **Segmentation head**: Multi-scale feature fusion → binary court mask (BCE + Dice loss)
- **Keypoint head**: Heatmap regression for 30 court keypoints + sub-pixel offset refinement + visibility classification
- **Homography**: DLT-based estimation from detected keypoints, mapping image coordinates to a standard court template

### 30 Keypoints

All line intersections on a standard doubles badminton court, organized as a 6×5 grid (6 rows along the court length × 5 columns across the width):

| Row | Keypoints | Description |
|-----|-----------|-------------|
| Baseline (left) | K0–K4 | Left baseline × 5 vertical lines (incl. center) |
| Long service (left) | K5–K9 | Left long service line × 5 vertical lines (incl. center) |
| Short service (left) | K10–K14 | Left short service line × 5 vertical lines (incl. center) |
| Short service (right) | K15–K19 | Right short service line × 5 vertical lines (incl. center) |
| Long service (right) | K20–K24 | Right long service line × 5 vertical lines (incl. center) |
| Baseline (right) | K25–K29 | Right baseline × 5 vertical lines (incl. center) |

Each row has 5 columns: doubles sideline top, singles sideline top, center, singles sideline bot, doubles sideline bot.

Outer corners: K0 (top-left), K4 (bottom-left), K25 (top-right), K29 (bottom-right).

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
    └── annotator.html         # Browser-based annotation tool
tests/                         # 119 unit tests covering all modules
```

## Setup

```bash
pip install -r requirements.txt
```

## Annotation

Two annotation tools are provided for labeling court keypoints on extracted frames.

### Browser Annotator (recommended for collaborators)

Open `src/tools/annotator.html` in Chrome. No installation needed.

**Workflow:**
1. Click **Load Folder** to load extracted frames
2. Click a keypoint on the court diagram (right panel) to select it
3. Click where that point appears on the image (left panel)
4. Repeat for all visible keypoints
5. Use **Sort Corners (A)** after placing all 4 corners to auto-order them
6. Use **Suggest Inner (G)** to auto-fill inner keypoints via homography
7. **Save (S)** or **Save All** to export annotation JSON files

**Controls:**
- Scroll to zoom, click-and-drag to pan
- Right-click on a placed point to remove it
- `N`/`P` or arrow keys to navigate frames
- `Z`/`Y` for undo/redo

### Desktop Annotator

```bash
python -m src.tools.annotator --input-dir data/frames/
```

Requires Tkinter (included with most Python installations).

## Training

### 1. Extract frames

```bash
python -m src.tools.extract_frames videos/ data/frames/ --fps 1
```

### 2. Annotate frames

Use either annotation tool to label keypoints. Save JSON annotations alongside the frame images.

### 3. Train

```bash
python -m src.training.train
```

Training configuration is in `src/training/config.py`. The training loop supports:
- Backbone freeze/unfreeze schedule
- Early stopping on validation loss
- Automatic learning rate scheduling
- Geometric consistency losses (collinearity, distance ratio, convexity)
- PCK@10 and MRE metrics logged during validation

### 4. Inference

```bash
python -c "
from src.inference.predict import CourtPredictor
predictor = CourtPredictor('checkpoints/best_model.pth')
result = predictor.predict('path/to/frame.jpg')
print(result['keypoints'])
"
```

## Evaluation

```bash
python -m src.evaluation.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --annotations data/annotations/test \
    --images data/frames/
```

Metrics: PCK@5/10/20, Mean Reprojection Error, Court IoU, Segmentation IoU.

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT

# CourtVisionNet — Design Specification

## Context

Badminton court detection from video is a prerequisite for player position tracking. Existing methods (CourtKeyNet, YOLOv8s-pose) are designed for broadcast camera angles and struggle with diverse non-standard viewpoints, partial court visibility, and multi-court environments. This project builds a robust court detection system that handles ground-level phone recordings from arbitrary angles — a common real-world scenario with no existing solution — and produces a publishable research paper.

**Reference**: CourtKeyNet (Machine Learning with Applications, 2026) is the current state-of-the-art, achieving KLA@0.05=88.9% on broadcast footage. Its limitations — partial visibility extrapolation, non-broadcast angles, multi-court disambiguation — are exactly the gaps this work addresses.

## System Overview

CourtVisionNet is a hybrid segmentation-keypoint architecture with geometric homography estimation. It combines:
- Multi-channel adaptive preprocessing
- A shared ResNet-50 + FPN backbone
- Dual heads: court line segmentation + 14-keypoint detection
- Homography estimation that fuses both signals
- Multi-court selection and geometric constraint enforcement
- Temporal smoothing for video

```
Input Frame (any angle, any orientation)
    │
    ▼
┌─────────────────────────────────┐
│  Module 1: Preprocessing        │
│  RGB + Grayscale + CLAHE +      │
│  Canny Edge + Court-Color Mask  │
│  → 7-channel tensor (640×640)   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Module 2: Shared Backbone      │
│  ResNet-50 + FPN                │
│  → Multi-scale feature pyramid  │
└──────┬──────────────┬───────────┘
       │              │
       ▼              ▼
┌──────────────┐ ┌────────────────┐
│ Module 3a:   │ │ Module 3b:     │
│ Segmentation │ │ Keypoint       │
│ Head         │ │ Head           │
│ → Line mask  │ │ → 14 keypoints │
│   (640×640)  │ │   + visibility │
└──────┬───────┘ └───────┬────────┘
       │                 │
       ▼                 ▼
┌─────────────────────────────────┐
│  Module 4: Homography Estimator │
│  Fuse segmentation + keypoints  │
│  → 3×3 homography matrix H     │
│  → All court points projected   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Module 5: Court Selection &    │
│  Refinement                     │
│  → Multi-court disambiguation   │
│  → Geometric constraints        │
│  → Temporal smoothing (video)   │
└──────────────┬──────────────────┘
               │
               ▼
Output: Court corners (14 keypoints), full court lines,
        homography matrix, confidence score
```

---

## Module 1: Adaptive Preprocessing Pipeline

### Purpose
Transform raw frames into a multi-channel tensor that emphasizes court lines while suppressing visual noise (shadows, glare, floor color similarity).

### Input
Raw frame (any resolution, any orientation) → resized to 640×640.

### Output Channels (7-channel tensor)
1. **R, G, B (3 ch)** — original color
2. **Grayscale (1 ch)** — luminance contrast; white lines pop against green floor
3. **CLAHE (1 ch)** — Contrast Limited Adaptive Histogram Equalization on grayscale; normalizes uneven indoor lighting from overhead LEDs
4. **Canny edge map (1 ch)** — all edges highlighted; model learns to distinguish court lines from noise edges
5. **Court-color mask (1 ch)** — HSV-based filter isolating green court surface; helps distinguish playing area from wood gaps and walls

### Data Augmentation (training only)
- Random brightness/contrast jitter (±20%)
- Random perspective warping (simulate different camera angles)
- Random crop + resize (simulate partial visibility)
- Horizontal flip
- Random rotation (±15°)
- Random occlusion patches (simulate players/signs blocking lines)
- Color jitter in HSV space
- Gaussian blur (simulate motion blur)
- Gaussian noise

### Rationale
Your footage shows white lines on green floor, but the challenge is distinguishing main court lines from adjacent court lines and non-court edges (barriers, signs, walls). The multi-channel approach gives the model multiple "views" — edges make lines explicit, the color mask identifies court surfaces, and CLAHE handles the venue's uneven lighting.

---

## Module 2: Shared Backbone

### Architecture
ResNet-50 with Feature Pyramid Network (FPN).

### Specifications
- **Input**: 7-channel 640×640 tensor
- **First conv layer**: Modified from 3→64 channels to 7→64 channels
  - RGB channels initialized with ImageNet pretrained weights
  - Extra 4 channels initialized with small random weights (He initialization)
- **FPN outputs**: Feature maps at 4 scales — P2 (160×160), P3 (80×80), P4 (40×40), P5 (20×20)
- **Parameters**: ~25.5M (ResNet-50) + ~2M (FPN) ≈ 27.5M total

### Why ResNet-50 + FPN
- Proven architecture with strong ImageNet/COCO pretrained weights
- FPN provides multi-scale features critical for detecting court lines at varying distances
- Computationally feasible on Colab GPUs (T4/A100)
- Lighter than CourtKeyNet's custom Octave Feature Extractor (15.7M params for their full model)

### Pretraining Strategy
1. Initialize ResNet-50 with COCO-pretrained weights
2. Replace first conv layer for 7-channel input
3. Freeze backbone for first 5 epochs during fine-tuning, then unfreeze

---

## Module 3a: Court Line Segmentation Head

### Purpose
Produce a binary mask of all visible court lines in the image.

### Architecture
Lightweight FPN-based segmentation decoder:
- Takes P2, P3, P4, P5 from the backbone
- Each level: 256-channel 3×3 conv + BN + ReLU
- Upsample all to P2 resolution (160×160)
- Concatenate → 1×1 conv → upsample to 640×640 → sigmoid
- Output: 640×640 single-channel probability map

### Loss Function
- Binary cross-entropy with hard negative mining
- Dice loss (handles class imbalance — lines are thin)
- Combined: L_seg = BCE + Dice

### Training Labels
Auto-generated from annotated keypoints:
- Given visible keypoint positions, draw court lines as polygons (line width ~3-5 pixels at 640×640)
- Lines generated using known badminton court geometry and the annotated corners
- This means annotating keypoints automatically gives us segmentation labels

---

## Module 3b: Keypoint Detection Head

### Purpose
Predict 14 court keypoints with per-keypoint visibility confidence.

### Keypoint Definitions
| ID | Location | Description |
|----|----------|-------------|
| K0 | Top-left outer corner | Doubles boundary |
| K1 | Top-right outer corner | Doubles boundary |
| K2 | Bottom-right outer corner | Doubles boundary |
| K3 | Bottom-left outer corner | Doubles boundary |
| K4 | Left short service / top sideline | |
| K5 | Left short service / bottom sideline | |
| K6 | Right short service / top sideline | |
| K7 | Right short service / bottom sideline | |
| K8 | Net / top sideline | |
| K9 | Net / bottom sideline | |
| K10 | Left short service / center line | |
| K11 | Right short service / center line | |
| K12 | Net / top singles sideline | |
| K13 | Net / bottom singles sideline | |

### Architecture
- Takes P2 (160×160) features from backbone
- **Heatmap branch**: 14 heatmaps at 160×160 (Gaussian blobs at keypoint locations)
- **Regression branch**: Per-keypoint offset refinement (sub-pixel accuracy)
- **Visibility branch**: 14-class sigmoid classifier — probability each keypoint is visible in frame

### Loss Function
- L_heatmap: MSE between predicted and ground truth heatmaps (only for visible keypoints)
- L_kpt: L1 regression loss on refined coordinates (only for visible keypoints)
- L_vis: Binary cross-entropy on visibility predictions
- Combined: L_keypoint = λ₁·L_heatmap + λ₂·L_kpt + λ₃·L_vis

### Key Innovation vs. CourtKeyNet
CourtKeyNet predicts all 4 corners regardless of visibility, producing poor extrapolation for invisible ones. We explicitly predict visibility and defer invisible keypoint inference to the homography module, which uses geometric priors for accurate extrapolation.

---

## Module 4: Homography Estimation

### Purpose
Estimate a 3×3 homography matrix H mapping the canonical court template to the image. This single matrix encodes the complete relationship between real-world court coordinates and image pixels.

### Canonical Court Template
A standardized top-down court with known dimensions (in meters):
- Full court: 13.4m × 6.1m (doubles) / 13.4m × 5.18m (singles)
- All 14 keypoints have known real-world coordinates in this template

### Fusion Strategy

**Case 1: ≥4 visible keypoints (high confidence)**
- Direct homography computation via DLT (Direct Linear Transform)
- Each visible keypoint provides a (image point ↔ template point) correspondence
- RANSAC for outlier rejection
- Highest accuracy path

**Case 2: 2-3 visible keypoints**
- Use visible keypoints as anchor constraints
- Extract line segments from segmentation mask (Hough transform or LSD)
- Match line segments to template lines by angle/position
- Each matched line provides a vanishing point or parallelism constraint
- Solve via Levenberg-Marquardt optimization minimizing reprojection error of keypoints + line alignment error, initialized with a rough homography from the available keypoints + court aspect ratio prior

**Case 3: 0-1 visible keypoints**
- Primary reliance on segmented line matching
- Cluster detected lines into parallel/perpendicular groups
- Match groups to template lines using court geometry priors (aspect ratio, line spacing)
- Estimate homography from line correspondences

**Case 4: Learned fallback**
- Small CNN regressor (3 conv layers + 2 FC layers) that directly predicts 8 homography parameters from the fused feature map
- Trained jointly with the rest of the network
- Used when geometric methods have low confidence

### Output
- 3×3 homography matrix H
- All 14 keypoints projected into image coordinates (including extrapolated invisible ones)
- Per-keypoint confidence (high for detected, lower for extrapolated)
- Overall detection confidence score

### Why Homography Solves Extrapolation
Once H is estimated from visible features, ANY point on the court plane can be projected to the image (or vice versa) — even points outside the frame. This is fundamentally more principled than CourtKeyNet's attempt to regress invisible corner locations.

---

## Module 5: Court Selection & Refinement

### Multi-Court Disambiguation
When multiple courts are visible:
1. Segmentation head finds ALL court lines
2. Cluster line segments into potential court candidates using:
   - Parallel line grouping (court sides are parallel pairs)
   - Perpendicular intersection detection (court corners)
   - Connected component analysis on the segmentation mask
3. Select the main court using (ranked criteria):
   - **Largest area** in the image (main court occupies most pixels when camera points at it)
   - **Most central** in frame (camera typically aimed at court of interest)
   - **Player presence** (optional — if person detector available, prefer court with players)

### Geometric Constraint Enforcement
Inspired by CourtKeyNet's Quadrilateral Constraint Module:
- **Edge ratio consistency**: Predicted court sides must match known aspect ratio (13.4:6.1)
- **Diagonal consistency**: Diagonals must intersect correctly
- **Angle consistency**: Internal angles must be valid for perspective-projected rectangle
- **Parallel line consistency**: Opposite sides must converge to valid vanishing points

Applied as both:
- A post-processing verification step (reject geometrically invalid predictions)
- A loss term during training (Geometric Consistency Loss from CourtKeyNet)

### Temporal Smoothing (Video Mode)
- **Kalman filter** on homography parameters H across consecutive frames
- State: 8 homography parameters (H is 3×3 with one degree of freedom fixed)
- Handles:
  - Frame-to-frame jitter (smooths noisy predictions)
  - Brief occlusions (carries forward previous estimate for ~10-15 frames)
  - Camera movement (allows larger state updates when motion detected)
- Optional: Exponential moving average (EMA) as simpler alternative for initial implementation

---

## Module 6: Annotation Tool

### Purpose
Enable efficient annotation of training data from video files.

### Features
- **Video frame extraction**: Load MOV/MP4 files, extract frames at configurable FPS
- **Keypoint annotation**: Click visible keypoints (any subset of K0-K13)
- **Visibility flags**: Automatically set based on which points are clicked
- **Court type selector**: Singles (0) / Doubles (1) / Alternative layout (2)
- **Auto-generated outputs**: From clicked keypoints:
  - Full court line segmentation mask (using known court geometry)
  - Homography matrix (for verification overlay)
  - Projected positions of all 14 keypoints
- **Geometric validation**: Real-time check that clicked points form valid court geometry
- **Overlay visualization**: Shows projected full court on the image for visual verification
- **Frame propagation**: Copy annotations to adjacent frames with optional offset adjustment
- **Undo/redo**: Full history stack
- **Export format**: JSON per frame with schema:
  ```json
  {
    "image_path": "frames/video_001/frame_0042.jpg",
    "image_size": [640, 640],
    "court_class": 1,
    "keypoints": [[x0, y0], [x1, y1], ..., [x13, y13]],
    "visibility": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    "bounding_box": [cx, cy, w, h],
    "mask_path": "masks/video_001/frame_0042.png"
  }
  ```
  All coordinates normalized to [0, 1]. Invisible keypoints have coordinates set to [-1, -1].

### Implementation
- Python with Tkinter (GUI) + OpenCV (image processing)
- Similar to CourtKeyNet's annotation tool but extended for 14 keypoints and auto-mask generation

---

## Training Pipeline

### Dataset Composition

**Phase 1 — Public data pretraining:**
- CourtKeyNet dataset: 5000 images (broadcast angles), available on GitHub
  - Source: https://github.com/adithyanraj03/Paper_09_Data-Set_CourtKeyNet
- BadmintonData (Jain, 2023): Additional broadcast footage
  - Source: https://universe.roboflow.com/himani-jain/badmintonc
- These provide 4-corner annotations; we auto-generate the remaining 10 keypoints from court geometry

**Phase 2 — Custom data fine-tuning:**
- Your 36 videos → extract frames (e.g., every 1 second ≈ ~30 fps × duration)
- Estimate ~500-2000 annotated frames after manual labeling
- Heavy augmentation to compensate for smaller dataset size

**Phase 3 — Self-training (optional, for paper):**
- Run trained model on unannotated frames
- Manually verify/correct predictions with high confidence
- Retrain with expanded dataset
- Report improvement in paper as evidence of generalization

### Training Configuration
- **Framework**: PyTorch
- **Hardware**: Google Colab Pro (T4 or A100 GPU)
- **Input resolution**: 640 × 640
- **Batch size**: 8-16 (depending on GPU memory)
- **Optimizer**: AdamW, lr=1e-3, weight decay=1e-4
- **Scheduler**: Cosine annealing
- **Epochs**: 100 (with early stopping, patience=15)
- **Backbone freeze**: First 5 epochs

### Total Loss Function
```
L_total = α·L_seg + β·L_keypoint + γ·L_geometric + δ·L_homography

Where:
- L_seg = BCE + Dice (segmentation)
- L_keypoint = λ₁·L_heatmap + λ₂·L_kpt + λ₃·L_vis (keypoints)
- L_geometric = L_edge + L_diag + L_angle (geometric consistency)
- L_homography = reprojection error of visible keypoints through estimated H

Initial weights: α=1.0, β=1.0, γ=1.0, δ=0.5
```

---

## Evaluation Metrics

### Primary Metrics (comparable to CourtKeyNet)
- **KLA@0.05**: Keypoint Localization Accuracy at 5% threshold
- **KLA@0.1**: Keypoint Localization Accuracy at 10% threshold
- **CD-IoU**: Court Detection Intersection over Union
- **PCR**: Perfect Court Rate (all keypoints correct)
- **MPE**: Mean Pixel Error

### Additional Metrics (novel to our evaluation)
- **KLA by visibility count**: Accuracy as a function of how many keypoints are visible (measures extrapolation quality)
- **Cross-angle accuracy**: Train on angle subset A, test on angle subset B
- **Multi-court precision**: Correct main court selection rate
- **Temporal consistency**: Frame-to-frame court boundary jitter (std of corner positions in stable video segments)
- **Homography reprojection error**: Mean distance between projected and actual keypoints

### Ablation Studies
1. RGB-only (3ch) vs. multi-channel preprocessing (7ch)
2. Segmentation-only vs. keypoint-only vs. hybrid
3. 4 keypoints vs. 14 keypoints
4. With vs. without geometric constraints
5. With vs. without temporal smoothing
6. Performance vs. number of visible keypoints
7. Cross-angle generalization
8. Effect of public data pretraining
9. Effect of self-training

---

## Paper Structure

**Target**: Machine Learning with Applications (same journal as CourtKeyNet) or CVPR/ECCV workshop

### Outline
1. **Introduction**: Court detection for player tracking; limitations of broadcast-only methods; our contributions
2. **Related Work**: Court detection (CourtKeyNet, YOLOv8s-pose, Hough Transform); keypoint detection; homography estimation; sports analytics
3. **Methodology**:
   - 3.1 Problem formulation (14-keypoint + homography)
   - 3.2 Preprocessing pipeline
   - 3.3 Shared backbone
   - 3.4 Dual-head architecture (segmentation + keypoints)
   - 3.5 Homography estimation and fusion
   - 3.6 Court selection and geometric constraints
   - 3.7 Temporal consistency
   - 3.8 Loss functions
4. **Dataset and Annotation**:
   - 4.1 Data collection (diverse angles, venues)
   - 4.2 Annotation tool
   - 4.3 Dataset statistics
5. **Experiments**:
   - 5.1 Implementation details
   - 5.2 Comparison with existing methods
   - 5.3 Ablation studies
   - 5.4 Cross-angle evaluation
   - 5.5 Partial visibility analysis
   - 5.6 Multi-court evaluation
   - 5.7 Temporal consistency analysis
   - 5.8 Inference speed
6. **Conclusion and Future Work**

---

## Project Structure (Repository)

```
badminton-court-finder/
├── data/
│   ├── raw/                  # Raw videos
│   ├── frames/               # Extracted frames
│   ├── annotations/          # JSON annotation files
│   └── masks/                # Generated segmentation masks
├── models/
│   ├── backbone.py           # ResNet-50 + FPN (7-channel)
│   ├── segmentation_head.py  # Court line segmentation
│   ├── keypoint_head.py      # 14-keypoint detection + visibility
│   ├── homography.py         # Homography estimation + fusion
│   ├── court_selection.py    # Multi-court disambiguation
│   ├── courtvisionnet.py     # Full model assembly
│   └── losses.py             # All loss functions
├── preprocessing/
│   ├── channels.py           # Multi-channel generation
│   └── augmentation.py       # Data augmentation pipeline
├── tools/
│   ├── annotator.py          # Annotation tool (Tkinter + OpenCV)
│   ├── extract_frames.py     # Video → frame extraction
│   └── visualize.py          # Result visualization
├── training/
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation metrics
│   ├── dataset.py            # PyTorch dataset class
│   └── config.py             # Hyperparameters
├── inference/
│   ├── predict.py            # Single image inference
│   └── video.py              # Video inference with temporal smoothing
├── notebooks/
│   ├── train_colab.ipynb     # Colab training notebook
│   └── analysis.ipynb        # Results analysis
├── docs/
│   └── paper/                # LaTeX paper drafts
├── videos/                   # Source videos (36 MOV files)
└── requirements.txt
```

---

## Implementation Order

1. **Annotation tool** — needed first to generate training data
2. **Preprocessing pipeline** — can be developed and tested independently
3. **Dataset class + data loading** — bridge between annotations and training
4. **Backbone + segmentation head** — train segmentation first (simpler task)
5. **Keypoint head** — add to backbone, joint training
6. **Homography estimation** — can be developed with geometric methods first, learned component later
7. **Court selection module** — requires working segmentation
8. **Temporal smoothing** — last module, applied at inference time
9. **Evaluation + ablation studies** — systematic testing
10. **Paper writing** — concurrent with experiments

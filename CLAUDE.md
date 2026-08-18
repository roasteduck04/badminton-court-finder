# CourtVisionNet

Badminton court keypoint detection from non-standard camera angles. Research project targeting a publishable paper.

## Project structure

- `src/models/` - CourtVisionNet architecture (ResNet-50 + FPN backbone, segmentation head, keypoint head)
- `src/preprocessing/` - 7-channel input pipeline (RGB + grayscale + CLAHE + Canny + court prior)
- `src/training/` - Dataset, training loop, config
- `src/evaluation/` - Metrics (PCK, MRE, Court IoU, Segmentation IoU)
- `src/inference/` - Prediction and visualization
- `src/tools/blender_*.py` - Blender synthetic data pipeline (runs inside Blender's Python, not project Python)
- `src/tools/blender_to_cvn.py` - Converter from Blender metadata to CVN format (runs with project Python)
- `src/tools/annotator.py` - Tkinter GUI court annotator
- `src/tools/serve_annotator.py` - HTTP server for browser-based annotator (port 8000)
- `src/tools/serve_review.py` - HTTP server for inference review UI (port 8001)
- `src/tools/scrape_courts.py` - Bing/Google image scraper
- `src/tools/extract_frames.py` - Video frame extraction
- `src/tools/convert_dataset.py` - Roboflow COCO format converter
- `src/tools/coco_to_cvn.py` - COCO keypoints to CVN format
- `src/tools/run_inference.py` - Roboflow API inference runner
- `src/court_geometry.py` - Court dimensions, 30 keypoints in 6x5 grid, homography utils
- `scripts/` - CLI utilities (generate.py, prepare_colab.py, prepare_roboflow.py)
- `notebooks/` - Colab/Kaggle training notebooks
- `tests/` - Unit tests (16 test files)

## Key conventions

- 30 keypoints (K0-K29) in a 6x5 grid: 6 rows along court length x 5 columns across width
- BWF court: 13.4m x 6.1m
- CVN annotation format: `{"image_path", "image_size": [h, w], "keypoints" (normalized 0-1), "visibility", "bounding_box"}`
- `image_size` uses `[height, width]` convention

## Data layout

- `data/images/` + `data/annotations/` - Hand-labeled real images (70 images, 69 annotations)
- `data/blender/` - Blender synthetic data (500 renders with metadata and CVN annotations)
- `data/cvn_dataset/` - Roboflow COCO export converted to CVN (train/valid/test splits)
- `data/colab_data/` - Merged dataset for Colab training (real + Roboflow + synthetic)
- `data/roboflow_upload/` - Images formatted for Roboflow upload

## Blender pipeline

Requires Blender 5.2+. The `blender_*.py` modules use `bpy` and run inside Blender's Python environment.

- Engine name is `BLENDER_EEVEE` (not `EEVEE`) in Blender 5.2
- Scene is built procedurally (no .blend template file needed)
- Uses `users_collection` pattern for collection management (Blender 5.2 compat)
- Adjacent courts support 0-8 in a 3x3 grid layout around the main court
- Blender modules: court, camera, lighting, occluders, environment, venue_details, mannequin, render

Generate: `blender --background --python src/tools/blender_render.py -- --count 500 --engine BLENDER_EEVEE`
Convert: `python src/tools/blender_to_cvn.py`

## Training

- Default config in `src/training/config.py`: 7 channels, 30 keypoints, 640px input, 160px heatmap, batch 8, 100 epochs, LR 3e-5, OneCycleLR
- Loss: BCE+Dice (seg) + MSE (heatmap) + L1 (offset) + BCE (visibility) + geometric (collinear, ratio, convex)
- AMP (mixed precision) training supported
- Backbone freeze for first 5 epochs, then unfreeze
- Resume from checkpoint via `--resume-from`

## Commands

```bash
# Run tests
python -m pytest tests/ -v

# Train
python -m src.training.train

# Evaluate
python -m src.evaluation.evaluate --checkpoint checkpoints/best_model.pt --annotations data/colab_data/test/annotations --images data/colab_data/test/images

# Generate synthetic data
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python src/tools/blender_render.py -- --count 500 --engine BLENDER_EEVEE --samples 32 --seed 42

# Convert synthetic data
python src/tools/blender_to_cvn.py --min-visible 4

# Prepare merged dataset for Colab
python scripts/prepare_colab.py

# Prepare for Roboflow upload
python scripts/prepare_roboflow.py

# Launch browser annotator
python src/tools/serve_annotator.py

# Launch inference review server
python src/tools/serve_review.py
```

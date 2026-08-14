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
- `src/court_geometry.py` - Court dimensions, 30 keypoints in 6x5 grid, homography utils
- `tests/` - Unit tests

## Key conventions

- 30 keypoints (K0-K29) in a 6x5 grid: 6 rows along court length x 5 columns across width
- BWF court: 13.4m x 6.1m
- CVN annotation format: `{"image_path", "image_size": [h, w], "keypoints" (normalized 0-1), "visibility", "bounding_box"}`
- `image_size` uses `[height, width]` convention

## Blender pipeline

Requires Blender 5.2+. The `blender_*.py` modules use `bpy` and run inside Blender's Python environment.

- Engine name is `BLENDER_EEVEE` (not `EEVEE`) in Blender 5.2
- Scene is built procedurally (no .blend template file needed)
- Uses `users_collection` pattern for collection management (Blender 5.2 compat)
- Adjacent courts support 0-8 in a 3x3 grid layout around the main court

Generate: `blender --background --python src/tools/blender_render.py -- --count 500 --engine BLENDER_EEVEE`
Convert: `python src/tools/blender_to_cvn.py`

## Commands

```bash
# Run tests
python -m pytest tests/ -v

# Train
python -m src.training.train

# Evaluate
python -m src.evaluation.evaluate --checkpoint checkpoints/best_model.pt --annotations data/annotations/test --images data/frames/

# Generate synthetic data
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python src/tools/blender_render.py -- --count 500 --engine BLENDER_EEVEE --samples 32 --seed 42

# Convert synthetic data
python src/tools/blender_to_cvn.py --min-visible 4
```

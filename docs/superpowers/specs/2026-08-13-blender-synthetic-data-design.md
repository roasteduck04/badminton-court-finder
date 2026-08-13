# Blender Synthetic Data Pipeline — Design Spec

**Date**: 2026-08-13
**Goal**: Generate synthetic badminton court images with ground-truth keypoint annotations using Blender, to augment the existing 753-image Roboflow dataset for CourtVisionNet training.

## Overview

A modular Blender Python pipeline that procedurally generates varied court scenes — randomizing camera angles, court surfaces, lighting, on-court occlusion, and venue environment — then renders images with projected keypoint metadata. A separate converter transforms the raw output into CVN annotation format ready for training.

**Target scale**: ~500 images initially, scaling up as needed.

**Variation dimensions** (all four):
- Camera angles (broadcast, sideline, corner, overhead, low, random)
- Court surface/color (green, blue, red, wood, grey)
- Lighting conditions (fluorescent, mixed/natural, dim, harsh shadows, competition)
- Occlusion and clutter (players, net, equipment, adjacent courts, venue surroundings)

## Architecture

Approach B — modular scene-builder + renderer. Six Blender modules plus one post-processing converter.

## Modules

### 1. Court Geometry (`src/tools/blender_court.py`)

Builds the 3D badminton court to BWF specifications using constants from `src/court_geometry.py`.

**Components:**
- **Court surface**: Single plane (13.4m × 6.1m) with a procedural material. Base color is randomizable (green, blue, red, wood, grey). Subtle noise texture for realism (floor grain, wear).
- **Court lines**: Separate mesh slightly above the surface (Z offset ~1mm). White by default, color randomizable (white, yellow, light grey). Line width ~40mm per BWF standard.
- **Keypoint empties**: 30 empties placed at line intersections (K0–K29), matching `COURT_KEYPOINTS_TEMPLATE` in `court_geometry.py`. These are the ground-truth 3D positions projected to 2D during export.
- **Net + posts**: Simple net mesh at center line (1.55m at edges, 1.524m at center). Cylinder posts. Toggleable for occlusion training.

**Interface**: `build_court(scene, config)` → `{"surface": obj, "lines": obj, "net": obj, "posts": [obj], "keypoints": [empty_0..empty_29]}`

### 2. Camera (`src/tools/blender_camera.py`)

Camera placement with named strategies covering angle diversity.

**Strategies:**
| Strategy | Description | Elevation |
|----------|-------------|-----------|
| Broadcast | High behind baseline, looking down | 30–45° |
| Sideline | Mid-height along sideline | 15–30° |
| Corner | Positioned at court corner, diagonal view | 20–40° |
| Overhead | Nearly top-down | 60–85° |
| Low angle | Ground-level, looking across court | 0–10° |
| Random | Uniform sample from hemisphere above court | Configurable |

**Per-strategy randomization:**
- Position jitter: ±1–2m
- Look-at target offset from court center
- Focal length: 28–85mm (phone cameras to broadcast lenses)

**Visibility computation**: Projects all 30 keypoint empties through the camera matrix, marks which fall within frame bounds (0–1 normalized).

**Interface**: `place_camera(scene, strategy, config)` → camera object + metadata dict (intrinsics, extrinsics, projected 2D keypoints, visibility)

### 3. Lighting (`src/tools/blender_lighting.py`)

Lighting rigs simulating real indoor badminton venues.

**Presets:**
| Preset | Description | Color temp |
|--------|-------------|------------|
| Indoor fluorescent | Multiple rectangular area lights, high-mounted, even | ~5000–5500K |
| Mixed/natural | Overhead + sun lamp at angle, simulating windows | Warm + cool |
| Dim venue | Fewer weak lights, visible falloff at edges | ~4000K |
| Harsh shadows | Single strong directional light | ~5500K |
| Competition | Bright, even, multi-position, minimal shadows | ~5500K |

**Per-preset randomization:**
- Intensity: ±20%
- Color temperature: ±300K
- Light position jitter
- Optional procedural sky/solid color for ambient fill

**Interface**: `setup_lighting(scene, preset, config)` → adds lights, returns metadata

### 4. Occluders (`src/tools/blender_occluders.py`)

On-court objects that partially block court lines.

**Occluder types:**
- **Players**: Low-poly capsule/cylinder figures (~1.7m), 0–4 per render, biased toward service areas
- **Net assembly**: Visibility toggle, sag/drape variation
- **Umpire chair**: Box+seat geometry at net post, present in ~20% of renders
- **Equipment**: Rackets (flat rectangles), shuttlecocks (cone+hemisphere), scattered on court

**Distribution:**
- ~30% of renders have zero occluders (clean court views)
- Typical: 0–2 occluders; occasionally 3–4 for heavy occlusion
- Player materials: dark/colored clothing tones

**Interface**: `add_occluders(scene, config)` → places objects, returns metadata

### 5. Environment (`src/tools/blender_environment.py`)

Venue surroundings beyond the main court — background noise the model must learn to ignore.

**Elements:**
- **Adjacent courts**: 0–2 partial courts alongside main court (~1–2m gap), possibly different surface colors
- **Venue floor**: Extended floor plane in a different material (concrete, rubber, wood)
- **Walls/barriers**: Vertical planes at venue edges (10–20m out), optional curtain dividers between courts
- **Spectator area**: Low-poly bench rows/chairs, present in ~40% of renders
- **Scoreboards**: Flat rectangle on wall/stand at courtside
- **Ceiling**: High plane (~9–12m) for realistic light reflection

**Randomization:**
- Venue size (tight community hall vs. spacious sports center)
- Adjacent court count and relative angle
- Divider curtain presence

**Interface**: `build_environment(scene, config)` → references to placed objects

### 6. Renderer & Orchestrator (`src/tools/blender_render.py`)

Main entry point that wires modules together for batch generation.

**Per-image flow:**
1. Clear scene
2. `build_court()` — random surface color/texture
3. `build_environment()` — random venue surroundings
4. `place_camera()` — random strategy (weighted)
5. `setup_lighting()` — random preset
6. `add_occluders()` — random count/placement
7. Render to `data/blender/raw/images/blender_NNNN.png`
8. Export metadata to `data/blender/raw/metadata/blender_NNNN.json`

**Default strategy weights**: broadcast 30%, sideline 20%, corner 20%, overhead 15%, low 10%, random 5%

**Render settings:**
- Engine: Cycles (realism) or EEVEE (speed during iteration)
- Resolution: 640×640 (matches training input size)
- Samples: 64–128 for Cycles

**Metadata JSON format:**
```json
{
  "image_file": "blender_0001.png",
  "resolution": [640, 640],
  "camera": {
    "strategy": "sideline",
    "position": [x, y, z],
    "rotation": [rx, ry, rz],
    "focal_length": 35.0,
    "keypoints_3d": [[x, y, z], ...],
    "keypoints_2d": [[px, py], ...],
    "visibility": [1, 1, 0, ...]
  },
  "lighting": {"preset": "indoor_fluorescent", "lights": [...]},
  "occluders": [{"type": "player", "position": [...]}],
  "court": {"surface_color": "green", "line_color": "white"},
  "environment": {"adjacent_courts": 1, "venue_size": "medium"}
}
```

**Batch config**: Top-level config dict controls total count, strategy weights, enabled presets, render engine, output paths.

**Headless execution**: `blender --background data/blender/court_template.blend --python src/tools/blender_render.py -- --count 500`

### 7. Converter (`src/tools/blender_to_cvn.py`)

Post-processing: transforms raw Blender output into CVN annotation format.

**Input**: `data/blender/raw/` (images + metadata JSONs)

**Processing per image:**
1. Read metadata JSON
2. Normalize `keypoints_2d` to [0, 1] by dividing by resolution
3. Use `visibility` array directly from camera module
4. Filter out images with fewer than 4 visible keypoints (configurable via `--min-visible`)
5. Compute bounding box from visible keypoints

**Output:**
- Images → `data/blender/images/blender_NNNN.png`
- Annotations → `data/blender/annotations/blender_NNNN.json` in CVN format:
```json
{
  "image_path": "blender_0001.png",
  "image_size": [640, 640],
  "keypoints": [[0.45, 0.82], ...],
  "visibility": [1, 1, 0, ...],
  "bounding_box": [min_x, min_y, w, h]
}
```

**Usage**: `python src/tools/blender_to_cvn.py` (no args, standard paths). Optional `--min-visible 4`.

## File Layout

```
src/tools/
├── blender_court.py
├── blender_camera.py
├── blender_lighting.py
├── blender_occluders.py
├── blender_environment.py
├── blender_render.py
└── blender_to_cvn.py

data/blender/
├── court_template.blend      # Base scene (built interactively via MCP)
├── raw/                      # Blender output (intermediate)
│   ├── images/
│   └── metadata/
├── images/                   # Final CVN-ready images
└── annotations/              # Final CVN-ready annotations
```

## Workflow

1. Build the court model interactively using Blender MCP → save as `data/blender/court_template.blend`
2. Run batch generation: `blender --background data/blender/court_template.blend --python src/tools/blender_render.py -- --count 500`
3. Convert to CVN format: `python src/tools/blender_to_cvn.py`
4. Verify in dashboard: `http://localhost:8000/dashboard` → "Blender Synthetic" source
5. Train: `CourtDataset` loads from `data/blender/` alongside `data/cvn_dataset/`

## Dependencies

- Blender's bundled Python (`bpy`) for generation scripts (modules 1–6)
- Only stdlib + numpy for the converter (module 7)
- No new pip packages required

## Integration with Existing Pipeline

- The converter output matches the exact format used by `coco_to_cvn.py` and expected by `CourtDataset`
- `serve_annotator.py` already has `/blender/` routes and dashboard support for browsing Blender images
- Training config (`TrainConfig`) can point at `data/blender/` paths or a merged dataset directory

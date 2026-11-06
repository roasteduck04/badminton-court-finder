# Keypoint Annotation System — Handover

Browser-based annotator for placing the 30 court keypoints (K0-K29) on images, with a
homography overlay, homography-based suggestions and automatic canonical labeling.
A Tkinter annotator with the same data model is kept for offline use.

| File | Size | Dependencies |
|---|---|---|
| `src/tools/annotator.html` | ~840 lines | none (plain JS in the browser) |
| `src/tools/serve_annotator.py` | ~530 lines | Python 3.10+ stdlib only |
| `src/tools/annotator.py` (Tkinter) | ~850 lines | numpy, OpenCV, Pillow, Tkinter, `src/court_geometry.py` |
| `main.bat` | 4 lines | launches the server |

---

## Architecture

```
 annotator.html  <-- fetch/JSON -->  serve_annotator.py (127.0.0.1:8000)  <-->  data/images/
  canvas UI, DLT homography,           GET  /api/images, /api/annotations          data/annotations/
  canonical labeling, undo/redo        POST /api/save, /api/scraped/{keep,discard} data/scraped/
```

The server binds to loopback by default, accepts only JSON POST bodies
(`Content-Type: application/json`), checks the `Origin` of state-changing requests,
validates filenames with a strict regex and writes annotations atomically. Errors come
back as 4xx/5xx with a JSON `{"error": ...}` body; the page checks `response.ok` on every
request and shows the error instead of silently continuing.

### Server API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | annotator.html |
| GET | `/api/images` | JSON array of image filenames in `data/images/` |
| GET | `/api/annotations` | JSON object `"stem.json" -> annotation` from `data/annotations/` |
| GET | `/images/{name}` | an image from `data/images/` |
| POST | `/api/save` | body `{"filename": "stem.json", "data": {...}}` |
| GET | `/api/scraped` | `{"files": [...], "total": n}` for `data/scraped/` |
| POST | `/api/scraped/keep` | moves a scraped image into `data/images/`; returns `{"ok": true, "filename": <final name>}` (renamed to `name_1.jpg` etc. on a collision) |
| POST | `/api/scraped/discard` | deletes a scraped image (permanent) |
| GET | `/dashboard`, `/api/dashboard/...`, `/dataset/...` | optional dataset dashboard |

---

## Annotation format (CVN)

One JSON per image, coordinates normalized to [0, 1]; unplaced points are `[-1, -1]`
with visibility 0.

| Field | Type | Description |
|---|---|---|
| `image_path` | string | image **basename** (both annotators save the basename) |
| `image_size` | [int, int] | `[height, width]` of the image as decoded by the browser / OpenCV (EXIF-rotated). The HTML annotator refuses to save if the image did not load; it never guesses a size |
| `keypoints` | 30 x [x, y] | normalized coordinates |
| `visibility` | 30 x {0,1} | 1 = point is labeled |
| `bounding_box` | [x, y, w, h] | of labeled points, normalized |
| `source` | string, optional | `"manual"` for annotator output (`"roboflow"`, `"blender"` elsewhere) |
| `canonical_perm` | string or null | symmetry applied to reach the canonical labeling (cumulative); `null` = orientation could not be determined, labels not canonicalized |
| `keypoint_origin` | 30 x string, optional | `"clicked"`, `"suggested"` (accepted homography suggestion) or `"none"`. Omitted when the file was created before origins were tracked and some points' origin is unknown |
| `occluded` | 30 x bool, optional | labeled but hidden (preserved by the annotators, not editable in the UI) |
| `group` | string, optional | split group (written by the data pipeline) |

---

## Canonical labeling (important)

A badminton court is symmetric under a left/right mirror and a 180-degree rotation, so
every image admits 4 labelings that fit the template equally well. Before this change the
hand labels were mostly the 180-degree rotation of the Roboflow labels, and the old
"Sort Corners" button sorted corners by image y, which picks different labelings for
different camera angles.

The canonical convention (`src/court_geometry.py`, `canonicalize`):

1. **Handedness:** the template->image homography has a positive Jacobian determinant
   (a camera above the floor never sees a mirrored court).
2. **Row direction:** the image direction of increasing row index (K0 row -> K25 row),
   measured at the centroid of the labeled template points, has a positive dot product
   with (1, -1)/sqrt(2), i.e. points up and/or right.

End-on view: K0 = near-left corner, K4 = near-right, K25 = far-left, K29 = far-right.
Side view: row 0 (K0-K4) is the image-left baseline.

Both annotators canonicalize **automatically on save** (and on `A` / "Canonicalize"),
so the annotator only has to label one consistent labeling of the visible court; the
mirror / 180-degree choice is fixed for them. This needs 4 labeled points with no 3 on
one court line; otherwise the file is saved unchanged with `canonical_perm: null` and a
red warning is shown.

The court diagram in the right panel is drawn in the canonical orientation (camera
behind the near baseline: K0 bottom-left, FAR at the top), so following the diagram
literally for an end-on view already gives canonical labels.

The JS port (`hasHomographySupport`, `courtOrientation`, `canonicalPermName`,
`permuteAnn`, `canonicalizeAnn`, between `/* BEGIN CANONICAL */` and
`/* END CANONICAL */`) and its `SYMMETRY_PERMS` / `TPL` constants are checked against
the Python implementation by `tests/test_annotator_logic.py` (the canonicalization test
runs the extracted JS under `node`; skipped if node is not installed).

To canonicalize existing files: `python scripts/canonicalize_annotations.py
data/annotations --dry-run` (reports; on the current 116 hand labels: 43 already
canonical, 71 need rot180, 2 a mirror), `--out DIR` (writes copies) or `--in-place`
(keeps `<file>.bak`). `scripts/prepare_colab.py` re-canonicalizes everything anyway.

---

## Workflow (browser)

1. `python src/tools/serve_annotator.py` (or double-click `main.bat`); the browser opens
   http://127.0.0.1:8000.
2. Click a keypoint on the court diagram (or press `0`-`9`, or click the strip), then
   click its position on the image. Selection advances to the next unplaced point.
3. With 4+ well-spread points the green wireframe appears. `G` proposes all unplaced
   points that project inside the image (yellow); `Y`/Enter accepts, `N`/Esc rejects.
   Suggestions need 4 placed points with no 3 on one court line (any such set works,
   not just the corners). Accepted points are recorded as `"suggested"`.
4. `S` saves (canonicalizing first) and jumps to the next unannotated image. The status
   banner says whether the labels were relabeled, or shows the server error.

| Key | Action |
|---|---|
| `0`-`9` | select K0-K9 |
| scroll / drag | zoom / pan |
| right-click | clear nearest point |
| `Z` / `Ctrl+Z` | undo |
| `Y` | redo, or accept a suggestion |
| `N` / `->`, `P` / `<-` | next / previous image (`N` rejects while a suggestion is shown) |
| `A` | canonicalize labels now |
| `G` | suggest unplaced points |
| `S` | save |
| `R` | reset zoom |
| `Esc` | deselect |

"Filter Scraped" reviews `data/scraped/`: `K` keeps (moves into `data/images/`, under the
name the server returns), `D` discards (deletes), `Esc` closes.

---

### Reviewing machine suggestions

Pre-annotations (for example from the classical detector, propagated within a video) go in `data/annotations_suggested/<stem>.json` in CVN format, with `keypoint_origin` = `"suggested"`. They are never training data on their own.

- **How they appear:** the annotator loads a suggestion only for images whose file in `data/annotations/` is missing or has no points. Such an image shows a blue badge in the list, and the status bar asks you to review it.
- **Reviewing:** check every point. Re-click wrong points (origin becomes `"clicked"`), and right-click points that should not be there.
- **Accepting:** `S` writes the reviewed labels to `data/annotations/` and moves on to the next image that still needs work.
- **Save All:** never saves unreviewed suggestions.

## Homography engine (JS)

`homoN(src, dst)` is a Hartley-normalized DLT with h33 = 1 (exact 8x8 solve for 4 points,
normal equations for more). `homoFromAnn` fits template -> normalized image coordinates
for the overlay and suggestions; canonicalization fits template -> pixels (normalized
coordinates times the image size) so the row direction is exact for non-square images.
There is no RANSAC and no lens-distortion model: one misplaced point skews the overlay,
and wide-angle images drift at the edges. Use `python scripts/qa_annotations.py` to find
labels with a large homography residual (default flag > 15 px at original resolution).

---

## Python annotator (Tkinter)

`python -m src.tools.annotator --input-dir data/images [--output-dir data/annotations]`.
Same format and behaviour as the browser version: `AnnotationState` (data model with
undo/redo, `set_keypoint`, `clear_keypoint`, `get_homography`, `project_full_template`,
`auto_suggest_inner`, `canonicalize`, `to_dict` / `from_dict`), `save_annotation`
(basename `image_path`, canonicalizes by default), `load_annotation`. On save it also
writes a line mask to `<output-dir>/../masks/`. Keys: `a` canonicalize, `g` suggest,
`s` save, `z`/`y` undo/redo, `n`/`p` next/prev, `r` reset zoom.

---

## After annotating: building training data

Do not split hand labels per image: photos from one burst are near-identical. Use

```
python scripts/qa_annotations.py                 # label QA per source
python scripts/prepare_colab.py                  # -> data/colab_data_v2 + manifest.json
```

`prepare_colab.py` groups hand labels into capture sessions (EXIF time, else the
filename timestamp, 10-minute gaps; phone counters `IMG_NNNN*` clustered), Roboflow
frames by source video (merging videos that share near-identical frames), keeps Blender
in train only, and writes `train`, `val`, `test_real`, `test_roboflow` plus a frozen
`manifest.json` (rebuild exactly with `--manifest`).

---

## Adapting for another project

1. Replace `TPL`, `NK`, `NAMES`, `KC`, `LINES` in annotator.html and the matching
   `COURT_KEYPOINTS_TEMPLATE` / `get_court_lines()` in Python.
2. Recompute `SYMMETRY_PERMS` for your template's symmetry group (or remove
   canonicalization if the template has no symmetry) and adapt `c2s` / `drawCourt()`.
3. Point `AnnotatorConfig` (`images_dir`, `annotations_dir`, `scraped_dir`) in
   serve_annotator.py at your data (there are no CLI options for the directories).
4. Label 2-3 images and check the overlay aligns before labeling at scale.

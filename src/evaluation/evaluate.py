"""Standalone evaluation CLI for CourtVisionNet.

    python -m src.evaluation.evaluate --checkpoint checkpoints/best_model.pt \
        --annotations data/x/annotations --images data/x/images \
        [--manifest path --split test_real] [--out results/run.json] \
        [--no-refine] [--tta] [--n-boot 1000] [--seg-thickness N]

Metrics are defined in ``src/evaluation/metrics.py``. With ``--out X.json``
three files are written: ``X.json`` (summary), ``X_rows.jsonl`` (one row per
image: errors, symmetry perm / chirality, RANSAC inlier count, ...) and
``X.md`` (markdown tables). The markdown table is always printed.
"""

import argparse
import json
import math
import os
import re
import sys
import time

import cv2
import numpy as np

from src.court_geometry import KEYPOINT_NAMES
from src.evaluation.metrics import (
    HEADLINE_METRICS,
    REF_LONG_SIDE,
    aggregate,
    evaluate_image,
    render_line_mask,
    segmentation_iou,
)
from src.preprocessing.letterbox import (
    LetterboxInfo,
    content_mask,
    keypoints_to_letterbox,
    letterbox,
)

# Training line_thickness default (TrainConfig); used when a checkpoint has no config.
DEFAULT_SEG_THICKNESS = 5

_PREFIX_SOURCES = {"mine_": "manual", "ds_": "roboflow", "bl_": "blender"}
_ANN_KEYS = ("annotation", "annotation_path", "ann_path", "ann", "json")
_IMG_KEYS = ("image", "image_file", "img_path", "image_path")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def infer_source(ann, name):
    """Source: annotation "source" field, else filename prefix, else unknown."""
    if ann.get("source"):
        return str(ann["source"])
    base = os.path.basename(name)
    for prefix, src in _PREFIX_SOURCES.items():
        if base.startswith(prefix):
            return src
    return "unknown"


def infer_group(ann, name, source):
    """Group: annotation "group" field, else Roboflow video stem
    (``[ds_]outputN_image_0001...`` -> ``outputN``), else the source."""
    if ann.get("group"):
        return str(ann["group"])
    base = os.path.basename(name)
    for prefix in _PREFIX_SOURCES:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    m = re.match(r"^(.+?)_image_\d+", base)
    if m:
        return m.group(1)
    return source


def _resolve(path, base_dirs):
    if os.path.isabs(path):
        return path
    for d in base_dirs:
        cand = os.path.join(d, path)
        if os.path.exists(cand):
            return cand
    return os.path.join(base_dirs[0], path) if base_dirs else path


def _find_image(ann, ann_path, images_dir, explicit=None):
    if explicit and os.path.exists(explicit):
        return explicit
    name = os.path.basename(ann.get("image_path", "") or "")
    cands = []
    if images_dir:
        cands.append(os.path.join(images_dir, name))
    raw = ann.get("image_path", "")
    if raw and os.path.isabs(raw):
        cands.append(raw)
    ann_dir = os.path.dirname(ann_path) if ann_path else ""
    cands += [os.path.join(ann_dir, "..", "images", name), os.path.join(ann_dir, name)]
    if not name:
        stem = os.path.splitext(os.path.basename(ann_path or ""))[0]
        for ext in (".jpg", ".png", ".jpeg"):
            if images_dir:
                cands.append(os.path.join(images_dir, stem + ext))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return explicit or (cands[0] if cands else None)


def _manifest_entries(manifest_path, split):
    """Entries of ``split`` from a JSON / JSONL manifest.

    Accepted layouts: {"splits": {name: [entry...]}} (as written by
    scripts/prepare_colab.py, entries {"file", "image", "source", "group",
    ...} relative to <manifest dir>/<split>/{annotations,images}),
    {name: [entry...]}, or a list (JSON or JSONL) of entries carrying a
    "split" field. Other entries are an
    annotation path string or a dict with an annotation path (one of
    annotation/annotation_path/ann_path/ann/json), an optional image path
    (image/image_file/img_path), optional source/group, or the annotation
    itself (a dict with "keypoints").
    """
    with open(manifest_path, encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        splits = data.get("splits", data)
        if split not in splits:
            raise KeyError(f"split {split!r} not in manifest (have: {sorted(splits)})")
        return splits[split]
    return [e for e in data if isinstance(e, dict) and e.get("split") == split]


def load_items(annotations_dir=None, images_dir=None, manifest=None, split=None):
    """List of eval items: dicts with name, ann_path, image_path, ann,
    source, group."""
    items = []
    if manifest:
        if not split:
            raise ValueError("--split is required with --manifest")
        mdir = os.path.dirname(os.path.abspath(manifest))
        bases = [mdir, os.getcwd()]
        for entry in _manifest_entries(manifest, split):
            ann_path, img_path, extra = None, None, {}
            if isinstance(entry, str):
                ann_path = _resolve(entry, bases)
            elif entry.get("file"):
                # scripts/prepare_colab.py layout: the built (canonicalized)
                # copies live in <manifest dir>/<split>/{annotations,images}/.
                # Never fall back to src_annotation (raw, pre-canonicalization).
                extra = entry
                ann_path = os.path.join(mdir, split, "annotations", entry["file"])
                if entry.get("image"):
                    img_path = os.path.join(images_dir or os.path.join(mdir, split, "images"),
                                            entry["image"])
            else:
                extra = entry
                for k in _ANN_KEYS:
                    if entry.get(k):
                        ann_path = _resolve(entry[k], bases)
                        break
                for k in _IMG_KEYS:
                    v = entry.get(k)
                    if v and not str(v).endswith(".json"):
                        img_path = _resolve(v, bases + ([images_dir] if images_dir else []))
                        break
            if ann_path:
                with open(ann_path, encoding="utf-8") as f:
                    ann = json.load(f)
            elif isinstance(entry, dict) and "keypoints" in entry:
                ann = entry
            else:
                raise ValueError(f"manifest entry without annotation: {entry!r}")
            for k in ("source", "group"):
                if extra.get(k) and not ann.get(k):
                    ann = dict(ann, **{k: extra[k]})
            img = _find_image(ann, ann_path, images_dir, img_path)
            items.append(_make_item(ann, ann_path, img))
    else:
        names = sorted(f for f in os.listdir(annotations_dir) if f.endswith(".json"))
        for fname in names:
            ann_path = os.path.join(annotations_dir, fname)
            with open(ann_path, encoding="utf-8") as f:
                ann = json.load(f)
            items.append(_make_item(ann, ann_path, _find_image(ann, ann_path, images_dir)))
    return items


def _make_item(ann, ann_path, image_path):
    name = os.path.basename(image_path or ann.get("image_path", "") or ann_path or "")
    source = infer_source(ann, name)
    return {
        "name": name,
        "ann_path": ann_path,
        "image_path": image_path,
        "ann": ann,
        "source": source,
        "group": infer_group(ann, name, source),
    }


# ---------------------------------------------------------------------------
# Segmentation helper
# ---------------------------------------------------------------------------

def letterbox_info(h, w, size):
    """LetterboxInfo for an (h, w) image, identical to ``letterbox``."""
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return LetterboxInfo(scale, (size - new_w) // 2, (size - new_h) // 2, w, h, size)


def seg_iou_letterbox(seg_mask, gt_kps, gt_vis, image_hw, size=REF_LONG_SIDE, thickness=DEFAULT_SEG_THICKNESS):
    """Seg IoU in the letterbox canvas (see metrics docstring), or None.

    ``seg_mask`` may be canvas-shaped (any square size, resized to ``size``)
    or original-image-shaped (letterboxed here). Probabilities or {0,1}/255.
    """
    if seg_mask is None:
        return None
    seg = np.asarray(seg_mask, dtype=np.float32).squeeze()
    if seg.ndim != 2:
        return None
    if seg.max() > 1.0:
        seg = seg / 255.0
    h, w = image_hw
    info = letterbox_info(h, w, size)
    if seg.shape == (h, w) and h != w:
        seg, _ = letterbox(seg, size)
    elif seg.shape[0] == seg.shape[1]:
        if seg.shape[0] != size:
            seg = cv2.resize(seg, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        return None
    gt_canvas = keypoints_to_letterbox(np.asarray(gt_kps, dtype=np.float64), info)
    gt_mask = render_line_mask(gt_canvas, gt_vis, size, thickness)
    cm = content_mask(info).astype(bool)
    iou = segmentation_iou(seg[cm], gt_mask[cm], threshold=0.5, empty_value=float("nan"))
    return None if (iou is None or math.isnan(iou)) else iou


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(items, predict_fn, use_raw=False, seg_size=REF_LONG_SIDE,
                   seg_thickness=DEFAULT_SEG_THICKNESS, n_boot=1000, seed=0, progress=True):
    """Evaluate ``predict_fn`` on ``items`` with the shared metric code.

    ``predict_fn(bgr_image)`` returns an object (or dict) following the
    CourtDetection contract: keypoints (30,2) normalized original coords,
    visibility (30,) probs, optional raw_keypoints / homography /
    inlier_mask / seg_mask.

    Returns (rows, summary).
    """
    rows = []
    skipped = []
    t0 = time.time()
    for i, item in enumerate(items):
        image = cv2.imread(item["image_path"]) if item["image_path"] else None
        if image is None:
            skipped.append(item["name"])
            continue
        h, w = image.shape[:2]
        ann = item["ann"]
        if "image_size" in ann and list(ann["image_size"]) != [h, w]:
            print(f"warning: {item['name']}: annotation image_size {ann['image_size']} != "
                  f"loaded [{h}, {w}]", file=sys.stderr)
        gt_kps = np.asarray(ann["keypoints"], dtype=np.float64)
        gt_vis = np.asarray(ann["visibility"], dtype=np.float64)

        det = predict_fn(image)
        get = (lambda k, d=None, det=det: det.get(k, d)) if isinstance(det, dict) else \
            (lambda k, d=None, det=det: getattr(det, k, d))
        kps = get("raw_keypoints") if use_raw else None
        if kps is None:
            kps = get("keypoints")
        pred_vis = get("visibility")

        extra = {"name": item["name"], "source": item["source"], "group": item["group"]}
        inl = get("inlier_mask")
        Hm = get("homography")
        extra["model_homography"] = Hm is not None
        extra["model_inliers"] = int(np.sum(inl)) if inl is not None else None
        seg = seg_iou_letterbox(get("seg_mask"), gt_kps, gt_vis, (h, w), seg_size, seg_thickness)
        extra["seg_iou"] = seg
        rows.append(evaluate_image(np.asarray(kps, dtype=np.float64), pred_vis, gt_kps, gt_vis,
                                   (h, w), **extra))
        if progress and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)} images ({time.time() - t0:.0f}s)", file=sys.stderr)

    summary = aggregate(rows, n_boot=n_boot, seed=seed)
    summary["n_skipped"] = len(skipped)
    summary["skipped"] = skipped
    return rows, summary


def evaluate(checkpoint_path, annotations_dir=None, images_dir=None, image_size=None,
             device=None, use_tta=False, refine=True, manifest=None, split=None,
             n_boot=1000, seed=0, seg_thickness=None, predictor=None, out=None):
    """Run full evaluation; returns the summary dict (see metrics.aggregate).

    ``predictor`` may be passed directly (anything with ``.predict(img)``);
    otherwise a CourtPredictor is built from ``checkpoint_path``.
    ``seg_thickness`` defaults to the checkpoint config's ``line_thickness``
    (the training target thickness), else DEFAULT_SEG_THICKNESS.
    """
    if predictor is None:
        from src.inference.predict import CourtPredictor
        kwargs = {"device": device}
        if image_size is not None:
            kwargs["image_size"] = image_size
        predictor = CourtPredictor(checkpoint_path, **kwargs)

    if seg_thickness is None:
        info = getattr(predictor, "info", None) or {}
        config = info.get("config") or {}
        seg_thickness = int(config.get("line_thickness", DEFAULT_SEG_THICKNESS))
    if image_size is None:
        image_size = getattr(predictor, "image_size", None)

    if use_tta:
        if hasattr(predictor, "predict_tta"):
            predict_fn = predictor.predict_tta
        else:
            raise RuntimeError("predictor has no predict_tta(); TTA unsupported")
    else:
        predict_fn = predictor.predict

    items = load_items(annotations_dir, images_dir, manifest, split)
    rows, summary = run_evaluation(
        items, predict_fn, use_raw=not refine, seg_size=image_size or REF_LONG_SIDE,
        seg_thickness=seg_thickness, n_boot=n_boot, seed=seed,
    )
    summary["meta"] = {
        "checkpoint": checkpoint_path,
        "annotations": annotations_dir,
        "images": images_dir,
        "manifest": manifest,
        "split": split,
        "refine": refine,
        "tta": use_tta,
        "seg_thickness": seg_thickness,
        "ref_long_side": REF_LONG_SIDE,
    }
    if out:
        write_outputs(out, rows, summary)
    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _clean(obj):
    """Recursively replace NaN/inf floats for strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def write_outputs(out_path, rows, summary):
    """Write summary JSON, per-image JSONL rows and the markdown tables."""
    stem, _ = os.path.splitext(out_path)
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(_clean(summary), f, indent=2)
    with open(stem + "_rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_clean(r)) + "\n")
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(markdown_report(summary))
    return stem + ".json", stem + "_rows.jsonl", stem + ".md"


_PCT = {"pck", "kla", "auc", "pcr", "rate", "f1", "iou"}


def _fmt(label, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, str):
        return v
    if any(p in label for p in _PCT):
        return f"{100 * v:.1f}"
    if "err_m" in label:
        return f"{v:.3f}"
    return f"{v:.2f}"


def markdown_report(summary, title="CourtVisionNet evaluation"):
    """Markdown tables: headline with 95% CI, breakdowns by source/group and
    by number of visible keypoints, per-keypoint table."""
    hl = summary.get("headline", {})
    ci = summary.get("ci", {})
    st = summary.get("strict", {})
    lines = [f"## {title}", ""]
    lines.append(
        f"Images: {summary.get('n_images', 0)} (with GT-visible points: "
        f"{st.get('n_images_with_visible', 0)}, none visible: {st.get('n_images_no_visible', 0)}, "
        f"unreadable: {summary.get('n_skipped', 0)}); scored points: {st.get('n_points', 0)}. "
        f"Percentages in %, px at {REF_LONG_SIDE}-px long side, metres on the court plane."
    )
    lines += ["", "| Metric | Value | 95% CI |", "|---|---|---|"]
    for label, _ in HEADLINE_METRICS:
        lo_hi = ci.get(label)
        ci_s = f"[{_fmt(label, lo_hi[0])}, {_fmt(label, lo_hi[1])}]" if lo_hi else ""
        lines.append(f"| {label} | {_fmt(label, hl.get(label))} | {ci_s} |")

    cols = ["n_images", "pck@10px", "pck_norm@0.02", "auc_norm", "sym_pck_norm@0.02",
            "err_ref_median", "homography_success_rate", "court_iou_mean", "vis_f1"]
    for key, name in (("by_source", "Source"), ("by_group", "Group"), ("by_n_visible", "# visible GT")):
        block = summary.get(key)
        if not block:
            continue
        lines += ["", f"| {name} | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for g, vals in block.items():
            cells = [str(vals.get("n_images", 0)) if c == "n_images" else _fmt(c, vals.get(c))
                     for c in cols]
            lines.append(f"| {g} | " + " | ".join(cells) + " |")

    per_kp = summary.get("by_keypoint")
    if per_kp:
        kcols = ["n", "err_ref_median", "pck@10px", "pck_norm@0.02", "sym_pck_norm@0.02",
                 "vis_recall"]
        lines += ["", "| Keypoint | " + " | ".join(kcols) + " |", "|---" * (len(kcols) + 1) + "|"]
        for r in per_kp:
            k = r["index"]
            cells = [str(r["n"]) if c == "n" else _fmt(c if c != "vis_recall" else "rate", r.get(c))
                     for c in kcols]
            lines.append(f"| K{k} {KEYPOINT_NAMES[k]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def print_summary(summary):
    """Print the markdown report."""
    print(markdown_report(summary))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate CourtVisionNet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations", help="annotation directory (ignored with --manifest)")
    parser.add_argument("--images", help="image directory")
    parser.add_argument("--manifest", help="split manifest (JSON/JSONL)")
    parser.add_argument("--split", help="split name inside the manifest, e.g. test_real")
    parser.add_argument("--out", help="summary path X.json (also writes X_rows.jsonl, X.md)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="model input size (default: predictor's own)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tta", action="store_true", help="test-time augmentation")
    parser.add_argument("--no-refine", action="store_true",
                        help="score raw decoder keypoints instead of homography-refined ones")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seg-thickness", type=int, default=None,
                        help="GT line thickness (canvas px); default: the checkpoint's "
                             "training line_thickness")
    args = parser.parse_args(argv)
    if not args.manifest and not args.annotations:
        parser.error("--annotations or --manifest is required")

    summary = evaluate(
        args.checkpoint, args.annotations, args.images, image_size=args.image_size,
        device=args.device, use_tta=args.tta, refine=not args.no_refine,
        manifest=args.manifest, split=args.split, n_boot=args.n_boot, seed=args.seed,
        seg_thickness=args.seg_thickness, out=args.out,
    )
    print_summary(summary)
    if args.out:
        print(f"wrote {os.path.splitext(args.out)[0]}.json/_rows.jsonl/.md")


if __name__ == "__main__":
    main()

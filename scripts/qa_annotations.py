"""Quality report for CVN annotation directories.

Per annotation file: handedness / permutation to the canonical labeling,
homography reprojection residual (pixels, original resolution), number of
visible keypoints, and declared image_size vs the actual image. Ends with a
summary per source.

Usage:
    python scripts/qa_annotations.py                       # hand labels, Roboflow, Blender
    python scripts/qa_annotations.py data/annotations --images data/images
    python scripts/qa_annotations.py data/colab_data_v2/test_real/annotations --csv qa.csv
    python scripts/qa_annotations.py --all-files           # print every file, not just flagged

A file is FLAGGED if it is not canonical, its orientation is undeterminable,
its residual exceeds --max-residual, or its image is missing / differently sized.
"""

import argparse
import collections
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_manifest import path_basename, qa_annotation, read_json  # noqa: E402

DEFAULT_DIRS = [
    os.path.join(ROOT, "data", "annotations"),
    os.path.join(ROOT, "data", "cvn_dataset", "train", "annotations"),
    os.path.join(ROOT, "data", "cvn_dataset", "valid", "annotations"),
    os.path.join(ROOT, "data", "cvn_dataset", "test", "annotations"),
    os.path.join(ROOT, "data", "blender", "annotations"),
]


def default_image_dir(ann_dir):
    """Sibling images/ dir of an annotations/ dir (data/annotations -> data/images)."""
    parent = os.path.dirname(os.path.abspath(ann_dir))
    return os.path.join(parent, "images")


def infer_source(ann, ann_dir):
    if ann.get("source"):
        return ann["source"]
    p = os.path.abspath(ann_dir).replace("\\", "/")
    if "/blender" in p:
        return "blender"
    if "cvn_dataset" in p:
        return "roboflow"
    return "manual"


def flags_for(rec, max_residual):
    flags = []
    perm = rec["perm_to_canonical"]
    if perm is None:
        flags.append("orientation_undeterminable")
    elif perm != "identity":
        flags.append(f"not_canonical({perm})")
    if rec["residual_max"] is not None and rec["residual_max"] > max_residual:
        flags.append(f"residual>{max_residual:g}px")
    if rec["size_status"] not in ("ok", "unchecked"):
        flags.append(f"image_{rec['size_status']}")
    return flags


def qa_directory(ann_dir, img_dir=None, max_residual=15.0, check_images=True):
    """QA every *.json in ann_dir. Returns a list of row dicts."""
    img_dir = img_dir or default_image_dir(ann_dir)
    rows = []
    for f in sorted(os.listdir(ann_dir)):
        if not f.endswith(".json"):
            continue
        ann = read_json(os.path.join(ann_dir, f))
        if "keypoints" not in ann or "visibility" not in ann:
            continue
        img = os.path.join(img_dir, path_basename(ann.get("image_path", ""))) if check_images else None
        rec = qa_annotation(ann, img)
        rec["file"] = os.path.join(ann_dir, f)
        rec["source"] = infer_source(ann, ann_dir)
        rec["flags"] = flags_for(rec, max_residual)
        rows.append(rec)
    return rows


def summarize(rows, max_residual):
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    lines = [f"{'source':<10}{'files':>7}{'canon':>7}{'rot':>5}{'mir':>5}{'undet':>7}"
             f"{'res_p50':>9}{'res_p95':>9}{'res_max':>9}{f'>{max_residual:g}px':>8}{'sizeBad':>9}{'flagged':>9}"]
    for src in sorted(by_src):
        rs = by_src[src]
        perms = collections.Counter(r["perm_to_canonical"] for r in rs)
        res = np.array([r["residual_max"] for r in rs if r["residual_max"] is not None])
        p50, p95, mx = (np.percentile(res, 50), np.percentile(res, 95), res.max()) if len(res) else (0, 0, 0)
        lines.append(
            f"{src:<10}{len(rs):>7}{perms.get('identity', 0):>7}{perms.get('rot180', 0):>5}"
            f"{perms.get('col_flip', 0) + perms.get('row_flip', 0):>5}{perms.get(None, 0):>7}"
            f"{p50:>9.1f}{p95:>9.1f}{mx:>9.1f}{int((res > max_residual).sum()):>8}"
            f"{sum(r['size_status'] not in ('ok', 'unchecked') for r in rs):>9}"
            f"{sum(bool(r['flags']) for r in rs):>9}")
    lines.append("(canon = already canonical, rot = needs rot180, mir = mirrored labeling, "
                 "undet = orientation undeterminable)")
    return "\n".join(lines)


CSV_FIELDS = ["file", "source", "n_visible", "perm_to_canonical", "handedness", "row_alignment",
              "residual_max", "residual_rms", "worst_kp", "image_size", "actual_size", "size_status", "flags"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="QA report for CVN annotations")
    parser.add_argument("dirs", nargs="*", help="annotation directories (default: all data sources)")
    parser.add_argument("--images", default=None,
                        help="image directory (default: sibling images/ of each annotations dir)")
    parser.add_argument("--max-residual", type=float, default=15.0)
    parser.add_argument("--no-image-check", action="store_true", help="skip reading image headers")
    parser.add_argument("--all-files", action="store_true", help="print every file, not only flagged")
    parser.add_argument("--csv", default=None, help="write per-file results to this CSV")
    args = parser.parse_args(argv)

    dirs = args.dirs or [d for d in DEFAULT_DIRS if os.path.isdir(d)]
    rows = []
    for d in dirs:
        rows += qa_directory(d, args.images, args.max_residual, not args.no_image_check)

    print(f"{'file':<60} {'src':<9}{'vis':>4} {'perm':<9}{'res_max':>8}  {'size':<11} flags")
    for r in rows:
        if r["flags"] or args.all_files:
            name = os.path.relpath(r["file"], ROOT) if r["file"].startswith(ROOT) else r["file"]
            res = "-" if r["residual_max"] is None else f"{r['residual_max']:.1f}"
            print(f"{name[-60:]:<60} {r['source']:<9}{r['n_visible']:>4} {str(r['perm_to_canonical']):<9}"
                  f"{res:>8}  {r['size_status']:<11} {', '.join(r['flags'])}")
    print()
    print(summarize(rows, args.max_residual))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: (";".join(r[k]) if k == "flags" else r.get(k)) for k in CSV_FIELDS})
        print(f"\nWrote {args.csv}")
    return rows


if __name__ == "__main__":
    main()

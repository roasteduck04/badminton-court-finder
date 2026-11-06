"""Convert Roboflow COCO keypoint export to CourtVisionNet annotation format.

Remaps new-point-0..29 to K0..K29, canonicalizes the labels (see
src.court_geometry.canonicalize) and writes one JSON per image in the format
expected by src.training.dataset.CourtDataset.

COCO visibility flags: v=2 (labeled, visible) -> visibility 1; v=1 (labeled,
not visible) -> visibility 1 with ``occluded`` True (location is known);
v=0 -> not labeled. If an image has several annotations, the one with the most
labeled keypoints is used (with a warning). ``group`` is the source video
(filename prefix before ``_image_``) and is what leak-free splits group by.

Usage:
    python src/tools/coco_to_cvn.py <coco_dir> <output_dir>

Example:
    python src/tools/coco_to_cvn.py data/roboflow_export data/cvn_dataset_v2

This creates (Roboflow's own random split is mirrored but should NOT be used
for evaluation -- scripts/prepare_colab.py pools it and splits by video):
    <output_dir>/
        train/ annotations/ images/
        valid/ annotations/ images/
        test/  annotations/ images/
"""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_manifest import canonicalize_annotation, roboflow_group  # noqa: E402

NP_TO_K = {
    0: 25, 1: 0, 2: 4, 3: 29, 4: 20, 5: 15, 6: 10, 7: 5,
    8: 26, 9: 27, 10: 28, 11: 21, 12: 22, 13: 23, 14: 24,
    15: 19, 16: 14, 17: 9, 18: 17, 19: 18, 20: 13, 21: 8,
    22: 3, 23: 1, 24: 2, 25: 7, 26: 6, 27: 11, 28: 12, 29: 16,
}

NK = 30


def _labeled_count(ann):
    kps = ann.get("keypoints") or []
    return sum(1 for i in range(2, len(kps), 3) if kps[i] > 0)


def pick_court_annotation(anns, file_name="", warn=True):
    """Pick the court annotation for an image: the one with most labeled keypoints.

    Annotations without a full 30-keypoint list are ignored. Returns None if
    none qualifies.
    """
    cands = [a for a in anns if len(a.get("keypoints") or []) == NK * 3]
    if not cands:
        return None
    best = max(cands, key=_labeled_count)
    if len(cands) > 1 and warn:
        print(f"  WARNING: {file_name}: {len(cands)} keypoint annotations, "
              f"using the one with {_labeled_count(best)} labeled points")
    return best


def coco_keypoints_to_cvn(flat, width, height, remap=True):
    """Flat COCO [x, y, v]*30 (pixels) -> (keypoints, visibility, occluded).

    ``remap`` applies NP_TO_K (Roboflow new-point order -> K order).
    """
    keypoints = [[-1.0, -1.0] for _ in range(NK)]
    visibility = [0] * NK
    occluded = [False] * NK
    for src in range(NK):
        x, y, v = flat[src * 3], flat[src * 3 + 1], flat[src * 3 + 2]
        k = NP_TO_K[src] if remap else src
        if v > 0 and (x > 0 or y > 0):
            keypoints[k] = [x / width, y / height]
            visibility[k] = 1
            occluded[k] = v == 1
    return keypoints, visibility, occluded


def coco_image_to_cvn(img_info, anns, remap=True, warn=True):
    """Build a canonical CVN annotation for one COCO image, or None."""
    fname = img_info["file_name"]
    ann = pick_court_annotation(anns, fname, warn=warn)
    if ann is None:
        return None
    iw, ih = img_info["width"], img_info["height"]
    keypoints, visibility, occluded = coco_keypoints_to_cvn(ann["keypoints"], iw, ih, remap)
    vis_kps = [i for i in range(NK) if visibility[i]]
    if len(vis_kps) < 2:
        return None
    xs = [keypoints[i][0] for i in vis_kps]
    ys = [keypoints[i][1] for i in vis_kps]
    cvn = {
        "image_path": os.path.basename(fname),
        "image_size": [ih, iw],
        "keypoints": keypoints,
        "visibility": visibility,
        "bounding_box": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
        "source": "roboflow",
        "group": roboflow_group(fname),
    }
    if any(occluded):
        cvn["occluded"] = occluded
    cvn, _ = canonicalize_annotation(cvn)
    return cvn


def convert_split(coco_dir, output_dir, split):
    coco_dir, output_dir = Path(coco_dir), Path(output_dir)
    json_path = coco_dir / split / "_annotations.coco.json"
    if not json_path.exists():
        print(f"  Skipping {split}: no COCO JSON found")
        return 0

    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    img_lookup = {img["id"]: img for img in coco["images"]}

    ann_by_image = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    ann_dir = output_dir / split / "annotations"
    img_dir = output_dir / split / "images"
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    undetermined = 0
    for img_id, img_info in img_lookup.items():
        fname = img_info["file_name"]
        src_img = coco_dir / split / fname
        if not src_img.exists():
            continue

        cvn_ann = coco_image_to_cvn(img_info, ann_by_image.get(img_id, []))
        if cvn_ann is None:
            continue
        if cvn_ann.get("canonical_perm") is None:
            undetermined += 1

        stem = Path(fname).stem
        with open(ann_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(cvn_ann, f, indent=2)

        shutil.copy2(src_img, img_dir / os.path.basename(fname))
        count += 1

    if undetermined:
        print(f"  {undetermined} annotation(s) with undeterminable orientation (canonical_perm null)")
    return count


def main():
    if len(sys.argv) < 3:
        print("Usage: python coco_to_cvn.py <coco_dir> <output_dir>")
        print("  coco_dir: directory with train/valid/test COCO exports")
        print("  output_dir: where to write CVN-format dataset")
        sys.exit(1)

    coco_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    total = 0
    for split in ["train", "valid", "test"]:
        print(f"Converting {split}...")
        n = convert_split(coco_dir, output_dir, split)
        print(f"  {n} images converted")
        total += n

    print(f"\nDone: {total} total images in {output_dir}")
    print("\nBuild leak-free training splits with:")
    print("  python scripts/prepare_colab.py")


if __name__ == "__main__":
    main()

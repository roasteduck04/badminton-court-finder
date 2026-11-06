"""Convert the Roboflow badminton-court-dataset annotations to K0-K29 scheme.

Reads the COCO export, remaps new-point-X to K-Y keypoints matching the
CourtVisionNet annotator layout, and canonicalizes each annotation's labels
(src.court_geometry.canonicalize; COCO v flags are kept, so v=1 "labeled but
not visible" points stay labeled). Images with several keypoint annotations
keep only the one with the most labeled keypoints (with a warning).

Usage:
    python src/tools/convert_dataset.py <coco_json_dir> <output_dir>

Example:
    python src/tools/convert_dataset.py data/roboflow_export data/converted
"""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.court_geometry import SYMMETRY_PERMS, canonicalize  # noqa: E402
from src.data_manifest import roboflow_group  # noqa: E402
from src.tools.coco_to_cvn import NP_TO_K, pick_court_annotation  # noqa: E402

K_NAMES = [
    "K0-BaselineL-DblTop", "K1-BaselineL-SglTop", "K2-BaselineL-Center",
    "K3-BaselineL-SglBot", "K4-BaselineL-DblBot",
    "K5-LongSvcL-DblTop", "K6-LongSvcL-SglTop", "K7-LongSvcL-Center",
    "K8-LongSvcL-SglBot", "K9-LongSvcL-DblBot",
    "K10-ShortSvcL-DblTop", "K11-ShortSvcL-SglTop", "K12-ShortSvcL-Center",
    "K13-ShortSvcL-SglBot", "K14-ShortSvcL-DblBot",
    "K15-ShortSvcR-DblTop", "K16-ShortSvcR-SglTop", "K17-ShortSvcR-Center",
    "K18-ShortSvcR-SglBot", "K19-ShortSvcR-DblBot",
    "K20-LongSvcR-DblTop", "K21-LongSvcR-SglTop", "K22-LongSvcR-Center",
    "K23-LongSvcR-SglBot", "K24-LongSvcR-DblBot",
    "K25-BaselineR-DblTop", "K26-BaselineR-SglTop", "K27-BaselineR-Center",
    "K28-BaselineR-SglBot", "K29-BaselineR-DblBot",
]

CL, CW = 13.4, 6.1
NP, SS, SO, LS = 6.7, 1.98, 0.46, 0.76
K_SKELETON = [
    [0, 5], [5, 10], [10, 15], [15, 20], [20, 25],
    [4, 9], [9, 14], [14, 19], [19, 24], [24, 29],
    [1, 6], [6, 11], [11, 16], [16, 21], [21, 26],
    [3, 8], [8, 13], [13, 18], [18, 23], [23, 28],
    [2, 7], [7, 12], [17, 22], [22, 27], [12, 17],
    [0, 1], [1, 2], [2, 3], [3, 4],
    [25, 26], [26, 27], [27, 28], [28, 29],
    [5, 6], [6, 7], [7, 8], [8, 9],
    [20, 21], [21, 22], [22, 23], [23, 24],
    [10, 11], [11, 12], [12, 13], [13, 14],
    [15, 16], [16, 17], [17, 18], [18, 19],
]


def canonicalize_coco_keypoints(flat, width, height):
    """Canonicalize a flat K-ordered COCO keypoint list [x, y, v]*30 (pixels).

    Returns (new_flat, perm_name); perm_name None = orientation undeterminable
    (list returned unchanged).
    """
    kps = [[flat[i * 3] / width, flat[i * 3 + 1] / height] for i in range(30)]
    vis = [1 if flat[i * 3 + 2] > 0 and (flat[i * 3] > 0 or flat[i * 3 + 1] > 0) else 0
           for i in range(30)]
    _, _, perm_name = canonicalize(kps, vis, [height, width])
    if perm_name is None:
        return list(flat), None
    perm = SYMMETRY_PERMS[perm_name]
    out = []
    for k in range(30):
        out.extend(flat[perm[k] * 3: perm[k] * 3 + 3])
    return out, perm_name


def convert_coco(input_path, output_path):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    new_cats = []
    for cat in data["categories"]:
        if "keypoints" in cat:
            cat = dict(cat)
            cat["keypoints"] = K_NAMES
            cat["skeleton"] = [[a + 1, b + 1] for a, b in K_SKELETON]
        new_cats.append(cat)
    data["categories"] = new_cats

    images = {img["id"]: img for img in data["images"]}
    by_image = {}
    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    kept, undetermined = [], 0
    for image_id, anns in by_image.items():
        img = images.get(image_id, {})
        kp_anns = [a for a in anns if len(a.get("keypoints") or []) == 90]
        others = [a for a in anns if len(a.get("keypoints") or []) != 90]
        ann = pick_court_annotation(kp_anns, img.get("file_name", str(image_id)))
        kept.extend(others)
        if ann is None:
            continue
        old_kps = ann["keypoints"]
        new_kps = [0.0] * 90
        for np_idx in range(30):
            k_idx = NP_TO_K[np_idx]
            new_kps[k_idx * 3] = old_kps[np_idx * 3]
            new_kps[k_idx * 3 + 1] = old_kps[np_idx * 3 + 1]
            new_kps[k_idx * 3 + 2] = old_kps[np_idx * 3 + 2]
        if img.get("width") and img.get("height"):
            new_kps, perm_name = canonicalize_coco_keypoints(new_kps, img["width"], img["height"])
        else:
            perm_name = None
        if perm_name is None:
            undetermined += 1
        ann["keypoints"] = new_kps
        ann["canonical_perm"] = perm_name
        if img.get("file_name"):
            ann["group"] = roboflow_group(img["file_name"])
        kept.append(ann)
    data["annotations"] = kept

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    n_ann = len(data["annotations"])
    n_img = len(data["images"])
    print(f"  Converted {n_ann} annotations across {n_img} images"
          + (f" ({undetermined} with undeterminable orientation)" if undetermined else ""))


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_dataset.py <input_dir> <output_dir>")
        print("  input_dir should contain train/, valid/, test/ with COCO JSON + images")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    for split in ["train", "valid", "test"]:
        split_in = input_dir / split
        split_out = output_dir / split
        json_in = split_in / "_annotations.coco.json"

        if not json_in.exists():
            print(f"Skipping {split}: no annotations found")
            continue

        print(f"Converting {split}...")
        split_out.mkdir(parents=True, exist_ok=True)
        convert_coco(str(json_in), str(split_out / "_annotations.coco.json"))

        for img in split_in.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                shutil.copy2(img, split_out / img.name)

        n_imgs = len(list(split_out.glob("*.jpg")) + list(split_out.glob("*.png")))
        print(f"  Copied {n_imgs} images")

    print(f"\nDone! Converted dataset at: {output_dir}")


if __name__ == "__main__":
    main()

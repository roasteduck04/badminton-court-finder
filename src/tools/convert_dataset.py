"""Convert the Roboflow badminton-court-dataset annotations to K0-K29 scheme.

Reads the COCO export and remaps new-point-X to K-Y keypoints
matching the CourtVisionNet annotator layout.

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

NP_TO_K = {
    0: 25, 1: 0, 2: 4, 3: 29, 4: 20, 5: 15, 6: 10, 7: 5,
    8: 26, 9: 27, 10: 28, 11: 21, 12: 22, 13: 23, 14: 24,
    15: 19, 16: 14, 17: 9, 18: 17, 19: 18, 20: 13, 21: 8,
    22: 3, 23: 1, 24: 2, 25: 7, 26: 6, 27: 11, 28: 12, 29: 16,
}

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


def convert_coco(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)

    new_cats = []
    for cat in data["categories"]:
        if "keypoints" in cat:
            cat = dict(cat)
            cat["keypoints"] = K_NAMES
            cat["skeleton"] = [[a + 1, b + 1] for a, b in K_SKELETON]
        new_cats.append(cat)
    data["categories"] = new_cats

    for ann in data["annotations"]:
        old_kps = ann["keypoints"]
        new_kps = [0.0] * 90
        for np_idx in range(30):
            k_idx = NP_TO_K[np_idx]
            new_kps[k_idx * 3] = old_kps[np_idx * 3]
            new_kps[k_idx * 3 + 1] = old_kps[np_idx * 3 + 1]
            new_kps[k_idx * 3 + 2] = old_kps[np_idx * 3 + 2]
        ann["keypoints"] = new_kps

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    n_ann = len(data["annotations"])
    n_img = len(data["images"])
    print(f"  Converted {n_ann} annotations across {n_img} images")


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

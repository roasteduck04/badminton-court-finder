"""Convert Roboflow COCO keypoint export to CourtVisionNet annotation format.

Remaps new-point-0..29 to K0..K29 and writes one JSON per image
in the format expected by src.training.dataset.CourtDataset.

Usage:
    python src/tools/coco_to_cvn.py <coco_dir> <output_dir>

Example:
    python src/tools/coco_to_cvn.py data/roboflow_export data/cvn_dataset

This creates:
    data/cvn_dataset/
        train/
            annotations/   (per-image JSON files)
            images/        (copied or symlinked images)
        valid/
            annotations/
            images/
        test/
            annotations/
            images/
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

NK = 30


def convert_split(coco_dir, output_dir, split):
    json_path = coco_dir / split / "_annotations.coco.json"
    if not json_path.exists():
        print(f"  Skipping {split}: no COCO JSON found")
        return 0

    with open(json_path) as f:
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
    for img_id, img_info in img_lookup.items():
        fname = img_info["file_name"]
        src_img = coco_dir / split / fname
        if not src_img.exists():
            continue

        anns = ann_by_image.get(img_id, [])
        if not anns:
            continue

        ann = anns[0]
        old_kps = ann["keypoints"]

        keypoints = [[-1.0, -1.0]] * NK
        visibility = [0] * NK

        iw = img_info["width"]
        ih = img_info["height"]

        for np_idx in range(NK):
            x = old_kps[np_idx * 3]
            y = old_kps[np_idx * 3 + 1]
            v = old_kps[np_idx * 3 + 2]

            k_idx = NP_TO_K[np_idx]

            if v > 0 and (x > 0 or y > 0):
                keypoints[k_idx] = [x / iw, y / ih]
                visibility[k_idx] = 1
            else:
                keypoints[k_idx] = [-1.0, -1.0]
                visibility[k_idx] = 0

        vis_kps = [i for i in range(NK) if visibility[i]]
        if len(vis_kps) < 2:
            continue

        xs = [keypoints[i][0] for i in vis_kps]
        ys = [keypoints[i][1] for i in vis_kps]
        bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

        stem = Path(fname).stem
        cvn_ann = {
            "image_path": fname,
            "image_size": [ih, iw],
            "keypoints": keypoints,
            "visibility": visibility,
            "bounding_box": bbox,
        }

        with open(ann_dir / f"{stem}.json", "w") as f:
            json.dump(cvn_ann, f, indent=2)

        shutil.copy2(src_img, img_dir / fname)
        count += 1

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
    print(f"\nTo train, update config paths or run:")
    print(f"  python -m src.tools.train_yolo --data {output_dir}")


if __name__ == "__main__":
    main()

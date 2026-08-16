"""Prepare all CVN data sources for Roboflow upload in COCO Keypoints format.

Merges three data sources:
  1. Your annotations (data/annotations/) — split 70/20/10
  2. Roboflow dataset (data/cvn_dataset/) — already split, kept as-is
  3. Blender synthetic (data/blender/) — added to training only

Output structure (ready for Roboflow upload):
    data/roboflow_upload/
        train/
            images/
            _annotations.coco.json
        valid/
            images/
            _annotations.coco.json
        test/
            images/
            _annotations.coco.json

Usage:
    python scripts/prepare_roboflow.py
    python scripts/prepare_roboflow.py --no-synthetic
    python scripts/prepare_roboflow.py --no-dataset
    python scripts/prepare_roboflow.py --split 80 10 10
"""

import argparse
import json
import os
import random
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.court_geometry import get_court_lines

REAL_ANN_DIR = os.path.join(ROOT, "data", "annotations")
REAL_IMG_DIR = os.path.join(ROOT, "data", "images")
BLENDER_ANN_DIR = os.path.join(ROOT, "data", "blender", "annotations")
BLENDER_IMG_DIR = os.path.join(ROOT, "data", "blender", "images")
DATASET_DIR = os.path.join(ROOT, "data", "cvn_dataset")
OUTPUT_DIR = os.path.join(ROOT, "data", "roboflow_upload")

NUM_KEYPOINTS = 30

SKELETON = [[s + 1, e + 1] for s, e in get_court_lines()]


def load_cvn_annotations(ann_dir, min_visible=4):
    annotations = []
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(ann_dir, fname)) as f:
            ann = json.load(f)
        if sum(ann.get("visibility", [])) >= min_visible:
            annotations.append(ann)
    return annotations


def cvn_to_coco_annotation(cvn_ann, ann_id, image_id):
    h, w = cvn_ann["image_size"]
    kps_flat = []
    num_visible = 0
    for i in range(NUM_KEYPOINTS):
        vis = cvn_ann["visibility"][i]
        if vis:
            x_norm, y_norm = cvn_ann["keypoints"][i]
            kps_flat.extend([round(x_norm * w, 2), round(y_norm * h, 2), 2])
            num_visible += 1
        else:
            kps_flat.extend([0, 0, 0])

    bbox = cvn_ann.get("bounding_box", [0, 0, 0, 0])
    bbox_px = [
        round(bbox[0] * w, 2),
        round(bbox[1] * h, 2),
        round(bbox[2] * w, 2),
        round(bbox[3] * h, 2),
    ]

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "keypoints": kps_flat,
        "num_keypoints": num_visible,
        "bbox": bbox_px,
        "area": round(bbox_px[2] * bbox_px[3], 2),
        "iscrowd": 0,
    }


def build_coco_entries(cvn_anns, img_dir, start_image_id=0, start_ann_id=0, prefix=""):
    images = []
    annotations = []
    image_id = start_image_id
    ann_id = start_ann_id

    for cvn_ann in cvn_anns:
        h, w = cvn_ann["image_size"]
        img_name = cvn_ann["image_path"]
        img_path = os.path.join(img_dir, img_name)
        if not os.path.isfile(img_path):
            print(f"  WARNING: Image not found, skipping: {img_path}")
            continue

        out_name = f"{prefix}{img_name}" if prefix else img_name
        images.append({
            "id": image_id,
            "file_name": out_name,
            "width": w,
            "height": h,
        })
        annotations.append(cvn_to_coco_annotation(cvn_ann, ann_id, image_id))
        image_id += 1
        ann_id += 1

    return images, annotations


def make_coco_json(images, annotations):
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "badminton_court",
            "supercategory": "sports",
            "keypoints": [f"K{i}" for i in range(NUM_KEYPOINTS)],
            "skeleton": SKELETON,
        }],
    }


def split_list(items, ratios, seed=42):
    random.seed(seed)
    shuffled = list(items)
    random.shuffle(shuffled)
    n = len(shuffled)
    i1 = round(n * ratios[0] / 100)
    i2 = round(n * (ratios[0] + ratios[1]) / 100)
    return shuffled[:i1], shuffled[i1:i2], shuffled[i2:]


def copy_images(cvn_anns, src_dir, dst_dir, prefix=""):
    os.makedirs(dst_dir, exist_ok=True)
    for ann in cvn_anns:
        src = os.path.join(src_dir, ann["image_path"])
        out_name = f"{prefix}{ann['image_path']}" if prefix else ann["image_path"]
        dst = os.path.join(dst_dir, out_name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)


def build_split(split_name, sources, output_dir):
    """Build one split from multiple (anns, img_dir, prefix) sources."""
    imgs_dir = os.path.join(output_dir, split_name, "images")
    all_images = []
    all_annotations = []

    for anns, img_dir, prefix, label in sources:
        if not anns:
            continue
        img_id_start = len(all_images)
        ann_id_start = len(all_annotations)
        images, annotations = build_coco_entries(
            anns, img_dir, img_id_start, ann_id_start, prefix
        )
        all_images.extend(images)
        all_annotations.extend(annotations)
        copy_images(anns, img_dir, imgs_dir, prefix)
        print(f"    {label}: {len(images)} images")

    coco = make_coco_json(all_images, all_annotations)
    json_path = os.path.join(output_dir, split_name, "_annotations.coco.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)

    return len(all_images)


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Roboflow upload")
    parser.add_argument("--split", nargs=3, type=int, default=[70, 20, 10],
                        help="Train/valid/test split for YOUR annotations (default: 70 20 10)")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Exclude Blender synthetic data")
    parser.add_argument("--no-dataset", action="store_true",
                        help="Exclude Roboflow cvn_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visible", type=int, default=4,
                        help="Minimum visible keypoints to include (default: 4)")
    args = parser.parse_args()

    assert sum(args.split) == 100, f"Split must sum to 100, got {sum(args.split)}"

    # 1. Load your annotations and split
    print("=== Your Annotations ===")
    real_anns = load_cvn_annotations(REAL_ANN_DIR, args.min_visible)
    print(f"  {len(real_anns)} images (>= {args.min_visible} visible keypoints)")
    train_real, valid_real, test_real = split_list(real_anns, args.split, args.seed)
    print(f"  Split: {len(train_real)} train / {len(valid_real)} valid / {len(test_real)} test")

    # 2. Load Roboflow dataset (already split)
    ds_train = ds_valid = ds_test = []
    if not args.no_dataset:
        print("\n=== Roboflow Dataset (cvn_dataset) ===")
        for split_name, var_name in [("train", "ds_train"), ("valid", "ds_valid"), ("test", "ds_test")]:
            ann_dir = os.path.join(DATASET_DIR, split_name, "annotations")
            if os.path.isdir(ann_dir):
                loaded = load_cvn_annotations(ann_dir, args.min_visible)
                print(f"  {split_name}: {len(loaded)} images")
                if var_name == "ds_train":
                    ds_train = loaded
                elif var_name == "ds_valid":
                    ds_valid = loaded
                else:
                    ds_test = loaded

    # 3. Load Blender synthetic (train only)
    synth_anns = []
    if not args.no_synthetic and os.path.isdir(BLENDER_ANN_DIR):
        print("\n=== Blender Synthetic ===")
        synth_anns = load_cvn_annotations(BLENDER_ANN_DIR, args.min_visible)
        print(f"  {len(synth_anns)} images (train only)")

    # Clean output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    ds_img_dirs = {
        "train": os.path.join(DATASET_DIR, "train", "images"),
        "valid": os.path.join(DATASET_DIR, "valid", "images"),
        "test": os.path.join(DATASET_DIR, "test", "images"),
    }

    # Build splits
    print("\n=== Building Splits ===")

    print("  TRAIN:")
    n_train = build_split("train", [
        (train_real, REAL_IMG_DIR, "mine_", "Your annotations"),
        (ds_train, ds_img_dirs["train"], "ds_", "Roboflow dataset"),
        (synth_anns, BLENDER_IMG_DIR, "bl_", "Blender synthetic"),
    ], OUTPUT_DIR)

    print("  VALID:")
    n_valid = build_split("valid", [
        (valid_real, REAL_IMG_DIR, "mine_", "Your annotations"),
        (ds_valid, ds_img_dirs["valid"], "ds_", "Roboflow dataset"),
    ], OUTPUT_DIR)

    print("  TEST:")
    n_test = build_split("test", [
        (test_real, REAL_IMG_DIR, "mine_", "Your annotations"),
        (ds_test, ds_img_dirs["test"], "ds_", "Roboflow dataset"),
    ], OUTPUT_DIR)

    # Summary
    print(f"\n{'='*50}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Total: {n_train + n_valid + n_test} images")
    print(f"  Train: {n_train}")
    print(f"  Valid: {n_valid}")
    print(f"  Test:  {n_test}")
    print(f"\nUpload each split folder to Roboflow as a COCO Keypoints project.")
    print(f"Each folder has images/ and _annotations.coco.json")


if __name__ == "__main__":
    main()

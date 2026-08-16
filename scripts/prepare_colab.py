"""Prepare all data sources for Colab training and zip for Google Drive upload.

Merges three data sources into a single train/valid/test structure:
  1. Your annotations (data/annotations/) — split 70/20/10
  2. Roboflow dataset (data/cvn_dataset/) — already split, kept as-is
  3. Blender synthetic (data/blender/) — added to training only

Output:
    data/colab_data/
        train/annotations/*.json   train/images/*
        valid/annotations/*.json   valid/images/*
        test/annotations/*.json    test/images/*
    data/colab_data.zip  (ready for Google Drive upload)

Usage:
    python scripts/prepare_colab.py
    python scripts/prepare_colab.py --no-synthetic
    python scripts/prepare_colab.py --split 80 10 10
"""

import argparse
import json
import os
import random
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_ANN_DIR = os.path.join(ROOT, "data", "annotations")
REAL_IMG_DIR = os.path.join(ROOT, "data", "images")
BLENDER_ANN_DIR = os.path.join(ROOT, "data", "blender", "annotations")
BLENDER_IMG_DIR = os.path.join(ROOT, "data", "blender", "images")
DATASET_DIR = os.path.join(ROOT, "data", "cvn_dataset")
OUTPUT_DIR = os.path.join(ROOT, "data", "colab_data")
ZIP_PATH = os.path.join(ROOT, "data", "colab_data.zip")


def load_cvn_annotations(ann_dir, min_visible=4):
    annotations = []
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(ann_dir, fname)) as f:
            ann = json.load(f)
        if sum(ann.get("visibility", [])) >= min_visible:
            annotations.append((fname, ann))
    return annotations


def copy_source(items, src_ann_dir, src_img_dir, dst_ann_dir, dst_img_dir, prefix=""):
    """Copy annotation+image pairs, prefixing filenames to avoid collisions."""
    os.makedirs(dst_ann_dir, exist_ok=True)
    os.makedirs(dst_img_dir, exist_ok=True)
    copied = 0
    for fname, ann in items:
        img_name = ann["image_path"]
        src_img = os.path.join(src_img_dir, img_name)
        if not os.path.isfile(src_img):
            print(f"  WARNING: Image not found, skipping: {src_img}")
            continue

        out_img_name = f"{prefix}{img_name}" if prefix else img_name
        out_ann_name = f"{prefix}{fname}" if prefix else fname

        ann_copy = dict(ann)
        ann_copy["image_path"] = out_img_name

        with open(os.path.join(dst_ann_dir, out_ann_name), "w") as f:
            json.dump(ann_copy, f, indent=2)
        shutil.copy2(src_img, os.path.join(dst_img_dir, out_img_name))
        copied += 1
    return copied


def split_list(items, ratios, seed=42):
    random.seed(seed)
    shuffled = list(items)
    random.shuffle(shuffled)
    n = len(shuffled)
    i1 = round(n * ratios[0] / 100)
    i2 = round(n * (ratios[0] + ratios[1]) / 100)
    return shuffled[:i1], shuffled[i1:i2], shuffled[i2:]


def zip_directory(dir_path, zip_path):
    print(f"\nZipping to {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(dir_path):
            for file in files:
                abs_path = os.path.join(root, file)
                arc_name = os.path.relpath(abs_path, os.path.dirname(dir_path))
                zf.write(abs_path, arc_name)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Colab training")
    parser.add_argument("--split", nargs=3, type=int, default=[70, 20, 10],
                        help="Train/valid/test split for YOUR annotations (default: 70 20 10)")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Exclude Blender synthetic data")
    parser.add_argument("--no-dataset", action="store_true",
                        help="Exclude Roboflow cvn_dataset")
    parser.add_argument("--no-zip", action="store_true",
                        help="Skip creating zip file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-visible", type=int, default=4,
                        help="Minimum visible keypoints to include (default: 4)")
    args = parser.parse_args()

    assert sum(args.split) == 100, f"Split must sum to 100, got {sum(args.split)}"

    # Clean output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # 1. Your annotations — split
    print("=== Your Annotations ===")
    real_items = load_cvn_annotations(REAL_ANN_DIR, args.min_visible)
    print(f"  {len(real_items)} images (>= {args.min_visible} visible keypoints)")
    train_real, valid_real, test_real = split_list(real_items, args.split, args.seed)
    print(f"  Split: {len(train_real)} train / {len(valid_real)} valid / {len(test_real)} test")

    totals = {"train": 0, "valid": 0, "test": 0}

    for split, items in [("train", train_real), ("valid", valid_real), ("test", test_real)]:
        n = copy_source(
            items, REAL_ANN_DIR, REAL_IMG_DIR,
            os.path.join(OUTPUT_DIR, split, "annotations"),
            os.path.join(OUTPUT_DIR, split, "images"),
            prefix="mine_",
        )
        totals[split] += n
        print(f"    {split}: {n} copied")

    # 2. Roboflow dataset — already split
    if not args.no_dataset and os.path.isdir(DATASET_DIR):
        print("\n=== Roboflow Dataset ===")
        for split in ["train", "valid", "test"]:
            ann_dir = os.path.join(DATASET_DIR, split, "annotations")
            img_dir = os.path.join(DATASET_DIR, split, "images")
            if not os.path.isdir(ann_dir):
                continue
            items = load_cvn_annotations(ann_dir, args.min_visible)
            n = copy_source(
                items, ann_dir, img_dir,
                os.path.join(OUTPUT_DIR, split, "annotations"),
                os.path.join(OUTPUT_DIR, split, "images"),
                prefix="ds_",
            )
            totals[split] += n
            print(f"    {split}: {n} copied")

    # 3. Blender synthetic — train only
    if not args.no_synthetic and os.path.isdir(BLENDER_ANN_DIR):
        print("\n=== Blender Synthetic (train only) ===")
        synth_items = load_cvn_annotations(BLENDER_ANN_DIR, args.min_visible)
        n = copy_source(
            synth_items, BLENDER_ANN_DIR, BLENDER_IMG_DIR,
            os.path.join(OUTPUT_DIR, "train", "annotations"),
            os.path.join(OUTPUT_DIR, "train", "images"),
            prefix="bl_",
        )
        totals["train"] += n
        print(f"    train: {n} copied")

    # Summary
    total = sum(totals.values())
    print(f"\n{'='*50}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Total: {total} images")
    print(f"  Train: {totals['train']}")
    print(f"  Valid: {totals['valid']}")
    print(f"  Test:  {totals['test']}")

    # Zip
    if not args.no_zip:
        zip_directory(OUTPUT_DIR, ZIP_PATH)
        print(f"\nUpload {ZIP_PATH} to Google Drive, then use the Colab notebook to train.")


if __name__ == "__main__":
    main()

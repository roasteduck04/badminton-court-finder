"""Standalone evaluation CLI for CourtVisionNet."""

import argparse
import json
import os

import cv2
import numpy as np

from src.court_geometry import NUM_KEYPOINTS, KEYPOINT_NAMES
from src.evaluation.metrics import (
    court_iou,
    mean_reprojection_error,
    pck_at_k,
    segmentation_iou,
)
from src.inference.predict import CourtPredictor
from src.court_geometry import generate_line_mask


def evaluate(checkpoint_path, annotations_dir, images_dir, image_size=640, device="cpu"):
    """Run full evaluation on a test set.

    Returns a dict with summary metrics.
    """
    predictor = CourtPredictor(checkpoint_path, device=device, image_size=image_size)

    ann_paths = sorted(
        [os.path.join(annotations_dir, f) for f in os.listdir(annotations_dir) if f.endswith(".json")]
    )

    results = {
        "pck_5": [], "pck_10": [], "pck_20": [],
        "mre": [], "court_iou": [], "seg_iou": [],
        "per_keypoint_pck_10": [[] for _ in range(NUM_KEYPOINTS)],
    }

    for ann_path in ann_paths:
        with open(ann_path) as f:
            ann = json.load(f)

        img_path = ann["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(images_dir, os.path.basename(img_path))
        image = cv2.imread(img_path)
        if image is None:
            continue

        gt_kps = np.array(ann["keypoints"], dtype=np.float32)
        gt_vis = np.array(ann["visibility"], dtype=np.float32)

        detection = predictor.predict(image)
        pred_kps = detection.keypoints.astype(np.float32)
        pred_vis = detection.visibility.astype(np.float32)

        h, w = image.shape[:2]

        for label, k in [("pck_5", 5), ("pck_10", 10), ("pck_20", 20)]:
            per_kp, mean_acc = pck_at_k(pred_kps, gt_kps, gt_vis, k=k, image_size=image_size)
            results[label].append(mean_acc)
            if label == "pck_10":
                for i in range(NUM_KEYPOINTS):
                    if gt_vis[i] > 0.5:
                        results["per_keypoint_pck_10"][i].append(float(per_kp[i]))

        mre = mean_reprojection_error(pred_kps, gt_kps, gt_vis, w, h)
        if mre is not None:
            results["mre"].append(mre)

        iou = court_iou(pred_kps, gt_kps, pred_vis, gt_vis)
        results["court_iou"].append(iou)

        gt_mask = generate_line_mask(gt_kps, gt_vis.astype(int).tolist(), w, h).astype(np.float32) / 255.0
        pred_mask = cv2.resize(detection.seg_mask.astype(np.float32), (w, h))
        results["seg_iou"].append(segmentation_iou(pred_mask, gt_mask))

    summary = {
        "n_images": len(ann_paths),
        "pck_5_mean": float(np.mean(results["pck_5"])) if results["pck_5"] else 0.0,
        "pck_10_mean": float(np.mean(results["pck_10"])) if results["pck_10"] else 0.0,
        "pck_20_mean": float(np.mean(results["pck_20"])) if results["pck_20"] else 0.0,
        "mre_mean": float(np.mean(results["mre"])) if results["mre"] else 0.0,
        "mre_std": float(np.std(results["mre"])) if results["mre"] else 0.0,
        "court_iou_mean": float(np.mean(results["court_iou"])) if results["court_iou"] else 0.0,
        "seg_iou_mean": float(np.mean(results["seg_iou"])) if results["seg_iou"] else 0.0,
    }

    per_kp_pck = {}
    for i in range(NUM_KEYPOINTS):
        vals = results["per_keypoint_pck_10"][i]
        per_kp_pck[KEYPOINT_NAMES[i]] = float(np.mean(vals)) if vals else 0.0
    summary["per_keypoint_pck_10"] = per_kp_pck

    return summary


def print_summary(summary):
    """Pretty-print evaluation summary."""
    print(f"\n{'='*40}")
    print(f" CourtVisionNet Evaluation")
    print(f"{'='*40}")
    print(f" Images:     {summary['n_images']}")
    print(f" PCK@5:      {summary['pck_5_mean']:.1%}")
    print(f" PCK@10:     {summary['pck_10_mean']:.1%}")
    print(f" PCK@20:     {summary['pck_20_mean']:.1%}")
    print(f" MRE:        {summary['mre_mean']:.2f} px (±{summary['mre_std']:.2f})")
    print(f" Court IoU:  {summary['court_iou_mean']:.3f}")
    print(f" Seg IoU:    {summary['seg_iou_mean']:.3f}")
    print(f"\n Per-keypoint PCK@10:")
    for name, val in summary["per_keypoint_pck_10"].items():
        print(f"   {name:30s} {val:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CourtVisionNet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = evaluate(
        args.checkpoint, args.annotations, args.images,
        image_size=args.image_size, device=args.device,
    )
    print_summary(summary)

"""Tests for src.evaluation.evaluate and validate() metrics integration."""

import json
import os

import cv2
import numpy as np
import pytest
import torch

from src.court_geometry import NUM_KEYPOINTS
from src.training.train import validate


def _create_test_dataset(tmp_path, n_images=2):
    """Create a minimal dataset for evaluation testing."""
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    for i in range(n_images):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img_path = str(img_dir / f"frame_{i:04d}.jpg")
        cv2.imwrite(img_path, img)

        kps = np.random.rand(NUM_KEYPOINTS, 2).tolist()
        vis = [1] * 4 + [0] * (NUM_KEYPOINTS - 4)

        ann = {
            "image_path": img_path,
            "keypoints": kps,
            "visibility": vis,
            "court_class": 1,
        }
        with open(ann_dir / f"frame_{i:04d}.json", "w") as f:
            json.dump(ann, f)

    return str(ann_dir), str(img_dir)


def test_validate_returns_metrics(tmp_path):
    """validate() should return loss, components, AND metrics."""
    from torch.utils.data import DataLoader
    from src.models.courtvisionnet import CourtVisionNet
    from src.models.losses import CourtVisionLoss
    from src.preprocessing.augmentation import get_val_transforms
    from src.training.dataset import CourtDataset

    ann_dir, img_dir = _create_test_dataset(tmp_path)
    ds = CourtDataset(ann_dir, img_dir, transform=get_val_transforms(64), image_size=64, heatmap_size=16)
    loader = DataLoader(ds, batch_size=2)

    model = CourtVisionNet(in_channels=7, image_size=64, heatmap_size=16, pretrained=False)
    loss_fn = CourtVisionLoss()

    avg_loss, avg_components, avg_metrics = validate(model, loader, loss_fn, device="cpu")

    assert isinstance(avg_loss, float)
    assert "seg_loss" in avg_components
    assert "pck_at_10" in avg_metrics
    assert "mre" in avg_metrics
    assert 0.0 <= avg_metrics["pck_at_10"] <= 1.0


def test_evaluate_returns_summary_dict(tmp_path):
    """evaluate() runs a full checkpoint-based evaluation and returns a summary dict."""
    from src.evaluation.evaluate import evaluate
    from src.models.courtvisionnet import CourtVisionNet

    ann_dir, img_dir = _create_test_dataset(tmp_path)

    checkpoint_path = str(tmp_path / "model.pt")
    model = CourtVisionNet(in_channels=7, image_size=64, heatmap_size=16, pretrained=False)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    summary = evaluate(checkpoint_path, ann_dir, img_dir, image_size=64, device="cpu")

    assert isinstance(summary, dict)
    assert summary["n_images"] == 2
    assert "pck_10_mean" in summary
    assert "mre_mean" in summary
    assert "court_iou_mean" in summary
    assert "seg_iou_mean" in summary
    assert "per_keypoint_pck_10" in summary
    assert len(summary["per_keypoint_pck_10"]) == NUM_KEYPOINTS


def test_print_summary_smoke(tmp_path, capsys):
    """print_summary() should not raise and should print key metric labels."""
    from src.evaluation.evaluate import print_summary

    summary = {
        "n_images": 2,
        "pck_5_mean": 0.1,
        "pck_10_mean": 0.2,
        "pck_20_mean": 0.3,
        "mre_mean": 5.0,
        "mre_std": 1.0,
        "court_iou_mean": 0.4,
        "seg_iou_mean": 0.5,
        "per_keypoint_pck_10": {"Baseline-L / Dbl-top": 0.5},
    }
    print_summary(summary)
    captured = capsys.readouterr()
    assert "CourtVisionNet Evaluation" in captured.out
    assert "PCK@10" in captured.out

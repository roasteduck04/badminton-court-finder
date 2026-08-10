import torch
import pytest
from src.models.segmentation_head import SegmentationHead
from src.models.keypoint_head import KeypointHead


def make_fake_features(batch_size=2):
    return {
        "p2": torch.randn(batch_size, 256, 160, 160),
        "p3": torch.randn(batch_size, 256, 80, 80),
        "p4": torch.randn(batch_size, 256, 40, 40),
        "p5": torch.randn(batch_size, 256, 20, 20),
    }


def test_segmentation_head_output_shape():
    head = SegmentationHead(in_channels=256, image_size=640)
    features = make_fake_features()
    out = head(features)
    assert out.shape == (2, 1, 640, 640)


def test_keypoint_head_output_shapes():
    head = KeypointHead(in_channels=256, num_keypoints=31, heatmap_size=160)
    features = make_fake_features()
    out = head(features)
    assert out["heatmaps"].shape == (2, 31, 160, 160)
    assert out["offsets"].shape == (2, 31, 2)
    assert out["visibility"].shape == (2, 31)


def test_segmentation_head_gradient():
    head = SegmentationHead(in_channels=256, image_size=640)
    features = make_fake_features()
    features["p2"].requires_grad_(True)
    out = head(features)
    out.sum().backward()
    assert features["p2"].grad is not None


def test_keypoint_head_gradient():
    head = KeypointHead(in_channels=256, num_keypoints=31, heatmap_size=160)
    features = make_fake_features()
    features["p2"].requires_grad_(True)
    out = head(features)
    loss = out["heatmaps"].sum() + out["offsets"].sum() + out["visibility"].sum()
    loss.backward()
    assert features["p2"].grad is not None

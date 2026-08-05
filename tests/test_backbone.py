import torch
import pytest
from src.models.backbone import CourtBackbone


def test_backbone_output_shapes():
    model = CourtBackbone(in_channels=7, pretrained=False)
    x = torch.randn(2, 7, 640, 640)
    features = model(x)

    assert "p2" in features
    assert "p3" in features
    assert "p4" in features
    assert "p5" in features

    assert features["p2"].shape == (2, 256, 160, 160)
    assert features["p3"].shape == (2, 256, 80, 80)
    assert features["p4"].shape == (2, 256, 40, 40)
    assert features["p5"].shape == (2, 256, 20, 20)


def test_backbone_3_channel_input():
    model = CourtBackbone(in_channels=3, pretrained=False)
    x = torch.randn(2, 3, 640, 640)
    features = model(x)
    assert features["p2"].shape == (2, 256, 160, 160)


def test_backbone_gradient_flow():
    model = CourtBackbone(in_channels=7, pretrained=False)
    x = torch.randn(1, 7, 256, 256, requires_grad=True)
    features = model(x)
    loss = sum(f.sum() for f in features.values())
    loss.backward()
    assert x.grad is not None

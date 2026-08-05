import torch
import pytest
from src.models.courtvisionnet import CourtVisionNet
from src.models.losses import DiceLoss, CourtVisionLoss


def test_model_forward():
    model = CourtVisionNet(in_channels=7, pretrained=False)
    x = torch.randn(2, 7, 640, 640)
    out = model(x)

    assert "seg_logits" in out
    assert "heatmaps" in out
    assert "offsets" in out
    assert "visibility" in out

    assert out["seg_logits"].shape == (2, 1, 640, 640)
    assert out["heatmaps"].shape == (2, 14, 160, 160)
    assert out["offsets"].shape == (2, 14, 2)
    assert out["visibility"].shape == (2, 14)


def test_model_parameter_count():
    model = CourtVisionNet(in_channels=7, pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    # Should be roughly 28-32M params
    assert 20_000_000 < total < 40_000_000


def test_model_gradient_flow():
    model = CourtVisionNet(in_channels=7, pretrained=False)
    x = torch.randn(1, 7, 640, 640)
    out = model(x)
    loss = (
        out["seg_logits"].sum()
        + out["heatmaps"].sum()
        + out["offsets"].sum()
        + out["visibility"].sum()
    )
    loss.backward()
    grad_found = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert grad_found


def test_freeze_backbone():
    model = CourtVisionNet(in_channels=7, pretrained=False)
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.seg_head.parameters())
    assert all(p.requires_grad for p in model.kpt_head.parameters())


def test_dice_loss_perfect_match():
    dice = DiceLoss()
    target = torch.ones(4, 1, 8, 8)
    pred_logits = torch.full((4, 1, 8, 8), 10.0)  # sigmoid(10) ~= 1
    loss = dice(pred_logits, target)
    assert loss.item() < 0.01


def test_loss_function():
    loss_fn = CourtVisionLoss()

    # requires_grad=True mirrors real usage, where these tensors are
    # produced by the model's forward pass (and thus grad-tracked).
    pred = {
        "seg_logits": torch.randn(2, 1, 640, 640, requires_grad=True),
        "heatmaps": torch.randn(2, 14, 160, 160, requires_grad=True),
        "offsets": torch.randn(2, 14, 2, requires_grad=True),
        "visibility": torch.randn(2, 14, requires_grad=True),
    }
    targets = {
        "mask": torch.randint(0, 2, (2, 1, 640, 640)).float(),
        "heatmaps": torch.randn(2, 14, 160, 160),
        "keypoints": torch.rand(2, 14, 2),
        "visibility": torch.randint(0, 2, (2, 14)).float(),
    }

    total_loss, components = loss_fn(pred, targets)
    assert total_loss.requires_grad
    assert "seg_loss" in components
    assert "heatmap_loss" in components
    assert "offset_loss" in components
    assert "visibility_loss" in components


def test_loss_zero_visible():
    """Loss should handle the case where no keypoints are visible."""
    loss_fn = CourtVisionLoss()
    pred = {
        "seg_logits": torch.randn(1, 1, 640, 640),
        "heatmaps": torch.randn(1, 14, 160, 160),
        "offsets": torch.randn(1, 14, 2),
        "visibility": torch.randn(1, 14),
    }
    targets = {
        "mask": torch.zeros(1, 1, 640, 640),
        "heatmaps": torch.zeros(1, 14, 160, 160),
        "keypoints": torch.full((1, 14, 2), -1.0),
        "visibility": torch.zeros(1, 14),
    }
    total_loss, _ = loss_fn(pred, targets)
    assert not torch.isnan(total_loss)


def test_loss_default_weights():
    """heatmap_weight should default to 5.0, others to 1.0."""
    loss_fn = CourtVisionLoss()
    assert loss_fn.seg_weight == 1.0
    assert loss_fn.heatmap_weight == 5.0
    assert loss_fn.offset_weight == 1.0
    assert loss_fn.vis_weight == 1.0


def test_loss_weights_configurable():
    pred = {
        "seg_logits": torch.zeros(1, 1, 4, 4),
        "heatmaps": torch.zeros(1, 2, 4, 4),
        "offsets": torch.zeros(1, 2, 2),
        "visibility": torch.zeros(1, 2),
    }
    targets = {
        "mask": torch.zeros(1, 1, 4, 4),
        "heatmaps": torch.ones(1, 2, 4, 4),
        "keypoints": torch.ones(1, 2, 2),
        "visibility": torch.ones(1, 2),
    }
    loss_fn_default = CourtVisionLoss()
    loss_fn_zeroed = CourtVisionLoss(
        seg_weight=0.0, heatmap_weight=0.0, offset_weight=0.0, vis_weight=0.0
    )

    total_default, _ = loss_fn_default(pred, targets)
    total_zeroed, _ = loss_fn_zeroed(pred, targets)

    assert total_default.item() > 0
    assert total_zeroed.item() == pytest.approx(0.0, abs=1e-6)

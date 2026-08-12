import torch
import pytest

from src.models.losses import CourtVisionLoss
from src.court_geometry import NUM_KEYPOINTS, CORNER_INDICES


def _dummy_pred_targets(batch_size=2, hm_size=16, img_size=64):
    """Create minimal pred/target dicts for loss computation."""
    K = NUM_KEYPOINTS
    # requires_grad=True mirrors real model outputs, which are always
    # differentiable w.r.t. network parameters.
    pred = {
        "seg_logits": torch.randn(batch_size, 1, img_size, img_size, requires_grad=True),
        "heatmaps": torch.randn(batch_size, K, hm_size, hm_size, requires_grad=True),
        "offsets": torch.randn(batch_size, K, 2, requires_grad=True),
        "visibility": torch.randn(batch_size, K, requires_grad=True),
    }
    targets = {
        "mask": torch.zeros(batch_size, 1, img_size, img_size),
        "heatmaps": torch.zeros(batch_size, K, hm_size, hm_size),
        "keypoints": torch.rand(batch_size, K, 2),
        "visibility": torch.ones(batch_size, K),
    }
    return pred, targets


class TestCollinearityLoss:
    def test_perfectly_collinear_gives_zero(self):
        """Points in a straight line -> collinearity loss = 0."""
        loss_fn = CourtVisionLoss(collinear_weight=1.0)
        pred, targets = _dummy_pred_targets()
        # Make all predicted offsets map to template-like positions (collinear rows)
        # by using keypoints that lie on straight lines
        kps = torch.zeros(2, NUM_KEYPOINTS, 2)
        for row_start in range(0, 30, 5):
            for j in range(5):
                kps[:, row_start + j, 0] = row_start / 30.0
                kps[:, row_start + j, 1] = j / 5.0
        targets["keypoints"] = kps
        pred["offsets"] = kps  # perfect prediction
        loss, components = loss_fn(pred, targets)
        assert components["collinear_loss"] < 1e-6

    def test_noncollinear_gives_positive_loss(self):
        loss_fn = CourtVisionLoss(collinear_weight=1.0)
        pred, targets = _dummy_pred_targets()
        # Deliberately put K2 off-line from K0-K1-K3-K4
        kps = targets["keypoints"].clone()
        kps[:, 2, :] += 0.5  # push center point far off line
        pred["offsets"] = kps
        loss, components = loss_fn(pred, targets)
        assert components["collinear_loss"] > 0.0


class TestDistanceRatioLoss:
    def test_correct_ratios_give_zero(self):
        loss_fn = CourtVisionLoss(ratio_weight=1.0)
        pred, targets = _dummy_pred_targets()
        from src.court_geometry import COURT_KEYPOINTS_TEMPLATE
        import numpy as np
        tpl_norm = torch.from_numpy(
            COURT_KEYPOINTS_TEMPLATE / np.array([13.4, 6.1])
        ).float()
        tpl_batch = tpl_norm.unsqueeze(0).expand(2, -1, -1)
        targets["keypoints"] = tpl_batch
        pred["offsets"] = tpl_batch
        loss, components = loss_fn(pred, targets)
        assert components["ratio_loss"] < 1e-4

    def test_wrong_ratios_give_positive_loss(self):
        loss_fn = CourtVisionLoss(ratio_weight=1.0)
        pred, targets = _dummy_pred_targets()
        # Random keypoints will have wrong distance ratios
        pred["offsets"] = torch.rand(2, NUM_KEYPOINTS, 2)
        loss, components = loss_fn(pred, targets)
        assert components["ratio_loss"] > 0.0


class TestConvexityLoss:
    def test_convex_quad_gives_zero(self):
        loss_fn = CourtVisionLoss(convex_weight=1.0)
        pred, targets = _dummy_pred_targets()
        kps = torch.zeros(2, NUM_KEYPOINTS, 2)
        # K0=TL, K25=TR, K29=BR, K4=BL -- convex
        kps[:, 0] = torch.tensor([0.1, 0.1])
        kps[:, 25] = torch.tensor([0.9, 0.1])
        kps[:, 29] = torch.tensor([0.9, 0.9])
        kps[:, 4] = torch.tensor([0.1, 0.9])
        pred["offsets"] = kps
        targets["visibility"] = torch.ones(2, NUM_KEYPOINTS)
        loss, components = loss_fn(pred, targets)
        assert components["convex_loss"] < 1e-6

    def test_bowtie_gives_positive_loss(self):
        loss_fn = CourtVisionLoss(convex_weight=1.0)
        pred, targets = _dummy_pred_targets()
        kps = torch.zeros(2, NUM_KEYPOINTS, 2)
        # Swap K25 and K4 to create a bowtie (non-convex)
        kps[:, 0] = torch.tensor([0.1, 0.1])
        kps[:, 25] = torch.tensor([0.1, 0.9])   # was TR, now BL
        kps[:, 29] = torch.tensor([0.9, 0.9])
        kps[:, 4] = torch.tensor([0.9, 0.1])    # was BL, now TR
        pred["offsets"] = kps
        targets["visibility"] = torch.ones(2, NUM_KEYPOINTS)
        loss, components = loss_fn(pred, targets)
        assert components["convex_loss"] > 0.0

    def test_missing_corner_skips_loss(self):
        loss_fn = CourtVisionLoss(convex_weight=1.0)
        pred, targets = _dummy_pred_targets()
        targets["visibility"] = torch.zeros(2, NUM_KEYPOINTS)  # no corners visible
        loss, components = loss_fn(pred, targets)
        assert components["convex_loss"] == 0.0


class TestGeometricLossIntegration:
    def test_total_loss_includes_geometric_terms(self):
        loss_fn = CourtVisionLoss(
            collinear_weight=0.1, ratio_weight=0.1, convex_weight=0.1
        )
        pred, targets = _dummy_pred_targets()
        loss, components = loss_fn(pred, targets)
        assert "collinear_loss" in components
        assert "ratio_loss" in components
        assert "convex_loss" in components
        assert loss.requires_grad

    def test_geometric_weights_zero_disables(self):
        loss_fn = CourtVisionLoss(
            collinear_weight=0.0, ratio_weight=0.0, convex_weight=0.0
        )
        pred, targets = _dummy_pred_targets()
        loss_with, _ = loss_fn(pred, targets)

        loss_fn_base = CourtVisionLoss()
        loss_base, _ = loss_fn_base(pred, targets)

        # With all geometric weights at 0, should equal the base loss
        assert abs(loss_with.item() - loss_base.item()) < 1e-5

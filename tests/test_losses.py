"""Tests for src.models.losses (replaces the old geometric-loss tests)."""

import cv2
import numpy as np
import pytest
import torch

from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, SYMMETRY_PERMS
from src.inference.decode import gaussian_heatmap
from src.models.losses import CourtVisionLoss, DiceLoss, js_divergence

IMG, HM, SIGMA = 128, 32, 1.5
ROT = SYMMETRY_PERMS["rot180"]


def _court_keypoints(seed):
    """Normalized canvas keypoints of a plausible perspective view."""
    rng = np.random.default_rng(seed)
    dst = np.array([[0.2, 0.85], [0.8, 0.85], [0.35, 0.2], [0.65, 0.2]]) + rng.uniform(-0.03, 0.03, (4, 2))
    H, _ = cv2.findHomography(COURT_KEYPOINTS_TEMPLATE[[0, 4, 25, 29]], dst, 0)
    ph = np.hstack([COURT_KEYPOINTS_TEMPLATE, np.ones((30, 1))]) @ H.T
    return (ph[:, :2] / ph[:, 2:]).astype(np.float32)


def _targets(batch=2, vis=None):
    kps = np.stack([_court_keypoints(s) for s in range(batch)])
    if vis is None:
        vis = np.ones((batch, 30), np.float32)
    hms = np.stack([[gaussian_heatmap(k * HM - 0.5, (HM, HM), SIGMA) * v for k, v in zip(ks, vs)]
                    for ks, vs in zip(kps, vis)])
    mask = np.zeros((batch, 1, IMG, IMG), np.float32)
    mask[:, :, 40:60, 20:100] = 1.0
    kps_t = torch.from_numpy(kps)
    kps_t[torch.from_numpy(vis) == 0] = -1.0
    return {
        "mask": torch.from_numpy(mask),
        "heatmaps": torch.from_numpy(hms.astype(np.float32)),
        "keypoints": kps_t,
        "visibility": torch.from_numpy(vis.astype(np.float32)),
    }


def _perfect_pred(t, heatmap_loss, beta=1.0):
    hm = t["heatmaps"]
    if heatmap_loss == "js":
        q = hm / hm.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        logits = torch.log(q + 1e-30) / beta
    else:
        logits = hm.clone()
    return {
        "seg_logits": (t["mask"] * 2 - 1) * 30.0,
        "heatmaps": logits,
        "offsets": t["keypoints"].clone(),
        "visibility": (t["visibility"] * 2 - 1) * 30.0,
        "beta": torch.tensor(beta),
    }


def _random_pred(batch=2, requires_grad=True):
    torch.manual_seed(0)
    pred = {
        "seg_logits": torch.randn(batch, 1, IMG, IMG),
        "heatmaps": torch.randn(batch, 30, HM, HM),
        "offsets": torch.rand(batch, 30, 2),
        "visibility": torch.randn(batch, 30),
        "beta": torch.tensor(2.0),
    }
    for v in pred.values():
        v.requires_grad_(requires_grad)
    return pred


def _relabel(t, perm):
    return {k: (v[:, perm] if k in ("heatmaps", "keypoints", "visibility") else v) for k, v in t.items()}


def test_default_weights():
    lf = CourtVisionLoss()
    assert (lf.seg_weight, lf.heatmap_weight, lf.offset_weight, lf.vis_weight, lf.reproj_weight) == \
        (1.0, 1.0, 5.0, 0.5, 0.01)
    assert lf.heatmap_loss == "js" and lf.symmetric is False


def test_contract_kwargs_and_validation():
    CourtVisionLoss(seg_weight=1.0, heatmap_weight=1.0, offset_weight=1.0, vis_weight=1.0,
                    reproj_weight=0.1, heatmap_loss="mse", symmetric=True)
    with pytest.raises(ValueError):
        CourtVisionLoss(heatmap_loss="l2")


@pytest.mark.parametrize("heatmap_loss", ["js", "mse"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_finite_and_differentiable(heatmap_loss, symmetric):
    pred = _random_pred()
    total, comp = CourtVisionLoss(heatmap_loss=heatmap_loss, symmetric=symmetric)(pred, _targets())
    assert torch.isfinite(total)
    total.backward()
    for k in ("heatmaps", "offsets", "visibility", "seg_logits"):
        assert pred[k].grad is not None and torch.isfinite(pred[k].grad).all()
    if heatmap_loss == "js":
        assert pred["beta"].grad is not None
    assert {"seg_loss", "heatmap_loss", "offset_loss", "visibility_loss", "reproj_loss"} <= set(comp)
    assert ("rot180_fraction" in comp) == symmetric


@pytest.mark.parametrize("heatmap_loss", ["js", "mse"])
def test_near_zero_at_perfect_prediction(heatmap_loss):
    t = _targets()
    total, comp = CourtVisionLoss(heatmap_loss=heatmap_loss)(_perfect_pred(t, heatmap_loss), t)
    assert comp["heatmap_loss"] < 1e-4
    assert comp["offset_loss"] < 1e-4
    assert comp["reproj_loss"] < 1e-3
    assert comp["visibility_loss"] < 1e-6
    assert comp["seg_loss"] < 0.05  # Dice smoothing keeps a small floor


def test_mse_is_one_for_zero_prediction():
    t = _targets()
    pred = _perfect_pred(t, "mse")
    pred["heatmaps"] = torch.zeros_like(pred["heatmaps"])
    _, comp = CourtVisionLoss(heatmap_loss="mse")(pred, t)
    assert comp["heatmap_loss"] == pytest.approx(1.0, abs=1e-5)


def test_js_bounded():
    p = torch.tensor([[1.0, 0.0, 0.0]])
    q = torch.tensor([[0.0, 0.0, 1.0]])
    assert js_divergence(p, q).item() == pytest.approx(np.log(2), abs=1e-5)
    assert js_divergence(p, p).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("heatmap_loss", ["js", "mse"])
def test_symmetric_loss_invariant_to_rot180_relabel(heatmap_loss):
    t = _targets()
    pred = _random_pred(requires_grad=False)
    sym = CourtVisionLoss(heatmap_loss=heatmap_loss, symmetric=True)
    a, _ = sym(pred, t)
    b, _ = sym(pred, _relabel(t, ROT))
    assert a.item() == pytest.approx(b.item(), rel=1e-5)

    # a perfect prediction of the rot180 labeling costs ~nothing with symmetric
    perfect = _perfect_pred(t, heatmap_loss)
    total_sym, comp = sym(perfect, _relabel(t, ROT))
    assert comp["rot180_fraction"] == 1.0
    assert comp["offset_loss"] < 1e-4 and comp["visibility_loss"] < 1e-6
    total_plain, _ = CourtVisionLoss(heatmap_loss=heatmap_loss)(perfect, _relabel(t, ROT))
    assert total_plain.item() > total_sym.item() + 0.5


def test_half_precision_inputs():
    t = _targets()
    pred = {k: v.detach().half() for k, v in _random_pred().items()}
    total, _ = CourtVisionLoss(symmetric=True)(pred, t)
    assert total.dtype == torch.float32 and torch.isfinite(total)


def test_zero_visible_is_finite():
    vis = np.zeros((2, 30), np.float32)
    t = _targets(vis=vis)
    pred = _random_pred()
    for hl in ("js", "mse"):
        total, comp = CourtVisionLoss(heatmap_loss=hl, symmetric=True)(pred, t)
        assert torch.isfinite(total)
        assert comp["heatmap_loss"] == 0.0 and comp["reproj_loss"] == 0.0


def test_reprojection_term():
    t = _targets()
    lf = CourtVisionLoss(seg_weight=0, heatmap_weight=0, offset_weight=0, vis_weight=0, reproj_weight=1.0)
    pred = _perfect_pred(t, "js")
    _, comp = lf(pred, t)
    assert comp["reproj_loss"] < 1e-3

    off = t["keypoints"].clone()
    off[0, 12] += torch.tensor([0.05, 0.0])  # 6.4 px off the court plane
    pred["offsets"] = off.requires_grad_(True)
    total, comp = lf(pred, t)
    assert comp["reproj_loss"] > 0.05
    total.backward()
    assert pred["offsets"].grad[0, 12].abs().sum() > 0
    g = pred["offsets"].grad
    assert g[1].abs().max() < 1e-2 * g[0, 12].abs().max()  # sample 1 is consistent


def test_reprojection_skips_samples_without_support():
    vis = np.zeros((2, 30), np.float32)
    vis[:, [0, 1, 2, 3, 4]] = 1.0  # one collinear row: no homography
    t = _targets(vis=vis)
    pred = _random_pred()
    _, comp = CourtVisionLoss(reproj_weight=1.0)(pred, t)
    assert comp["reproj_loss"] == 0.0


def test_reprojection_differentiable_through_fit():
    t = _targets()
    lf = CourtVisionLoss(seg_weight=0, heatmap_weight=0, offset_weight=0, vis_weight=0,
                         reproj_weight=1.0, reproj_detach_fit=False)
    pred = _random_pred()
    total, _ = lf(pred, t)
    total.backward()
    assert torch.isfinite(pred["offsets"].grad).all()


def test_valid_mask_ignores_padding():
    t = _targets()
    pred = _perfect_pred(t, "js")
    seg = pred["seg_logits"].clone()
    seg[:, :, :20] = 30.0  # wrong, but only inside the padding band
    pred["seg_logits"] = seg
    valid = torch.ones_like(t["mask"])
    valid[:, :, :20] = 0
    lf = CourtVisionLoss()
    _, with_valid = lf(pred, dict(t, valid=valid))
    _, without = lf(pred, t)
    assert with_valid["seg_loss"] < 0.05 < without["seg_loss"]


def test_dice_loss_per_image():
    dice = DiceLoss()
    target = torch.ones(4, 1, 8, 8)
    assert dice(torch.full((4, 1, 8, 8), 10.0), target).item() < 0.01
    # one bad image out of two -> ~0.5, not dominated by pixel counts
    pred = torch.full((2, 1, 8, 8), 10.0)
    pred[1] = -10.0
    assert dice(pred, torch.ones(2, 1, 8, 8)).item() == pytest.approx(0.5, abs=0.02)

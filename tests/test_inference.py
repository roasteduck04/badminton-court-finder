"""Tests for src.inference.predict and src.inference.visualize."""

import cv2
import numpy as np
import pytest
import torch

from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, SYMMETRY_PERMS, get_court_lines
from src.inference.predict import (
    CourtDetection,
    CourtPredictor,
    fit_court_homography,
    load_model,
    projected_court_lines,
    refine_keypoints,
)
from src.inference.visualize import (
    create_debug_visualization, draw_court_overlay, draw_keypoints, draw_refinement,
)
from src.models.courtvisionnet import CourtVisionNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _camera_homography(w=640, h=480, seed=0):
    rng = np.random.default_rng(seed)
    src = COURT_KEYPOINTS_TEMPLATE[[0, 4, 25, 29]]
    # end-on view: near baseline (row 0) at the bottom, far baseline at the top
    dst = np.array([[0.15 * w, 0.9 * h], [0.85 * w, 0.9 * h],
                    [0.35 * w, 0.25 * h], [0.65 * w, 0.25 * h]])
    dst += rng.uniform(-5, 5, dst.shape)
    H, _ = cv2.findHomography(src, dst, 0)
    return H


def _project_norm(H, w, h):
    ph = np.hstack([COURT_KEYPOINTS_TEMPLATE, np.ones((30, 1))]) @ H.T
    return ph[:, :2] / ph[:, 2:] / np.array([w, h])


def _sample_detection():
    kps = np.linspace(0.05, 0.95, 30 * 2).reshape(30, 2)
    return CourtDetection(
        keypoints=kps,
        visibility=np.ones(30),
        confidence=0.9,
        homography=np.eye(3),
        seg_mask=np.zeros((640, 640), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# CourtDetection dataclass
# ---------------------------------------------------------------------------

def test_court_detection_defaults():
    det = CourtDetection(keypoints=np.zeros((30, 2)), visibility=np.zeros(30), confidence=0.5)
    assert det.homography is None
    assert det.seg_mask is None
    assert det.inlier_mask is None
    np.testing.assert_array_equal(det.raw_keypoints, det.keypoints)
    assert det.raw_keypoints is not det.keypoints


# ---------------------------------------------------------------------------
# Homography fitting / refinement
# ---------------------------------------------------------------------------

def test_fit_court_homography_recovers_truth():
    w, h = 640, 480
    H_true = _camera_homography(w, h)
    kps = _project_norm(H_true, w, h)
    cand = np.ones(30, dtype=bool)
    H, inliers, status = fit_court_homography(kps, cand, (h, w))
    assert status == "ok"
    assert inliers.all()
    np.testing.assert_allclose(_project_norm(H, w, h), kps, atol=1e-4)


def test_ransac_rejects_injected_outlier_and_refines_it():
    w, h = 640, 480
    H_true = _camera_homography(w, h, seed=3)
    truth = _project_norm(H_true, w, h)
    rng = np.random.default_rng(0)
    raw = truth + rng.normal(0, 0.5 / w, truth.shape)
    raw[12] += np.array([0.15, -0.1])  # gross outlier
    cand = np.ones(30, dtype=bool)
    cand[[27, 28]] = False  # "not visible"
    raw[[27, 28]] = 0.5     # junk raw positions for unseen points

    H, inliers, status = fit_court_homography(raw, cand, (h, w), threshold_frac=0.01)
    assert status == "ok"
    assert not inliers[12]
    assert inliers[cand & (np.arange(30) != 12)].sum() >= 25

    refined, proj = refine_keypoints(raw, H, cand, inliers, (h, w), fill="in_frame")
    assert np.linalg.norm((refined[12] - truth[12]) * [w, h]) < 2.0
    assert np.linalg.norm((refined[27] - truth[27]) * [w, h]) < 2.0
    np.testing.assert_array_equal(refined[0], raw[0])  # inliers untouched

    only_outliers, _ = refine_keypoints(raw, H, cand, inliers, (h, w), fill="outliers")
    np.testing.assert_array_equal(only_outliers[27], raw[27])
    none, _ = refine_keypoints(raw, H, cand, inliers, (h, w), fill="none")
    np.testing.assert_array_equal(none, raw)


def test_fit_rejects_mirrored_court():
    w, h = 640, 480
    kps = _project_norm(_camera_homography(w, h), w, h)
    mirrored = kps[SYMMETRY_PERMS["col_flip"]]
    H, inliers, status = fit_court_homography(mirrored, np.ones(30, bool), (h, w))
    # USAC's own orientation check may already reject it; otherwise ours does
    assert H is None and inliers is None and status in ("mirrored", "fit_failed")


def test_fit_needs_enough_points():
    w, h = 640, 480
    kps = _project_norm(_camera_homography(w, h), w, h)
    cand = np.zeros(30, bool)
    cand[[0, 4, 25]] = True
    H, _, status = fit_court_homography(kps, cand, (h, w))
    assert H is None and status.startswith("too_few_points")
    cand[:] = False
    cand[[0, 1, 2, 3, 4, 5, 9]] = False
    cand[[0, 1, 2, 3, 4]] = True  # one collinear row only
    cand[[25]] = True
    H, _, status = fit_court_homography(kps, cand, (h, w))
    assert H is None


def test_projected_court_lines():
    lines = projected_court_lines(_camera_homography())
    assert len(lines) == len(get_court_lines())


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def test_draw_court_overlay():
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    original = img.copy()
    result = draw_court_overlay(img, _sample_detection())
    assert result.shape == img.shape and result.dtype == np.uint8
    np.testing.assert_array_equal(img, original)


def test_draw_court_overlay_without_homography():
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    det = CourtDetection(keypoints=np.full((30, 2), 0.5), visibility=np.zeros(30), confidence=0.0)
    result = draw_court_overlay(img, det)
    assert result.shape == img.shape


def test_draw_keypoints_skips_low_visibility_points():
    img = np.zeros((100, 150, 3), dtype=np.uint8)
    det = _sample_detection()
    det.visibility = np.zeros(30)
    np.testing.assert_array_equal(draw_keypoints(img, det), img)


def test_draw_refinement_and_debug_view():
    w, h = 320, 240
    H = _camera_homography(w, h)
    truth = _project_norm(H, w, h)
    raw = truth.copy()
    raw[12] += 0.1
    inliers = np.ones(30, bool)
    inliers[12] = False
    det = CourtDetection(keypoints=truth, visibility=np.ones(30), confidence=1.0, homography=H,
                         raw_keypoints=raw, inlier_mask=inliers, homography_status="ok",
                         seg_mask_image=np.zeros((h, w), np.uint8))
    img = np.zeros((h, w, 3), dtype=np.uint8)
    out = draw_refinement(img, det)
    assert out.shape == img.shape and out.any()
    dbg = create_debug_visualization(img, det)
    assert dbg.shape == (h, w * 3, 3)


# ---------------------------------------------------------------------------
# CourtPredictor end-to-end
# ---------------------------------------------------------------------------

TINY = dict(in_channels=7, num_keypoints=30, image_size=128, heatmap_size=32)


@pytest.fixture
def v2_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = CourtVisionNet(pretrained=False, **TINY)
    path = tmp_path / "v2.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {}, "scheduler_state_dict": {}, "epoch": 1,
        "best_metric": 0.0, "best_metric_name": "pck", "git_sha": "x", "history": [],
        "rng_state": None, "format_version": 2,
        "config": dict(TINY, pretrained=True, heatmap_loss="js", learning_rate=1e-4),
    }, path)
    return str(path)


@pytest.fixture
def legacy_checkpoint(tmp_path):
    """Mimics the Run-7 layout: old beta key and old visibility head."""
    torch.manual_seed(0)
    sd = CourtVisionNet(pretrained=False, **TINY).state_dict()
    sd = {k: v for k, v in sd.items() if not k.startswith("kpt_head.vis")}
    sd.pop("kpt_head.beta_raw")
    sd["kpt_head.beta"] = torch.tensor(10.0)
    sd["kpt_head.visibility_conv.0.weight"] = torch.randn(128, 256, 3, 3)
    sd["kpt_head.visibility_conv.5.weight"] = torch.randn(30, 128)
    path = tmp_path / "legacy.pt"
    torch.save({"epoch": 3, "model_state_dict": sd, "optimizer_state_dict": {}, "val_loss": 1.0}, path)
    return str(path)


def _check_detection(det, h, w):
    assert isinstance(det, CourtDetection)
    assert det.keypoints.shape == (30, 2)
    assert det.raw_keypoints.shape == (30, 2)
    assert det.visibility.shape == (30,)
    assert np.all((det.visibility >= 0) & (det.visibility <= 1))
    assert det.image_size == [h, w]
    assert det.seg_mask.shape == (128, 128) and det.seg_mask.dtype == np.float32
    assert det.seg_mask_image.shape == (h, w)
    assert 0.0 <= det.confidence <= 1.0
    if det.homography is not None:
        assert det.homography.shape == (3, 3)
        assert det.inlier_mask.shape == (30,)


def test_predictor_v2_checkpoint(v2_checkpoint):
    predictor = CourtPredictor(v2_checkpoint, device=None)
    assert predictor.image_size == 128 and not predictor.legacy
    assert predictor.heatmap_kind == "logits"
    image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    det = predictor.predict(image)
    _check_detection(det, 240, 320)
    # letterbox: 320x240 -> content occupies the full width, so x in [0, 1]
    assert np.all(np.isfinite(det.raw_keypoints))

    det_tta = predictor.predict_tta(image)
    _check_detection(det_tta, 240, 320)

    det_off = CourtPredictor(v2_checkpoint, device="cpu", use_offsets=True).predict(image)
    assert det_off.raw_keypoints.shape == (30, 2)


def test_load_model_v2_rebuilds_from_config(v2_checkpoint):
    model, info = load_model(v2_checkpoint, "cpu", image_size=640)  # config wins
    assert info["image_size"] == 128 and model.heatmap_size == 32
    assert not info["legacy"]


def test_predictor_legacy_checkpoint_warns(legacy_checkpoint):
    with pytest.warns(UserWarning, match="legacy"):
        predictor = CourtPredictor(legacy_checkpoint, device="cpu", image_size=128)
    assert predictor.legacy and predictor.resize_mode == "squash"
    assert not predictor.model.backbone.normalize_input
    assert predictor.model.kpt_head.beta.item() == pytest.approx(10.0, rel=1e-3)
    det = predictor.predict(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
    _check_detection(det, 100, 150)


def test_predictor_bare_state_dict(tmp_path):
    model = CourtVisionNet(pretrained=False, **TINY)
    path = tmp_path / "raw.pt"
    torch.save(model.state_dict(), path)
    predictor = CourtPredictor(str(path), device="cpu", image_size=128, heatmap_size=32)
    det = predictor.predict(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    assert det.keypoints.shape == (30, 2)


def test_predictor_output_is_visualizable(v2_checkpoint):
    predictor = CourtPredictor(v2_checkpoint, device="cpu")
    image = np.random.randint(0, 255, (128, 96, 3), dtype=np.uint8)
    det = predictor.predict(image)
    assert draw_court_overlay(image, det).shape == image.shape
    assert create_debug_visualization(image, det).shape == (128, 96 * 3, 3)


def test_tta_alignment_relabels_mirrored_view(v2_checkpoint):
    """A mirror view, un-flipped in x, comes back with col_flip labels."""
    predictor = CourtPredictor(v2_checkpoint, device="cpu")
    w, h = 640, 480
    k0 = _project_norm(_camera_homography(w, h), w, h)
    v0 = np.ones(30)
    kf = k0[SYMMETRY_PERMS["col_flip"]]  # what the flipped pass looks like after x un-flip
    perm = predictor._align_flipped(k0, v0, kf, v0, (h, w))
    np.testing.assert_allclose(kf[perm], k0)

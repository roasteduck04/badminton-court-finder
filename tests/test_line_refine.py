"""Tests for src.inference.line_refine (C1 line-aware refinement, C2
multi-hypothesis verification) and their CourtPredictor wiring.

Two kinds of input stand in for a trained model's segmentation:
GT-derived masks (``render_line_mask`` of the GT keypoints, the seg training
target) and synthetic images with a known homography, segmented by a
classical white-line detector.
"""

import cv2
import numpy as np
import pytest
import torch

from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, COURT_WIDTH_DOUBLES, get_court_lines
from src.evaluation.metrics import render_line_mask
from src.inference.line_refine import (
    PAINTED_LINES,
    assign_lines,
    canvas_affine,
    content_region,
    line_agreement,
    line_pixels,
    project,
    refine_homography,
    render_model_lines,
    select_homography,
)
from src.inference.predict import (
    CourtDetection,
    CourtPredictor,
    estimate_court_homography,
    fit_court_homography,
)
from src.models.courtvisionnet import CourtVisionNet
from src.preprocessing.letterbox import content_mask, keypoints_to_letterbox, letterbox

SIZE = 640
W, H_IMG = 1280, 720


def _camera(w=W, h=H_IMG, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dst = np.array([[0.15 * w, 0.9 * h], [0.85 * w, 0.9 * h],
                    [0.35 * w, 0.25 * h], [0.65 * w, 0.25 * h]])
    dst += rng.uniform(-jitter, jitter, dst.shape)
    H, _ = cv2.findHomography(COURT_KEYPOINTS_TEMPLATE[[0, 4, 25, 29]], dst, 0)
    return H


def _gt_seg(H, w=W, h=H_IMG):
    """GT-derived seg: the seg training target rendered from GT keypoints."""
    _, info = letterbox(np.zeros((h, w), np.uint8), SIZE)
    kps = project(H, COURT_KEYPOINTS_TEMPLATE)[0] / [w, h]
    canvas_kps = keypoints_to_letterbox(kps, info)
    return render_line_mask(canvas_kps, np.ones(30), SIZE, 5).astype(np.float32), kps


def _noisy(kps, seed, sigma_px=8.0, outlier_rate=0.15, w=W, h=H_IMG):
    rng = np.random.default_rng(seed)
    out = kps + rng.normal(0, sigma_px, kps.shape) / [w, h]
    bad = rng.random(len(kps)) < outlier_rate
    out[bad] += rng.uniform(-0.05, 0.05, (bad.sum(), 2))
    return out


def _err(H, kps, w=W, h=H_IMG):
    return np.linalg.norm(project(H, COURT_KEYPOINTS_TEMPLATE)[0] - kps * [w, h], axis=1).mean()


def _synthetic_image(H, w=W, h=H_IMG, seed=0):
    """Green floor, white court lines under H, a neighbouring court, players."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (60, 110, 50), np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 10, (h, w, 3)).astype(np.float32), (0, 0), 3)
    for dy in (0.0, COURT_WIDTH_DOUBLES + 2.0):
        Hs = H @ np.array([[1, 0, 0], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        P, depth = project(Hs, COURT_KEYPOINTS_TEMPLATE)
        for a, b in PAINTED_LINES:
            if depth[a] > 0 and depth[b] > 0:
                pa, pb = np.clip(P[a], -1e4, 1e4), np.clip(P[b], -1e4, 1e4)
                cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb),
                         (235, 235, 235), 3, cv2.LINE_AA)
    for _ in range(5):
        c = rng.uniform([0, 0.3 * h], [w, h])
        cv2.rectangle(img, (int(c[0]) - 12, int(c[1]) - 70), (int(c[0]) + 12, int(c[1])),
                      (40, 40, 160), -1)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def _classical_seg(image):
    canvas, _ = letterbox(image, SIZE)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    return np.clip((th - 25.0) / 40.0, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hw", [(720, 1280), (1000, 600), (480, 480)])
def test_canvas_affine_matches_letterbox(hw):
    h, w = hw
    _, info = letterbox(np.zeros((h, w), np.uint8), SIZE)
    A = canvas_affine((h, w), SIZE)
    kps = np.random.default_rng(0).random((30, 2))
    via_affine = project(A, kps * [w, h])[0] / SIZE
    np.testing.assert_allclose(via_affine, keypoints_to_letterbox(kps, info), atol=1e-12)
    assert np.array_equal(content_region(A, (h, w), SIZE), content_mask(info).astype(bool))


def test_canvas_affine_squash():
    A = canvas_affine((720, 1280), SIZE, "squash")
    np.testing.assert_allclose(project(A, np.array([[1280.0, 720.0]]))[0], [[SIZE, SIZE]])
    assert content_region(A, (720, 1280), SIZE).all()


def test_painted_lines_exclude_net_segment():
    pairs = {tuple(sorted(p)) for p in PAINTED_LINES.tolist()}
    assert (12, 17) not in pairs and len(pairs) == len(get_court_lines()) - 1


def test_render_model_lines_matches_gt_target():
    H = _camera()
    seg, _ = _gt_seg(H)
    A = canvas_affine((H_IMG, W), SIZE)
    model = render_model_lines(A @ H, SIZE, 5)
    inter = (model & (seg > 0.5)).sum()
    assert inter / ((model > 0) | (seg > 0.5)).sum() > 0.9


def test_render_model_lines_handles_points_behind_camera():
    H = _camera()
    # Move the camera so the far court half is behind it.
    H_behind = H @ np.array([[1, 0, 0], [0, 1, 0], [-0.2, 0, 1]], dtype=np.float64)
    _, depth = project(H_behind, COURT_KEYPOINTS_TEMPLATE)
    assert (depth < 0).any() and (depth > 0).any()
    A = canvas_affine((H_IMG, W), SIZE)
    mask = render_model_lines(A @ H_behind, SIZE)
    assert mask.shape == (SIZE, SIZE) and mask.dtype == np.uint8


def test_assign_lines_gates_far_points():
    H = _camera()
    A = canvas_affine((H_IMG, W), SIZE)
    Hc = A @ H
    P = project(Hc, COURT_KEYPOINTS_TEMPLATE)[0]
    mid = (P[0] + P[5]) / 2                     # on segment (0, 5)
    far = np.array([5.0, 5.0])                  # canvas corner, far from all lines
    idx, dist = assign_lines(Hc, np.stack([mid, far]), gate=4.0)
    assert tuple(sorted(PAINTED_LINES[idx[0]])) == (0, 5) and dist[0] < 1e-6
    assert idx[1] == -1


def test_line_pixels_subsample_is_deterministic():
    seg = np.zeros((64, 64), np.float32)
    seg[10:50, 20] = 1.0
    pts = line_pixels(seg, max_points=10, seed=3)
    assert pts.shape == (10, 2) and np.all(pts[:, 0] == 20.5)
    np.testing.assert_array_equal(pts, line_pixels(seg, max_points=10, seed=3))
    valid = np.zeros_like(seg, bool)
    assert len(line_pixels(seg, valid=valid)) == 0


# ---------------------------------------------------------------------------
# Verification score (C2)
# ---------------------------------------------------------------------------

def test_line_agreement_true_vs_wrong_homography():
    H = _camera()
    seg, _ = _gt_seg(H)
    A = canvas_affine((H_IMG, W), SIZE)
    valid = content_region(A, (H_IMG, W), SIZE)
    good = line_agreement(A @ H, seg, valid=valid)
    assert good["f1"] > 0.9 and good["precision"] > 0.9 and good["recall"] > 0.9
    shift = np.array([[1, 0, 25], [0, 1, 15], [0, 0, 1]], dtype=np.float64)
    bad = line_agreement(A @ shift @ H, seg, valid=valid)
    assert bad["f1"] < 0.5 * good["f1"]
    empty = line_agreement(A @ H, np.zeros_like(seg), valid=valid)
    assert empty["f1"] == 0.0 and empty["n_seg"] == 0


# ---------------------------------------------------------------------------
# Refinement (C1)
# ---------------------------------------------------------------------------

def test_refinement_reduces_error_gt_derived():
    """Noisy keypoints + outliers, GT-derived line mask: refinement must beat
    the MAGSAC fit on average and on most images."""
    H = _camera()
    seg, gt = _gt_seg(H)
    A = canvas_affine((H_IMG, W), SIZE)
    valid = content_region(A, (H_IMG, W), SIZE)
    before, after = [], []
    for seed in range(8):
        kps = _noisy(gt, seed)
        H0, inl, status = fit_court_homography(kps, np.ones(30, bool), (H_IMG, W))
        assert status == "ok"
        H1, info = refine_homography(H0, kps, inl, (H_IMG, W), seg, A, valid=valid)
        assert info["status"] == "ok" and info["n_line"] > 500
        assert len(info["rounds"]) == 3
        before.append(_err(H0, gt))
        after.append(_err(H1, gt))
    before, after = np.array(before), np.array(after)
    assert after.mean() < 0.8 * before.mean()
    assert (after < before).mean() >= 0.75


def test_refinement_reduces_error_synthetic_image():
    """Known H, classical seg of a rendered image (with a neighbouring court
    and players): no GT-derived mask involved."""
    H = _camera(jitter=10, seed=1)
    gt = project(H, COURT_KEYPOINTS_TEMPLATE)[0] / [W, H_IMG]
    seg = _classical_seg(_synthetic_image(H))
    A = canvas_affine((H_IMG, W), SIZE)
    valid = content_region(A, (H_IMG, W), SIZE)
    assert line_agreement(A @ H, seg, valid=valid)["recall"] > 0.8
    before, after = [], []
    for seed in range(6):
        kps = _noisy(gt, 100 + seed)
        H0, inl, _ = fit_court_homography(kps, np.ones(30, bool), (H_IMG, W))
        H1, info = refine_homography(H0, kps, inl, (H_IMG, W), seg, A, valid=valid)
        assert info["status"] == "ok"
        before.append(_err(H0, gt))
        after.append(_err(H1, gt))
    assert all(x < y for x, y in zip(after, before))
    assert np.mean(after) < 0.9 * np.mean(before)


def test_refinement_ignores_distractor_lines():
    """A parallel line 40 px away (another court / an ad board) is gated out."""
    H = _camera()
    seg, gt = _gt_seg(H)
    A = canvas_affine((H_IMG, W), SIZE)
    Hc = A @ H
    P = project(Hc, COURT_KEYPOINTS_TEMPLATE)[0]
    off = np.array([0.0, -40.0])
    cv2.line(seg, tuple(int(v) for v in P[25] + off), tuple(int(v) for v in P[29] + off), 1.0, 5)
    kps = _noisy(gt, 7)
    H0, inl, _ = fit_court_homography(kps, np.ones(30, bool), (H_IMG, W))
    H1, info = refine_homography(H0, kps, inl, (H_IMG, W), seg, A)
    assert info["status"] == "ok"
    assert _err(H1, gt) < _err(H0, gt)


def test_refinement_options_and_fallbacks():
    H = _camera()
    seg, gt = _gt_seg(H)
    A = canvas_affine((H_IMG, W), SIZE)
    kps = _noisy(gt, 3)
    H0, inl, _ = fit_court_homography(kps, np.ones(30, bool), (H_IMG, W))
    for loss in ("huber", "cauchy", "linear"):
        H1, info = refine_homography(H0, kps, inl, (H_IMG, W), seg, A, loss=loss)
        assert info["status"] == "ok" and np.all(np.isfinite(H1))
    # alpha = 0: keypoints only, no line pixels used
    _, info = refine_homography(H0, kps, inl, (H_IMG, W), seg, A, alpha=0.0)
    assert info["status"] == "ok" and info["n_line"] == 0
    # alpha = 1 with no seg: nothing to fit
    H1, info = refine_homography(H0, kps, inl, (H_IMG, W), np.zeros_like(seg), A, alpha=1.0)
    assert info["status"].startswith("skipped") and H1 is H0
    # an impossibly tight shift budget reverts to the input
    H1, info = refine_homography(H0, kps, inl, (H_IMG, W), seg, A, max_shift=1e-3)
    assert info["status"].startswith("reverted") and H1 is H0
    # too few inliers
    few = np.zeros(30, bool)
    few[:3] = True
    _, info = refine_homography(H0, kps, few, (H_IMG, W), seg, A)
    assert info["status"].startswith("skipped")
    with pytest.raises(ValueError):
        refine_homography(H0, kps, inl, (H_IMG, W), seg, A, loss="l2")


def test_refinement_keeps_orientation_and_scale():
    H = _camera()
    seg, gt = _gt_seg(H)
    A = canvas_affine((H_IMG, W), SIZE)
    kps = _noisy(gt, 5)
    H0, inl, _ = fit_court_homography(kps, np.ones(30, bool), (H_IMG, W))
    H1, _ = refine_homography(H0, kps, inl, (H_IMG, W), seg, A)
    assert np.all(project(H1, COURT_KEYPOINTS_TEMPLATE)[1] > 0)
    assert np.linalg.norm(H1) == pytest.approx(np.linalg.norm(H0))


# ---------------------------------------------------------------------------
# Multi-hypothesis (C2)
# ---------------------------------------------------------------------------

def _fake_fit(table):
    """fit_fn returning table[(subset size, threshold)] -> H (all inliers)."""
    def fit(keypoints, mask, image_hw, thr, min_inliers):
        H = table.get((int(np.sum(mask)), thr))
        if H is None:
            return None, None, "fit_failed"
        return H, np.asarray(mask, bool).copy(), "ok"
    return fit


def test_select_homography_prefers_line_agreement():
    H = _camera()
    seg, gt = _gt_seg(H)
    A = canvas_affine((H_IMG, W), SIZE)
    wrong = np.array([[1, 0, 30], [0, 1, 20], [0, 0, 1]], dtype=np.float64) @ H
    conf = np.linspace(0, 1, 30)
    table = {(30, 0.01): wrong, (30, 0.02): wrong * 1.0001, (15, 0.01): H}
    Hs, inl, status, score, hyps = select_homography(
        gt, np.ones(30, bool), conf, (H_IMG, W), seg, A, _fake_fit(table),
        thresholds=(0.01, 0.02))
    assert Hs is H and status.startswith("ok (confident")
    assert score["f1"] > 0.9 and inl.sum() >= 4
    assert any(h[0] == "all" for h in hyps)
    # Without conf there is no "confident" subset: the default fit stays.
    Hs, _, status, _, _ = select_homography(
        gt, np.ones(30, bool), None, (H_IMG, W), seg, A, _fake_fit(table),
        thresholds=(0.01, 0.02))
    assert Hs is wrong and status == "ok"


def test_select_homography_keeps_default_within_margin():
    H = _camera()
    seg, gt = _gt_seg(H)
    A = canvas_affine((H_IMG, W), SIZE)
    near = np.array([[1, 0, 0.3], [0, 1, 0], [0, 0, 1]], dtype=np.float64) @ H
    table = {(30, 0.01): H, (15, 0.01): near}
    Hs, _, status, _, _ = select_homography(
        gt, np.ones(30, bool), np.linspace(0, 1, 30), (H_IMG, W), seg, A, _fake_fit(table),
        thresholds=(0.01,), margin=0.05)
    assert Hs is H and status == "ok"


def test_select_homography_recovers_failed_default():
    H = _camera()
    seg, gt = _gt_seg(H)
    A = canvas_affine((H_IMG, W), SIZE)
    Hs, _, status, _, _ = select_homography(
        gt, np.ones(30, bool), None, (H_IMG, W), seg, A, _fake_fit({(30, 0.02): H}),
        thresholds=(0.01, 0.02))
    assert Hs is H and status == "ok (all@0.02)"
    Hs, inl, status, score, hyps = select_homography(
        gt, np.ones(30, bool), None, (H_IMG, W), seg, A, _fake_fit({}))
    assert Hs is None and inl is None and status == "fit_failed" and hyps == []


def test_select_homography_with_real_fits():
    H = _camera()
    seg, gt = _gt_seg(H)
    A = canvas_affine((H_IMG, W), SIZE)
    kps = _noisy(gt, 11)
    Hs, inl, status, score, hyps = select_homography(
        kps, np.ones(30, bool), np.random.default_rng(0).random(30), (H_IMG, W), seg, A,
        fit_court_homography, subsets=("all", "confident", "rows0-2", "rows3-5"))
    assert Hs is not None and status.startswith("ok") and score["f1"] > 0.5
    assert {h[0] for h in hyps} >= {"all", "confident"}
    with pytest.raises(ValueError):
        select_homography(kps, np.ones(30, bool), None, (H_IMG, W), seg, A,
                          fit_court_homography, subsets=("bogus",))


# ---------------------------------------------------------------------------
# estimate_court_homography / reject flag
# ---------------------------------------------------------------------------

def test_estimate_without_seg_is_plain_magsac():
    H = _camera()
    _, gt = _gt_seg(H)
    kps = _noisy(gt, 2)
    fit = estimate_court_homography(kps, np.ones(30, bool), (H_IMG, W), line_refine=True,
                                    multi_hypothesis=True, reject_threshold=0.5)
    H0, inl0, st0 = fit_court_homography(kps, np.ones(30, bool), (H_IMG, W))
    np.testing.assert_allclose(fit["H"], H0)
    assert fit["status"] == st0 and fit["line_score"] is None and not fit["rejected"]


def test_estimate_full_pipeline_and_reject():
    H = _camera()
    seg, gt = _gt_seg(H)
    A = canvas_affine((H_IMG, W), SIZE)
    kps = _noisy(gt, 4)
    cands = np.ones(30, bool)
    base = estimate_court_homography(kps, cands, (H_IMG, W), seg, A)
    full = estimate_court_homography(kps, cands, (H_IMG, W), seg, A, line_refine=True,
                                     multi_hypothesis=True, reject_threshold=0.3)
    assert base["refine_status"] == "not_run" and full["refine_status"] == "ok"
    assert full["line_score"] > base["line_score"] > 0.3 and not full["rejected"]
    assert _err(full["H"], gt) < _err(base["H"], gt)
    # A segmentation that shows no court: the fit is rejected.
    rej = estimate_court_homography(kps, cands, (H_IMG, W), np.zeros_like(seg), A,
                                    reject_threshold=0.3)
    assert rej["rejected"] and rej["H"] is None and rej["inliers"] is None
    assert rej["status"].startswith("rejected") and rej["line_score"] == 0.0


# ---------------------------------------------------------------------------
# CourtPredictor wiring
# ---------------------------------------------------------------------------

TINY = dict(in_channels=7, num_keypoints=30, image_size=128, heatmap_size=32)


@pytest.fixture
def v2_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = CourtVisionNet(pretrained=False, **TINY)
    path = tmp_path / "v2.pt"
    torch.save({"model_state_dict": model.state_dict(), "format_version": 2,
                "config": dict(TINY, heatmap_loss="js")}, path)
    return str(path)


def test_court_detection_line_fields_default():
    det = CourtDetection(keypoints=np.zeros((30, 2)), visibility=np.zeros(30))
    assert det.line_score is None and det.rejected is False and det.refine_status == "not_run"


def test_predictor_line_options(v2_checkpoint, monkeypatch):
    image = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    pred = CourtPredictor(v2_checkpoint, device="cpu")
    assert pred.line_refine and pred.multi_hypothesis and pred.reject_threshold is None

    # Make the random model's output a clean court so a homography is found.
    h, w = image.shape[:2]
    Hc = _camera(w, h)
    gt = project(Hc, COURT_KEYPOINTS_TEMPLATE)[0] / [w, h]
    canvas, info = letterbox(image, 128)
    seg = render_line_mask(keypoints_to_letterbox(gt, info), np.ones(30), 128, 2).astype(np.float32)
    noisy = _noisy(gt, 0, sigma_px=2.0, outlier_rate=0.0, w=w, h=h)

    def fake_run(self, img):
        return noisy, np.ones(30), np.ones(30), seg, np.zeros((h, w), np.float32)

    monkeypatch.setattr(CourtPredictor, "_run", fake_run)
    det = pred.predict(image)
    assert det.homography is not None and det.refine_status == "ok"
    assert 0.0 < det.line_score <= 1.0 and not det.rejected

    off = CourtPredictor(v2_checkpoint, device="cpu", line_refine=False, multi_hypothesis=False)
    det_off = off.predict(image)
    assert det_off.refine_status == "not_run" and det_off.homography is not None
    assert det_off.line_score is not None

    strict = CourtPredictor(v2_checkpoint, device="cpu", reject_threshold=1.01,
                            line_options={"alpha": 0.7}, hypothesis_options={"margin": 0.1})
    det_rej = strict.predict(image)
    assert det_rej.rejected and det_rej.homography is None
    assert det_rej.homography_status.startswith("rejected")
    np.testing.assert_allclose(det_rej.keypoints, noisy)


def test_predictor_runs_end_to_end_with_line_refine(v2_checkpoint):
    image = np.random.default_rng(1).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    for tta in (False, True):
        det = CourtPredictor(v2_checkpoint, device="cpu", line_refine=True,
                             multi_hypothesis=True, reject_threshold=0.2).predict(image, tta=tta)
        assert det.keypoints.shape == (30, 2)
        assert isinstance(det.refine_status, str) and isinstance(det.rejected, bool)

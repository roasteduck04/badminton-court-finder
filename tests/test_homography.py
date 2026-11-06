"""Tests for the differentiable DLT in src.models.homography."""

import cv2
import numpy as np
import pytest
import torch

from src.court_geometry import COURT_KEYPOINTS_TEMPLATE
from src.models.homography import fit_homography, project

TPL = torch.tensor(COURT_KEYPOINTS_TEMPLATE)


def _camera_homography(seed=0):
    """A plausible template(metres) -> 640 px image perspective homography."""
    rng = np.random.default_rng(seed)
    src = COURT_KEYPOINTS_TEMPLATE[[0, 4, 25, 29]]
    dst = np.array([[150, 500], [520, 520], [250, 150], [420, 160]], dtype=np.float64)
    dst += rng.uniform(-30, 30, dst.shape)
    H, _ = cv2.findHomography(src, dst, 0)
    return H


def _apply(H, pts):
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return ph[:, :2] / ph[:, 2:]


@pytest.mark.parametrize("method", ["solve", "svd"])
def test_exact_fit_matches_cv2(method):
    H_true = _camera_homography()
    dst = _apply(H_true, COURT_KEYPOINTS_TEMPLATE)
    H = fit_homography(TPL, torch.tensor(dst), method=method).numpy()
    H_cv, _ = cv2.findHomography(COURT_KEYPOINTS_TEMPLATE, dst, 0)
    np.testing.assert_allclose(H, H_cv / H_cv[2, 2], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("method", ["solve", "svd"])
def test_noisy_fit_close_to_cv2(method):
    rng = np.random.default_rng(1)
    H_true = _camera_homography(1)
    dst = _apply(H_true, COURT_KEYPOINTS_TEMPLATE) + rng.normal(0, 1.0, (30, 2))
    H = fit_homography(TPL, torch.tensor(dst), method=method).numpy()
    H_cv, _ = cv2.findHomography(COURT_KEYPOINTS_TEMPLATE, dst, 0)
    diff = np.linalg.norm(_apply(H, COURT_KEYPOINTS_TEMPLATE) - _apply(H_cv, COURT_KEYPOINTS_TEMPLATE), axis=1)
    assert diff.max() < 0.5  # px; cv2 additionally refines with LM


def test_zero_weight_points_are_ignored():
    H_true = _camera_homography(2)
    dst = _apply(H_true, COURT_KEYPOINTS_TEMPLATE)
    w = np.zeros(30)
    w[[0, 2, 4, 12, 17, 25, 27, 29]] = 1.0
    dst_bad = dst.copy()
    dst_bad[w == 0] = 1e4 * np.random.default_rng(0).random(((w == 0).sum(), 2))
    H = fit_homography(TPL, torch.tensor(dst_bad), torch.tensor(w)).numpy()
    np.testing.assert_allclose(H, H_true / H_true[2, 2], rtol=1e-5, atol=1e-7)


def test_batched_equals_individual():
    dsts = [_apply(_camera_homography(s), COURT_KEYPOINTS_TEMPLATE) for s in range(3)]
    batch = torch.tensor(np.stack(dsts))
    Hb = fit_homography(TPL, batch)
    for i, d in enumerate(dsts):
        torch.testing.assert_close(Hb[i], fit_homography(TPL, torch.tensor(d)))


def test_project_returns_depth():
    H_true = torch.tensor(_camera_homography())
    pts, w = project(H_true, TPL)
    assert pts.shape == (30, 2) and w.shape == (30,)
    assert torch.all(w > 0)


@pytest.mark.parametrize("method", ["solve", "svd"])
def test_gradcheck(method):
    torch.manual_seed(0)
    idx = [0, 4, 12, 25, 29]
    src = TPL[idx].clone()
    base = torch.tensor(_apply(_camera_homography(), COURT_KEYPOINTS_TEMPLATE[idx]))
    dst = (base + torch.randn_like(base)).requires_grad_(True)
    w = torch.tensor([1.0, 0.8, 1.0, 0.6, 1.0], dtype=torch.float64, requires_grad=True)
    query = TPL[[2, 7, 27]].clone()

    def f(d, ww):
        H = fit_homography(src, d, ww, method=method)
        return project(H, query)[0]

    assert torch.autograd.gradcheck(f, (dst, w), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_solve_gradient_finite_for_collapsed_points():
    """Early in training all predictions can sit on one spot: still finite."""
    dst = torch.full((1, 30, 2), 320.0, dtype=torch.float64) + 1e-3 * torch.randn(1, 30, 2, dtype=torch.float64)
    dst.requires_grad_(True)
    H = fit_homography(TPL, dst)
    reproj, _ = project(H, TPL)
    loss = (reproj - dst).abs().sum()
    loss.backward()
    assert torch.isfinite(H).all()
    assert torch.isfinite(dst.grad).all()

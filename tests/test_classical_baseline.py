"""Tests for scripts/classical_baseline.py on synthetic court images.

Court lines (40 mm wide, so their image width follows the perspective) are
drawn through a known template->image homography; the detector must recover
the canonicalised keypoints within a few pixels.
"""

import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import classical_baseline as cb  # noqa: E402
from src.court_geometry import (  # noqa: E402
    COURT_KEYPOINTS_TEMPLATE,
    CORNER_INDICES,
    canonicalize,
    permute_keypoints,
    project_points,
)

W, H = 1280, 720
LINE_W = 0.04
FLOOR = (90, 140, 40)       # BGR green
WHITE = (235, 235, 235)


def _homography(corners_px):
    """Template->image H from image positions of K0, K25, K29, K4."""
    src = COURT_KEYPOINTS_TEMPLATE[CORNER_INDICES].astype(np.float32)
    return cv2.getPerspectiveTransform(src, np.asarray(corners_px, dtype=np.float32))


def _draw_band(img, Hm, p, q, colour, width=LINE_W):
    """Fill the projected rectangle of a model line segment p->q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = (q - p) / np.linalg.norm(q - p)
    n = np.array([-d[1], d[0]]) * width / 2
    p, q = p - d * width / 2, q + d * width / 2
    poly = project_points(Hm, np.array([p + n, q + n, q - n, p - n]))
    cv2.fillConvexPoly(img, np.rint(poly * 16).astype(np.int32), colour, cv2.LINE_AA, shift=4)


def render(Hm, floor=FLOOR, colour=WHITE, extra=(), seed=0):
    img = np.empty((H, W, 3), np.uint8)
    img[:] = floor
    for k in range(len(cb.SEG_P) if Hm is not None else 0):
        _draw_band(img, Hm, cb.SEG_P[k], cb.SEG_Q[k], colour)
    for p, q in extra:
        _draw_band(img, Hm, p, q, colour)
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, 4, img.shape)
    return cv2.GaussianBlur(np.clip(noisy, 0, 255).astype(np.uint8), (3, 3), 0)


def gt_keypoints(Hm):
    """Canonical GT keypoints and in-image visibility."""
    kps = project_points(Hm, COURT_KEYPOINTS_TEMPLATE) / np.array([W, H])
    vis = np.all((kps >= 0) & (kps <= 1), axis=1).astype(float)
    _, _, perm = canonicalize(kps, np.ones(30), image_size=(H, W))
    return permute_keypoints(kps, vis, perm)


VIEWS = {
    # image positions of K0, K25, K29, K4
    "end_on": [(330, 690), (520, 80), (760, 80), (950, 690)],
    "side": [(120, 620), (1160, 620), (980, 260), (300, 260)],
    "oblique": [(140, 520), (760, 690), (1180, 260), (560, 140)],
    "partial": [(-250, 1100), (420, -120), (860, -120), (1530, 1100)],
}


# The default failure threshold is calibrated on real val images and rejects
# the narrow synthetic end-on court (score ~2.25, see docs/classical-baseline.md);
# these tests check the geometry, so they accept any clear fit.
TEST_THRESHOLD = 1.0


def _detector(**kw):
    return cb.ClassicalCourtDetector(cb.ClassicalParams(score_threshold=TEST_THRESHOLD, **kw))


@pytest.fixture(scope="module")
def detector():
    return _detector()


def _check(det, img, Hm, tol_px=3.0):
    out = det.predict(img)
    assert out["status"] == "ok", out["status"]
    gt, gvis = gt_keypoints(Hm)
    vis = gvis > 0
    assert vis.sum() >= 4
    err = np.linalg.norm((out["keypoints"][vis] - gt[vis]) * np.array([W, H]), axis=1)
    assert np.all(err < tol_px), f"max error {err.max():.2f}px"
    # predicted visibility = in image
    assert np.array_equal(out["visibility"] > 0, vis)
    return out


@pytest.mark.parametrize("view", sorted(VIEWS))
def test_recovers_keypoints(detector, view):
    Hm = _homography(VIEWS[view])
    _check(detector, render(Hm), Hm)


def test_distractor_lines(detector):
    """An adjacent court's sideline and another sport's diagonal line."""
    Hm = _homography(VIEWS["oblique"])
    extra = [((-1.0, 7.0), (14.4, 7.0)), ((-1.0, -0.8), (9.0, 6.6))]
    _check(detector, render(Hm, extra=extra), Hm)


def test_coloured_lines_chroma_mode():
    """Yellow lines darker than a light-grey floor: luminance misses them,
    the chroma distance to the floor colour does not."""
    Hm = _homography(VIEWS["end_on"])
    img = render(Hm, floor=(205, 205, 205), colour=(0, 190, 215))
    _check(_detector(line_mode="chroma"), img, Hm)
    assert _detector().predict(img)["status"] != "ok"


def test_blank_image_fails(detector):
    img = render(None)
    out = detector.predict(img)
    assert out["status"] != "ok"
    assert np.all(np.isnan(out["keypoints"]))
    assert np.all(out["visibility"] == 0)
    assert out["homography"] is None


def test_low_score_fails():
    """A fit below the score threshold is reported as a failure."""
    Hm = _homography(VIEWS["side"])
    out = cb.ClassicalCourtDetector(cb.ClassicalParams(score_threshold=100.0)).predict(render(Hm))
    assert out["status"] == "low_score"
    assert out["score_norm"] is not None and out["score_norm"] < 100.0
    assert np.all(np.isnan(out["keypoints"])) and out["homography"] is None


def test_split_families_perpendicular_sets():
    angles = np.deg2rad([0, 5, -4, 88, 92, 95])
    lines = np.stack([np.cos(angles), np.sin(angles), np.zeros(6)], axis=1)
    fam = cb.split_families(lines)
    assert list(fam) == [False, False, False, True, True, True]


def test_square_to_quad_maps_corners():
    q = np.array([[[10, 20], [200, 30], [180, 220], [5, 160]]], dtype=float)
    Hq = cb.square_to_quad(q)[0]
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert np.allclose(project_points(Hq, unit), q[0])

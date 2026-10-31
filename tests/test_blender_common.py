"""blender_common mirrors court_geometry's constants (Blender has no cv2)."""

import random

import numpy as np

from src import court_geometry as cg
from src.tools import blender_common as bc


def test_court_dimensions_match():
    assert bc.COURT_LENGTH == cg.COURT_LENGTH
    assert bc.COURT_WIDTH == cg.COURT_WIDTH_DOUBLES
    assert abs(bc.SINGLES_OFFSET - cg.SINGLES_SIDELINE_OFFSET) < 1e-12
    assert bc.NET_POS == cg.NET_POSITION
    assert bc.SHORT_SERVICE == cg.SHORT_SERVICE_LINE
    assert bc.LONG_SERVICE_DBL == cg.LONG_SERVICE_LINE_DOUBLES
    assert bc.NUM_KEYPOINTS == cg.NUM_KEYPOINTS


def test_keypoint_template_order_matches():
    tpl = np.array(bc.KEYPOINTS_TEMPLATE)
    assert tpl.shape == (30, 2)
    np.testing.assert_allclose(tpl, cg.COURT_KEYPOINTS_TEMPLATE, atol=1e-12)
    assert sorted(bc.CORNER_INDICES) == sorted(cg.CORNER_INDICES)


def test_homography_support_matches_court_geometry():
    rng = random.Random(0)
    for _ in range(300):
        p = rng.choice([0.1, 0.2, 0.4, 0.7])
        vis = [1 if rng.random() < p else 0 for _ in range(30)]
        assert bc.has_homography_support(vis) == cg.has_homography_support(np.array(vis))
    assert not bc.has_homography_support([1, 1, 1, 1, 1] + [0] * 25)  # one row only


def test_image_seed_deterministic_and_distinct():
    assert bc.image_seed(42, 7) == bc.image_seed(42, 7)
    seeds = {bc.image_seed(42, i) for i in range(1, 2001)}
    assert len(seeds) == 2000
    assert bc.image_seed(42, 1) != bc.image_seed(43, 1)
    assert all(0 <= s < 2 ** 31 for s in seeds)

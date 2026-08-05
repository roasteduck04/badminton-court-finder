import numpy as np
import pytest
from src.court_geometry import (
    COURT_KEYPOINTS_TEMPLATE,
    COURT_WIDTH_DOUBLES,
    COURT_LENGTH,
    get_court_lines,
    compute_homography,
    project_points,
    validate_quadrilateral,
    generate_line_mask,
)


def test_template_has_14_keypoints():
    assert COURT_KEYPOINTS_TEMPLATE.shape == (14, 2)


def test_template_dimensions_match_spec():
    # K0 is top-left (0,0), K2 is bottom-right (13.4, 6.1)
    k0 = COURT_KEYPOINTS_TEMPLATE[0]
    k2 = COURT_KEYPOINTS_TEMPLATE[2]
    length = abs(k2[0] - k0[0])
    width = abs(k2[1] - k0[1])
    assert abs(length - 13.4) < 0.01
    assert abs(width - 6.1) < 0.01


def test_court_lines_returns_pairs():
    lines = get_court_lines()
    assert len(lines) > 0
    for start_idx, end_idx in lines:
        assert 0 <= start_idx < 14
        assert 0 <= end_idx < 14


def test_compute_homography_identity():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    H = compute_homography(pts, pts)
    assert H.shape == (3, 3)
    # Should be close to identity
    np.testing.assert_allclose(H / H[2, 2], np.eye(3), atol=1e-6)


def test_project_points_roundtrip():
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    dst = np.array([[10, 10], [200, 20], [190, 180], [15, 170]], dtype=np.float64)
    H = compute_homography(src, dst)
    projected = project_points(H, src)
    np.testing.assert_allclose(projected, dst, atol=1.0)


def test_validate_quadrilateral_valid():
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    assert validate_quadrilateral(corners) is True


def test_validate_quadrilateral_invalid_crossed():
    corners = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=np.float64)
    assert validate_quadrilateral(corners) is False


def test_generate_line_mask():
    keypoints = np.array([
        [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
        [-1, -1], [-1, -1], [-1, -1], [-1, -1],
        [-1, -1], [-1, -1], [-1, -1], [-1, -1],
        [-1, -1], [-1, -1],
    ], dtype=np.float64)
    visibility = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mask = generate_line_mask(keypoints, visibility, width=640, height=640, line_thickness=3)
    assert mask.shape == (640, 640)
    assert mask.dtype == np.uint8
    assert mask.max() == 255
    assert mask.min() == 0

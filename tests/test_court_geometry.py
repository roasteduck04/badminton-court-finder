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


def test_template_has_31_keypoints():
    assert COURT_KEYPOINTS_TEMPLATE.shape == (31, 2)


def test_template_dimensions_match_spec():
    # K0 is top-left (0,0), K30 is bottom-right (13.4, 6.1)
    k0 = COURT_KEYPOINTS_TEMPLATE[0]
    k30 = COURT_KEYPOINTS_TEMPLATE[30]
    length = abs(k30[0] - k0[0])
    width = abs(k30[1] - k0[1])
    assert abs(length - 13.4) < 0.01
    assert abs(width - 6.1) < 0.01


def test_court_lines_returns_pairs():
    lines = get_court_lines()
    assert len(lines) > 0
    for start_idx, end_idx in lines:
        assert 0 <= start_idx < 31
        assert 0 <= end_idx < 31


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
    kps = [[0.1, 0.1], [0.1, 0.15], [0.1, 0.85], [0.1, 0.9]]  # K0-K3
    kps += [[-1, -1]] * 27  # K4-K30
    keypoints = np.array(kps, dtype=np.float64)
    visibility = [1, 1, 1, 1] + [0] * 27
    mask = generate_line_mask(keypoints, visibility, width=640, height=640, line_thickness=3)
    assert mask.shape == (640, 640)
    assert mask.dtype == np.uint8
    assert mask.max() == 255
    assert mask.min() == 0


def test_compute_homography_insufficient_src_points():
    """Test that compute_homography raises ValueError with < 4 source points."""
    src = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)  # Only 3 points
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)  # 4 points
    with pytest.raises(ValueError, match="src_points must have at least 4 points"):
        compute_homography(src, dst)


def test_compute_homography_insufficient_dst_points():
    """Test that compute_homography raises ValueError with < 4 destination points."""
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)  # 4 points
    dst = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)  # Only 3 points
    with pytest.raises(ValueError, match="dst_points must have at least 4 points"):
        compute_homography(src, dst)


def test_compute_homography_collinear_points():
    """Test that compute_homography raises ValueError with collinear points."""
    # All points on a horizontal line (y=0)
    src = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
    dst = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
    with pytest.raises(ValueError, match="Failed to compute homography"):
        compute_homography(src, dst)


def test_project_points_with_none_homography():
    """Test that project_points raises ValueError when H is None."""
    points = np.array([[0, 0], [1, 1]], dtype=np.float64)
    with pytest.raises(ValueError, match="Homography matrix H cannot be None"):
        project_points(None, points)

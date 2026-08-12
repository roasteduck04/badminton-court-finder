import numpy as np
import pytest
from src.court_geometry import (
    COURT_KEYPOINTS_TEMPLATE,
    COURT_WIDTH_DOUBLES,
    COURT_LENGTH,
    get_court_lines,
    get_collinear_groups,
    compute_homography,
    project_points,
    validate_quadrilateral,
    generate_line_mask,
)


def test_template_has_30_keypoints():
    assert COURT_KEYPOINTS_TEMPLATE.shape == (30, 2)


def test_template_dimensions_match_spec():
    k0 = COURT_KEYPOINTS_TEMPLATE[0]   # (0, 0)
    k29 = COURT_KEYPOINTS_TEMPLATE[29]  # (13.4, 6.1)
    assert abs(k29[0] - k0[0] - 13.4) < 0.01
    assert abs(k29[1] - k0[1] - 6.1) < 0.01


def test_template_6x5_grid_structure():
    """Rows share the same x, columns share the same y."""
    tpl = COURT_KEYPOINTS_TEMPLATE
    # Row 0 (Baseline L): K0-K4 all at x=0
    for i in range(5):
        assert abs(tpl[i, 0] - 0.0) < 1e-6
    # Row 5 (Baseline R): K25-K29 all at x=13.4
    for i in range(25, 30):
        assert abs(tpl[i, 0] - 13.4) < 1e-6
    # Column 0 (Dbl-top): K0, K5, K10, K15, K20, K25 all at y=0
    for i in [0, 5, 10, 15, 20, 25]:
        assert abs(tpl[i, 1] - 0.0) < 1e-6


def test_num_keypoints_constant():
    from src.court_geometry import NUM_KEYPOINTS
    assert NUM_KEYPOINTS == 30


def test_flip_pairs_count():
    from src.court_geometry import FLIP_PAIRS
    assert len(FLIP_PAIRS) == 12
    paired = {i for pair in FLIP_PAIRS for i in pair}
    assert len(paired) == 24  # 12 pairs x 2
    # Center column excluded
    for center in [2, 7, 12, 17, 22, 27]:
        assert center not in paired


def test_corner_indices():
    from src.court_geometry import CORNER_INDICES
    assert set(CORNER_INDICES) == {0, 4, 25, 29}


def test_court_lines_indices_in_range():
    lines = get_court_lines()
    for start_idx, end_idx in lines:
        assert 0 <= start_idx < 30
        assert 0 <= end_idx < 30


def test_get_collinear_groups():
    groups = get_collinear_groups()
    # 6 rows + 5 columns = 11 collinear groups
    assert len(groups) == 11
    # Each group has at least 2 points
    for group in groups:
        assert len(group) >= 2
    # Row 0 (Baseline L) should be [0, 1, 2, 3, 4]
    assert [0, 1, 2, 3, 4] in groups


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
    kps = [[0.1, 0.1], [0.1, 0.15], [0.1, 0.5], [0.1, 0.85], [0.1, 0.9]]  # K0-K4
    kps += [[-1, -1]] * 25  # K5-K29
    keypoints = np.array(kps, dtype=np.float64)
    visibility = [1, 1, 1, 1, 1] + [0] * 25
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

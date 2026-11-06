"""Tests for the per-image auto-FOV maths (src/tools/camera_framing.py)."""

import math
import random

import numpy as np
import pytest

from src.tools.camera_framing import (
    EQUIV_DIAGONAL_MM, PHONE_ULTRAWIDE, ZOOM_RANGE, choose_focal, court_coverage,
    elevation_deg, equivalent_sensor_long_side, frame_points, half_extents, jitter_aim,
    look_rotation, sample_framing_target, to_camera_plane,
)

L, W = 13.4, 6.1
CORNERS = np.array([(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0)], float)


def _image_coords(points, cam_pos, R, focal, width, height, sensor):
    """Pixel coordinates (x right, y down) of world points."""
    uv, _ = to_camera_plane(points, cam_pos, R)
    hx, hy = half_extents(width, height, sensor)
    x = width / 2 + uv[:, 0] * focal / hx * width / 2
    y = height / 2 - uv[:, 1] * focal / hy * height / 2
    return np.stack([x, y], 1)


@pytest.mark.parametrize("w,h", [(640, 480), (1280, 720), (480, 853), (600, 600)])
def test_equivalent_sensor_has_full_frame_diagonal(w, h):
    long_mm = equivalent_sensor_long_side(w, h)
    short_mm = long_mm * min(w, h) / max(w, h)
    assert math.hypot(long_mm, short_mm) == pytest.approx(EQUIV_DIAGONAL_MM)


def test_look_rotation_is_orthonormal_and_level():
    for d, roll in [((1, 0.3, -0.4), 0.0), ((0.1, -1, -0.2), 8.0), ((0, 0, -1), 0.0)]:
        R = look_rotation(d, roll)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)
        assert np.allclose(-R[:, 2], np.asarray(d) / np.linalg.norm(d))   # looks along d
    R = look_rotation((1, 0.3, -0.4), 0.0)
    assert abs(R[2, 0]) < 1e-9          # no roll: image x axis is horizontal
    assert R[2, 1] > 0                  # image up points up


@pytest.mark.parametrize("cam_pos,w,h,margin", [
    ((-8.0, 3.0, 6.0), 1280, 720, 0.1),    # broadcast, behind the baseline
    ((6.7, -6.0, 2.0), 720, 1280, 0.05),   # portrait, side-on
    ((-3.0, -2.0, 4.0), 640, 640, 0.2),    # corner
])
def test_frame_points_fits_and_centres_the_court(cam_pos, w, h, margin):
    sensor = equivalent_sensor_long_side(w, h)
    R, f = frame_points(cam_pos, CORNERS, 0.0, w, h, sensor, margin)
    px = _image_coords(CORNERS, cam_pos, R, f, w, h, sensor)
    assert (px[:, 0] >= 0).all() and (px[:, 0] <= w).all()
    assert (px[:, 1] >= 0).all() and (px[:, 1] <= h).all()
    # Centred: equal slack on both sides along x and along y (in tan units).
    uv, _ = to_camera_plane(CORNERS, cam_pos, R)
    assert uv[:, 0].max() == pytest.approx(-uv[:, 0].min(), abs=1e-6)
    assert uv[:, 1].max() == pytest.approx(-uv[:, 1].min(), abs=1e-6)
    # Tight: the tightest side leaves exactly the requested margin.
    hx, hy = half_extents(w, h, sensor)
    fill = max(np.abs(uv[:, 0]).max() * f / hx, np.abs(uv[:, 1]).max() * f / hy)
    assert fill == pytest.approx(1 - margin, abs=1e-6)


def test_frame_points_rejects_points_behind_camera():
    # Camera 5 cm above the court centre: the corners lie in the image plane.
    assert frame_points((6.7, 3.05, 0.05), CORNERS, 0.0, 640, 480, 36.0) is None


def test_jitter_keeps_points_in_frame():
    rng = random.Random(0)
    cam, w, h = (-10.0, 3.0, 7.0), 1280, 720
    sensor = equivalent_sensor_long_side(w, h)
    R, f = frame_points(cam, CORNERS, 3.0, w, h, sensor, 0.25)
    moved = 0
    for _ in range(20):
        R2 = jitter_aim(R, 3.0, f, CORNERS, cam, w, h, sensor, rng)
        px = _image_coords(CORNERS, cam, R2, f, w, h, sensor)
        assert (px >= -1e-6).all() and (px[:, 0] <= w + 1e-6).all() and (px[:, 1] <= h + 1e-6).all()
        moved += not np.allclose(R2, R)
    assert moved > 10


def test_choose_focal_phone_and_zoom():
    rng = random.Random(1)
    for req in (8.0, 20.0, 30.0, 90.0, 300.0):
        for _ in range(50):
            f, info = choose_focal(req, "phone", rng)
            assert info["lens"] in ("ultrawide", "main", "tele")
            assert 1.0 <= info["digital_zoom"] <= 3.0
            if req < PHONE_ULTRAWIDE[0]:                 # cannot fit: widest lens, cropped
                assert info["lens"] == "ultrawide" and info["digital_zoom"] == 1.0
            else:
                assert f <= req + 1e-9                   # never tighter than required
    for req in (5.0, 60.0, 900.0):
        f, _ = choose_focal(req, "zoom", rng)
        assert ZOOM_RANGE[0] <= f <= ZOOM_RANGE[1]
    with pytest.raises(ValueError):
        choose_focal(30.0, "fisheye", rng)


def test_sample_framing_target_modes():
    rng = random.Random(3)
    modes = set()
    for _ in range(200):
        mode, pts, margin = sample_framing_target(rng, L, W)
        modes.add(mode)
        assert pts.shape[1] == 3
        if mode == "full":
            assert len(pts) == 6 and 0.0 <= margin <= 0.25    # 4 corners + 2 net-post tops
        else:
            assert len(pts) == 4 and -0.15 <= margin <= 0.15
            assert pts[:, 0].min() >= -0.1 * L - 1e-9 and pts[:, 0].max() <= 1.1 * L + 1e-9
    assert modes == {"full", "partial"}


def test_court_coverage_overhead():
    w, h = 1280, 720
    sensor = equivalent_sensor_long_side(w, h)
    cam = (L / 2, W / 2, 30.0)
    # Looking straight down without roll, the court length runs along the
    # image's short (vertical) axis; fitted with no margin it spans the full
    # height and W / L * 9 / 16 of the width.
    R, f = frame_points(cam, CORNERS, 0.0, w, h, sensor, 0.0)
    cov, vis = court_coverage(cam, R, f, w, h, sensor, L, W)
    assert vis == pytest.approx(1.0)
    assert cov == pytest.approx((W / L) * (9 / 16), abs=0.03)
    # Zoomed 2x: half the length and all the width fit, i.e. half the court
    # is visible and it covers twice the image fraction.
    cov2, vis2 = court_coverage(cam, R, 2 * f, w, h, sensor, L, W)
    assert cov2 == pytest.approx(2 * cov, abs=0.03)
    assert vis2 == pytest.approx(0.5, abs=0.03)


def test_elevation():
    assert elevation_deg((L / 2, W / 2, 10.0), L, W) == pytest.approx(90.0)
    assert elevation_deg((L / 2 + 10.0, W / 2, 10.0), L, W) == pytest.approx(45.0)

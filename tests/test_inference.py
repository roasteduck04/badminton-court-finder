"""Tests for src.inference.predict and src.inference.visualize."""

import numpy as np
import pytest
import torch

from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, COURT_LENGTH, COURT_WIDTH_DOUBLES, get_court_lines
from src.inference.predict import (
    CourtDetection,
    CourtPredictor,
    estimate_homography_and_fill,
    extract_keypoints_from_heatmaps,
)
from src.inference.visualize import create_debug_visualization, draw_court_overlay, draw_keypoints
from src.models.courtvisionnet import CourtVisionNet


# ---------------------------------------------------------------------------
# CourtDetection dataclass
# ---------------------------------------------------------------------------

def test_court_detection_dataclass():
    det = CourtDetection(
        keypoints=np.zeros((14, 2)),
        visibility=np.zeros(14),
        confidence=0.9,
        homography=np.eye(3),
        seg_mask=np.zeros((640, 640)),
    )
    assert det.confidence == 0.9
    assert det.keypoints.shape == (14, 2)


def test_court_detection_optional_fields_default_to_none():
    det = CourtDetection(
        keypoints=np.zeros((14, 2)),
        visibility=np.zeros(14),
        confidence=0.5,
    )
    assert det.homography is None
    assert det.seg_mask is None
    assert det.projected_lines is None


def _sample_detection():
    return CourtDetection(
        keypoints=np.array([
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
            [0.3, 0.1], [0.3, 0.9], [0.7, 0.1], [0.7, 0.9],
            [0.5, 0.1], [0.5, 0.9], [0.3, 0.5], [0.7, 0.5],
            [0.5, 0.2], [0.5, 0.8],
        ]),
        visibility=np.ones(14),
        confidence=0.9,
        homography=np.eye(3),
        seg_mask=np.zeros((640, 640)),
    )


# ---------------------------------------------------------------------------
# extract_keypoints_from_heatmaps
# ---------------------------------------------------------------------------

def test_extract_keypoints_from_heatmaps_peak_plus_offset():
    heatmaps = np.zeros((2, 5, 5), dtype=np.float32)
    heatmaps[0, 1, 3] = 1.0  # row=1 (y), col=3 (x)
    heatmaps[1, 4, 0] = 1.0  # row=4 (y), col=0 (x)
    offsets = np.array([[0.1, -0.05], [0.0, 0.0]], dtype=np.float32)

    kpts = extract_keypoints_from_heatmaps(heatmaps, offsets)

    assert kpts.shape == (2, 2)
    assert kpts[0, 0] == pytest.approx(3 / 5 + 0.1)
    assert kpts[0, 1] == pytest.approx(1 / 5 - 0.05)
    assert kpts[1, 0] == pytest.approx(0.0)
    assert kpts[1, 1] == pytest.approx(4 / 5)


# ---------------------------------------------------------------------------
# estimate_homography_and_fill
# ---------------------------------------------------------------------------

def test_estimate_homography_insufficient_visible_points():
    keypoints = np.random.rand(14, 2)
    visibility = np.zeros(14)  # nothing visible

    homography, filled, lines = estimate_homography_and_fill(keypoints, visibility, 640, 480)

    assert homography is None
    assert lines is None
    np.testing.assert_array_equal(filled, keypoints)


def test_estimate_homography_fills_missing_keypoints():
    image_w, image_h = 200, 100
    sx = image_w / COURT_LENGTH
    sy = image_h / COURT_WIDTH_DOUBLES
    true_h = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)

    true_px = (true_h[:2, :2] @ COURT_KEYPOINTS_TEMPLATE.T).T
    keypoints_norm = true_px / np.array([image_w, image_h])

    visibility = np.zeros(14)
    visibility[:4] = 1.0  # only the 4 outer corners are confidently detected

    homography, filled, lines = estimate_homography_and_fill(
        keypoints_norm, visibility, image_w, image_h,
    )

    assert homography is not None
    assert homography.shape == (3, 3)
    assert lines is not None
    assert len(lines) == len(get_court_lines())
    # The detected (visible) corners must be untouched.
    np.testing.assert_allclose(filled[:4], keypoints_norm[:4])
    # The unseen keypoints should be recovered (near-exactly, since the
    # 4 corners exactly determine this affine-style homography).
    np.testing.assert_allclose(filled[4:], keypoints_norm[4:], atol=1e-6)


# ---------------------------------------------------------------------------
# draw_court_overlay / draw_keypoints / create_debug_visualization
# ---------------------------------------------------------------------------

def test_draw_court_overlay():
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    det = _sample_detection()
    result = draw_court_overlay(img, det)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_draw_court_overlay_does_not_mutate_input():
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    original = img.copy()
    draw_court_overlay(img, _sample_detection())
    np.testing.assert_array_equal(img, original)


def test_draw_court_overlay_without_homography_or_projected_lines():
    img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    det = CourtDetection(
        keypoints=np.full((14, 2), 0.5),
        visibility=np.zeros(14),
        confidence=0.0,
        homography=None,
        seg_mask=None,
        projected_lines=None,
    )
    result = draw_court_overlay(img, det)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_draw_keypoints_shape_and_dtype():
    img = np.zeros((100, 150, 3), dtype=np.uint8)
    det = _sample_detection()
    result = draw_keypoints(img, det)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_draw_keypoints_skips_low_visibility_points():
    img = np.zeros((100, 150, 3), dtype=np.uint8)
    det = _sample_detection()
    det.visibility = np.zeros(14)  # nothing meets even the extrapolated threshold
    result = draw_keypoints(img, det)
    # Nothing should have been drawn on top of the all-black image.
    np.testing.assert_array_equal(result, img)


def test_create_debug_visualization_shape():
    img = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    det = _sample_detection()
    result = create_debug_visualization(img, det)
    assert result.dtype == np.uint8
    assert result.shape == (120, 160 * 3, 3)


# ---------------------------------------------------------------------------
# CourtPredictor end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_checkpoint(tmp_path):
    """A small, fast-to-run CourtVisionNet checkpoint for CPU testing."""
    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32, pretrained=False)
    ckpt_path = tmp_path / "tiny_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, ckpt_path)
    return str(ckpt_path)


def test_court_predictor_end_to_end(tiny_checkpoint):
    predictor = CourtPredictor(tiny_checkpoint, device="cpu", image_size=128, heatmap_size=32)
    image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)  # non-square input

    detection = predictor.predict(image)

    assert isinstance(detection, CourtDetection)
    assert detection.keypoints.shape == (14, 2)
    assert detection.visibility.shape == (14,)
    assert 0.0 <= detection.confidence <= 1.0
    assert detection.seg_mask.shape == (128, 128)
    assert detection.seg_mask.dtype == np.uint8
    if detection.homography is not None:
        assert detection.homography.shape == (3, 3)


def test_court_predictor_loads_raw_state_dict(tmp_path):
    """Checkpoints saved as a bare state_dict (no wrapper dict) should also load."""
    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32, pretrained=False)
    ckpt_path = tmp_path / "raw_state_dict.pt"
    torch.save(model.state_dict(), ckpt_path)

    predictor = CourtPredictor(str(ckpt_path), device="cpu", image_size=128, heatmap_size=32)
    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    detection = predictor.predict(image)

    assert detection.keypoints.shape == (14, 2)


def test_court_predictor_output_is_visualizable(tiny_checkpoint):
    """The predictor's output should plug directly into draw_court_overlay."""
    predictor = CourtPredictor(tiny_checkpoint, device="cpu", image_size=128, heatmap_size=32)
    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    detection = predictor.predict(image)
    result = draw_court_overlay(image, detection)

    assert result.shape == image.shape
    assert result.dtype == np.uint8

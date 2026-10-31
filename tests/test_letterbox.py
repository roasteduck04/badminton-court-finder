import numpy as np
import pytest

from src.preprocessing.letterbox import (
    content_mask,
    keypoints_from_letterbox,
    keypoints_to_letterbox,
    letterbox,
)


@pytest.mark.parametrize("h,w", [(1080, 1920), (1920, 1080), (640, 640), (300, 500)])
def test_letterbox_shape_and_padding(h, w):
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    canvas, info = letterbox(img, 640)
    assert canvas.shape == (640, 640, 3)
    mask = content_mask(info)
    assert (canvas[mask == 1] == 200).all()
    assert (canvas[mask == 0] == 0).all()
    assert max(info.content_w, info.content_h) == 640


def test_letterbox_preserves_aspect_ratio():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _, info = letterbox(img, 640)
    assert info.content_w == 640
    assert info.content_h == 360
    assert info.pad_y == 140 and info.pad_x == 0


def test_keypoint_round_trip():
    img = np.zeros((1920, 1080, 3), dtype=np.uint8)
    _, info = letterbox(img, 640)
    kps = np.array([[0.0, 0.0], [1.0, 1.0], [0.25, 0.8], [-1.0, -1.0]])
    lb = keypoints_to_letterbox(kps, info)
    np.testing.assert_allclose(lb[3], [-1, -1])
    assert lb[0, 0] == pytest.approx(info.pad_x / 640)
    np.testing.assert_allclose(keypoints_from_letterbox(lb, info), kps, atol=1e-12)


def test_keypoint_lands_on_same_pixel():
    img = np.zeros((400, 1000, 3), dtype=np.uint8)
    img[100, 700] = 255  # marker at x=700, y=100
    canvas, info = letterbox(img, 640)
    kp = keypoints_to_letterbox(np.array([[700.5 / 1000, 100.5 / 400]]), info)[0]
    x, y = int(kp[0] * 640), int(kp[1] * 640)
    assert canvas[y - 1:y + 2, x - 1:x + 2].max() > 0


def test_single_channel_image():
    img = np.ones((100, 200), dtype=np.float32)
    canvas, info = letterbox(img, 64)
    assert canvas.shape == (64, 64)

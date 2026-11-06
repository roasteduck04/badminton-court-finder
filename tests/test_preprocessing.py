import cv2
import numpy as np

from src.preprocessing.channels import generate_channels
from src.preprocessing.letterbox import content_mask, letterbox


def test_generate_channels_shape():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = generate_channels(img)
    assert result.shape == (480, 640, 7)
    assert result.dtype == np.float32


def test_generate_channels_rgb_preserved():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = generate_channels(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    np.testing.assert_allclose(result[:, :, :3], rgb, atol=1e-5)


def test_generate_channels_grayscale():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = generate_channels(img)
    gray_expected = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    np.testing.assert_allclose(result[:, :, 3], gray_expected, atol=1e-5)


def test_generate_channels_range():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = generate_channels(img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def _letterboxed_bright_image():
    img = np.full((240, 320, 3), (60, 160, 60), dtype=np.uint8)  # green court colour
    cv2.line(img, (20, 120), (300, 120), (255, 255, 255), 3)
    return letterbox(img, 320)


def test_valid_mask_suppresses_padding_edges():
    canvas, info = _letterboxed_bright_image()
    valid = content_mask(info)
    plain = generate_channels(canvas)
    masked = generate_channels(canvas, valid)
    # Without the mask the content/padding boundary is a strong Canny edge.
    assert plain[info.pad_y - 2:info.pad_y + 2, :, 5].sum() > 0
    assert masked[info.pad_y - 3:info.pad_y + 3, :, 5].sum() == 0
    # Derived channels are zero in the padding; RGB/gray untouched.
    pad = valid == 0
    assert masked[pad][:, 4:].max() == 0
    np.testing.assert_array_equal(masked[..., :4], plain[..., :4])
    # The painted line inside the content still produces edges.
    assert masked[info.pad_y + 110:info.pad_y + 130, :, 5].sum() > 0
    # Court-colour channel kept inside the content.
    assert masked[~pad][:, 6].mean() > 0.9


def test_valid_mask_all_ones_is_identity():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        generate_channels(img), generate_channels(img, np.ones((64, 64), np.uint8)))

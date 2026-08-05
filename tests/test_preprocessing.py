import numpy as np
import cv2
import pytest
from src.preprocessing.channels import generate_channels
from src.preprocessing.augmentation import get_train_transforms, get_val_transforms


def test_generate_channels_shape():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = generate_channels(img)
    assert result.shape == (480, 640, 7)
    assert result.dtype == np.float32


def test_generate_channels_rgb_preserved():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = generate_channels(img)
    # First 3 channels should be RGB (converted from BGR), normalized to [0,1]
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


def test_train_transforms_output_shape():
    transform = get_train_transforms(image_size=640)
    img = np.random.randint(0, 255, (480, 640, 7), dtype=np.uint8)
    # albumentations expects keypoints in a specific format
    result = transform(image=img)
    assert result["image"].shape[:2] == (640, 640)


def test_val_transforms_deterministic():
    transform = get_val_transforms(image_size=640)
    img = np.random.randint(0, 255, (480, 640, 7), dtype=np.uint8)
    r1 = transform(image=img)["image"]
    r2 = transform(image=img)["image"]
    np.testing.assert_array_equal(r1, r2)

import json
import os
import numpy as np
import cv2
import torch
import pytest
from src.training.dataset import CourtDataset


@pytest.fixture
def mock_dataset(tmp_path):
    """Create a small mock dataset with 3 annotated frames."""
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    for i in range(3):
        # Create a fake image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = str(img_dir / f"frame_{i:03d}.jpg")
        cv2.imwrite(img_path, img)

        # Create annotation (30 keypoints)
        kps = [[round(j/29, 3), round(j/29, 3)] for j in range(30)]
        ann = {
            "image_path": img_path,
            "image_size": [640, 640],
            "court_class": 1,
            "keypoints": kps,
            "visibility": [1] * 30,
            "bounding_box": [0.5, 0.5, 0.8, 0.8],
        }
        with open(str(ann_dir / f"frame_{i:03d}.json"), "w") as f:
            json.dump(ann, f)

    return str(img_dir), str(ann_dir)


def test_dataset_length(mock_dataset):
    img_dir, ann_dir = mock_dataset
    ds = CourtDataset(ann_dir, img_dir, image_size=640)
    assert len(ds) == 3


def test_dataset_item_shapes(mock_dataset):
    img_dir, ann_dir = mock_dataset
    ds = CourtDataset(ann_dir, img_dir, image_size=640)
    sample = ds[0]

    assert sample["image"].shape == (7, 640, 640)
    assert sample["image"].dtype == torch.float32
    assert sample["heatmaps"].shape == (30, 160, 160)
    assert sample["keypoints"].shape == (30, 2)
    assert sample["visibility"].shape == (30,)
    assert sample["mask"].shape == (1, 640, 640)


def test_dataset_invisible_keypoints(tmp_path):
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "frame.jpg"), img)

    ann = {
        "image_path": str(img_dir / "frame.jpg"),
        "image_size": [640, 640],
        "court_class": 1,
        "keypoints": [[0.1, 0.1], [0.9, 0.1]] + [[-1, -1]] * 28,
        "visibility": [1, 1] + [0] * 28,
        "bounding_box": [0.5, 0.1, 0.8, 0.0],
    }
    with open(str(ann_dir / "frame.json"), "w") as f:
        json.dump(ann, f)

    ds = CourtDataset(str(ann_dir), str(img_dir), image_size=640)
    sample = ds[0]

    # Invisible keypoints should have zero heatmaps
    assert sample["visibility"][2] == 0
    assert sample["visibility"][0] == 1
    assert sample["heatmaps"][2].sum() == 0  # no heatmap for invisible
    assert sample["heatmaps"][0].sum() > 0   # heatmap present for visible


def test_dataset_with_train_transform(mock_dataset):
    """Dataset should work end-to-end with the real augmentation pipeline."""
    from src.preprocessing.augmentation import get_train_transforms

    img_dir, ann_dir = mock_dataset
    ds = CourtDataset(
        ann_dir, img_dir, transform=get_train_transforms(640), image_size=640
    )
    sample = ds[0]

    assert sample["image"].shape == (7, 640, 640)
    assert sample["heatmaps"].shape == (30, 160, 160)
    assert sample["keypoints"].shape == (30, 2)
    assert sample["mask"].shape == (1, 640, 640)


def test_dataset_with_val_transform(mock_dataset):
    """Val transforms are deterministic resize-only; keypoints stay unchanged."""
    from src.preprocessing.augmentation import get_val_transforms

    img_dir, ann_dir = mock_dataset
    ds = CourtDataset(
        ann_dir, img_dir, transform=get_val_transforms(640), image_size=640
    )
    sample = ds[0]

    assert sample["image"].shape == (7, 640, 640)
    assert torch.all(sample["visibility"] == 1)

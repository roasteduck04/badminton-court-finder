"""Tests for src.training.config and src.training.train."""

import json
import os

import cv2
import numpy as np
import pytest
import torch

from src.training.config import TrainConfig


@pytest.fixture
def mock_data(tmp_path):
    """Create minimal mock dataset for training tests."""
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    for i in range(4):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"f{i}.jpg"), img)
        ann = {
            "image_path": str(img_dir / f"f{i}.jpg"),
            "image_size": [128, 128],
            "court_class": 1,
            "keypoints": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]] + [[-1, -1]] * 27,
            "visibility": [1, 1, 1, 1] + [0] * 27,
            "bounding_box": [0.5, 0.5, 0.8, 0.8],
        }
        with open(str(ann_dir / f"f{i}.json"), "w") as f:
            json.dump(ann, f)
    return str(img_dir), str(ann_dir)


class TestTrainConfig:
    def test_defaults(self):
        config = TrainConfig()
        assert config.batch_size == 8
        assert config.num_epochs == 100
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 1e-4
        assert config.patience == 10
        assert config.freeze_backbone_epochs == 5
        assert config.in_channels == 7
        assert config.num_keypoints == 31
        assert config.image_size == 640

    def test_overridable(self):
        config = TrainConfig(batch_size=4, num_epochs=2, learning_rate=1e-3)
        assert config.batch_size == 4
        assert config.num_epochs == 2
        assert config.learning_rate == 1e-3

    def test_is_dataclass_instance(self):
        from dataclasses import is_dataclass

        assert is_dataclass(TrainConfig())


def test_train_one_epoch(mock_data):
    from torch.utils.data import DataLoader

    from src.models.courtvisionnet import CourtVisionNet
    from src.models.losses import CourtVisionLoss
    from src.training.dataset import CourtDataset
    from src.training.train import train_one_epoch

    img_dir, ann_dir = mock_data
    ds = CourtDataset(ann_dir, img_dir, image_size=128)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32, pretrained=False)
    loss_fn = CourtVisionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    avg_loss = train_one_epoch(model, loader, loss_fn, optimizer, device="cpu")
    assert isinstance(avg_loss, float)
    assert avg_loss > 0


def test_validate(mock_data):
    from torch.utils.data import DataLoader

    from src.models.courtvisionnet import CourtVisionNet
    from src.models.losses import CourtVisionLoss
    from src.training.dataset import CourtDataset
    from src.training.train import validate

    img_dir, ann_dir = mock_data
    ds = CourtDataset(ann_dir, img_dir, image_size=128)
    loader = DataLoader(ds, batch_size=2)

    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32, pretrained=False)
    loss_fn = CourtVisionLoss()

    val_loss, metrics = validate(model, loader, loss_fn, device="cpu")
    assert isinstance(val_loss, float)
    assert "seg_loss" in metrics


def test_validate_does_not_update_weights(mock_data):
    """validate() must not mutate model parameters (no gradient step)."""
    from torch.utils.data import DataLoader

    from src.models.courtvisionnet import CourtVisionNet
    from src.models.losses import CourtVisionLoss
    from src.training.dataset import CourtDataset
    from src.training.train import validate

    img_dir, ann_dir = mock_data
    ds = CourtDataset(ann_dir, img_dir, image_size=128)
    loader = DataLoader(ds, batch_size=2)

    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32, pretrained=False)
    loss_fn = CourtVisionLoss()

    params_before = [p.clone() for p in model.parameters()]
    validate(model, loader, loss_fn, device="cpu")
    for before, after in zip(params_before, model.parameters()):
        assert torch.equal(before, after)


def test_train_one_epoch_updates_weights(mock_data):
    """train_one_epoch() should actually change model parameters."""
    from torch.utils.data import DataLoader

    from src.models.courtvisionnet import CourtVisionNet
    from src.models.losses import CourtVisionLoss
    from src.training.dataset import CourtDataset
    from src.training.train import train_one_epoch

    img_dir, ann_dir = mock_data
    ds = CourtDataset(ann_dir, img_dir, image_size=128)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32, pretrained=False)
    loss_fn = CourtVisionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    params_before = [p.clone() for p in model.parameters()]
    train_one_epoch(model, loader, loss_fn, optimizer, device="cpu")
    changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, model.parameters())
    )
    assert changed


def test_full_train_loop_short_run(mock_data, tmp_path):
    """End-to-end smoke test: train() runs for a couple epochs and returns metrics."""
    from src.training.train import train

    img_dir, ann_dir = mock_data
    checkpoint_dir = tmp_path / "checkpoints"

    config = TrainConfig(
        train_annotations=ann_dir,
        val_annotations=ann_dir,
        train_images=img_dir,
        val_images=img_dir,
        image_size=128,
        heatmap_size=32,
        pretrained=False,
        batch_size=2,
        num_epochs=2,
        freeze_backbone_epochs=1,
        patience=10,
        checkpoint_dir=str(checkpoint_dir),
    )

    result = train(config)
    assert isinstance(result, dict)
    assert "best_val_loss" in result
    assert "final_epoch" in result
    assert os.path.exists(os.path.join(str(checkpoint_dir), "best_model.pt"))


def test_early_stopping_triggers(mock_data, tmp_path):
    """With patience=0, training should stop after the first epoch that fails to improve."""
    from src.training.train import train

    img_dir, ann_dir = mock_data
    checkpoint_dir = tmp_path / "checkpoints"

    config = TrainConfig(
        train_annotations=ann_dir,
        val_annotations=ann_dir,
        train_images=img_dir,
        val_images=img_dir,
        image_size=128,
        heatmap_size=32,
        pretrained=False,
        batch_size=2,
        num_epochs=50,
        freeze_backbone_epochs=0,
        patience=0,
        checkpoint_dir=str(checkpoint_dir),
    )

    result = train(config)
    # Should stop well before the configured num_epochs due to patience=0.
    assert result["final_epoch"] < config.num_epochs

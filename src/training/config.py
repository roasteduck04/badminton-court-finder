"""Training configuration for CourtVisionNet."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """All hyperparameters for a CourtVisionNet training run.

    Data paths default to the project's standard layout but can be
    overridden per-run (e.g. in the Colab notebook or tests).
    """

    # Data
    train_annotations: str = "data/annotations/train"
    val_annotations: str = "data/annotations/val"
    train_images: str = "data/frames"
    val_images: str = "data/frames"

    # Model
    in_channels: int = 7
    num_keypoints: int = 31
    image_size: int = 640
    heatmap_size: int = 160
    pretrained: bool = True

    # Training
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 10
    freeze_backbone_epochs: int = 5

    # Loss weights
    seg_weight: float = 1.0
    heatmap_weight: float = 5.0
    offset_weight: float = 1.0
    vis_weight: float = 1.0

    # Output
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10

    # DataLoader
    # Default 0 for safety on Windows (multiprocessing workers there require
    # the launching script to guard with `if __name__ == "__main__":`).
    # Bump this to 2-4 on Colab/Linux for faster data loading.
    num_workers: int = 0

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transforms(image_size=640):
    """Training augmentation pipeline for 7-channel input with keypoints.

    Works with albumentations' keypoint format.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.5),
        A.Perspective(scale=(0.05, 0.15), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.05, 0.2),
            hole_width_range=(0.05, 0.2),
            fill=0, p=0.3
        ),
    ], keypoint_params=A.KeypointParams(
        format="xy", remove_invisible=False, angle_in_degrees=True
    ))


def get_val_transforms(image_size=640):
    """Validation/test transforms — deterministic resize only."""
    return A.Compose([
        A.Resize(image_size, image_size),
    ], keypoint_params=A.KeypointParams(
        format="xy", remove_invisible=False, angle_in_degrees=True
    ))

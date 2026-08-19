import albumentations as A
import numpy as np


def get_train_transforms(image_size=640):
    """Training augmentation pipeline for 7-channel input with keypoints.

    Works with albumentations' keypoint format.

    Uses ``A.ReplayCompose`` (rather than ``A.Compose``) so that callers can
    inspect ``result["replay"]`` to determine which transforms actually
    fired for a given sample -- in particular, whether ``HorizontalFlip``
    was applied. This is required by
    ``src.training.dataset.CourtDataset._apply_transform`` to correctly
    swap left/right keypoint pairs (see ``src.court_geometry.FLIP_PAIRS``)
    after a horizontal flip, since albumentations mirrors keypoint
    coordinates but does not re-map their semantic left/right identity.
    """
    return A.ReplayCompose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.85, 1.15), translate_percent=(-0.08, 0.08),
            rotate=(-25, 25), shear=(-8, 8),
            border_mode=0, p=0.6,
        ),
        A.Perspective(scale=(0.05, 0.20), p=0.4),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.5
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.MotionBlur(blur_limit=(3, 7), p=0.15),
        A.GaussNoise(std_range=(0.012, 0.04), p=0.25),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(0.03, 0.15),
            hole_width_range=(0.03, 0.15),
            fill=0, p=0.4
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

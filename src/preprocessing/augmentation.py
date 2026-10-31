"""Albumentations pipelines for CourtVisionNet training.

The transforms run on the letterboxed BGR uint8 canvas (see
``src.preprocessing.letterbox``) BEFORE the derived channels are computed, so
photometric ops (HSV, gamma, JPEG, noise) see a real colour image and the
CLAHE/Canny/court-colour channels are recomputed from the augmented pixels.

Keypoints are passed in albumentations' pixel-INDEX coordinates, i.e.
``kp_norm * size - 0.5`` (pixel i has its centre at index i). Albumentations
2.x maps keypoints in that convention (e.g. hflip is ``x -> W - 1 - x``), so
converting back with ``(x + 0.5) / size`` keeps labels aligned with pixels.

A ``mask`` target carries the content-validity mask (1 = real image pixel,
0 = letterbox / affine padding) through the geometric transforms.

``A.ReplayCompose`` is used so the dataset can tell whether a horizontal flip
fired (needed for the fallback relabeling when ``canonicalize`` cannot
determine the court orientation).
"""

import random

import albumentations as A
import cv2
import numpy as np
import torch

_KEYPOINT_PARAMS = dict(format="xy", remove_invisible=False)


def _geometric_transforms():
    return [
        A.HorizontalFlip(p=0.5),
        # Scale 0.6-1.3 covers zoomed-out views and partial-court crops.
        A.Affine(
            scale=(0.6, 1.3), translate_percent=(-0.1, 0.1),
            rotate=(-25, 25), shear=(-8, 8),
            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7,
        ),
        A.Perspective(
            scale=(0.02, 0.1), border_mode=cv2.BORDER_CONSTANT,
            fill=0, fill_mask=0, p=0.4,
        ),
        # Independent per-side crop/pad then resize back: aspect jitter and
        # partial-court crops.
        A.CropAndPad(
            percent=(-0.15, 0.1), keep_size=True, sample_independently=True,
            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.3,
        ),
    ]


def _photometric_transforms():
    return [
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.5
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 9)),
        ], p=0.25),
        A.ImageCompression(quality_range=(30, 95), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.25),
        # Occluders (players, net posts, rackets). The validity mask is left
        # untouched: occluded pixels are still real image content.
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(0.03, 0.15),
            hole_width_range=(0.03, 0.15),
            fill="random_uniform", p=0.4,
        ),
    ]


def get_train_transforms(image_size=640, seed=None, geometric=True, photometric=True):
    """Training augmentation for the letterboxed BGR uint8 canvas.

    Call as ``t(image=canvas, mask=valid, keypoints=kps_index_xy)``. Output
    image size equals input size (``image_size`` is kept for API
    compatibility; the canvas is already square).

    Args:
        seed: seed for albumentations' internal RNG (it does NOT use the
            global numpy/random state). DataLoader workers should reseed via
            ``seed_worker``.
        geometric / photometric: enable each family (tests use one at a time).
    """
    transforms = []
    if geometric:
        transforms += _geometric_transforms()
    if photometric:
        transforms += _photometric_transforms()
    compose = A.ReplayCompose(
        transforms, keypoint_params=A.KeypointParams(**_KEYPOINT_PARAMS)
    )
    compose.set_random_seed(seed)
    return compose


def get_val_transforms(image_size=640):
    """Validation/test: no-op (letterboxing is done by the dataset)."""
    return A.ReplayCompose(
        [A.NoOp()], keypoint_params=A.KeypointParams(**_KEYPOINT_PARAMS)
    )


def reseed_transform(transform, seed):
    """Reseed an albumentations Compose (and all children) if it supports it."""
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(int(seed) % (2 ** 32))


def seed_worker(worker_id):
    """DataLoader ``worker_init_fn``: seed every RNG a worker uses.

    torch already gives each worker ``base_seed + worker_id`` (and base_seed
    comes from the DataLoader's generator, so it is reproducible when that
    generator is seeded). Albumentations keeps its own RNGs inside every
    transform, which are copied verbatim into each worker, so they must be
    reseeded here or all workers would produce identical augmentations.
    """
    info = torch.utils.data.get_worker_info()
    worker_seed = (info.seed if info is not None else torch.initial_seed()) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if info is not None:
        dataset = info.dataset
        # Unwrap torch.utils.data.Subset / ConcatDataset-style wrappers.
        while hasattr(dataset, "dataset") and not hasattr(dataset, "transform"):
            dataset = dataset.dataset
        reseed_transform(getattr(dataset, "transform", None), worker_seed)

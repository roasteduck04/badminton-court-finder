"""PyTorch Dataset for CourtVisionNet badminton court detection.

Loads annotation JSONs produced by `src.tools.annotator`, applies the
7-channel preprocessing pipeline, optional albumentations augmentation,
and generates Gaussian keypoint heatmaps plus a court-line segmentation
mask for training.
"""

import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.court_geometry import FLIP_PAIRS, generate_line_mask
from src.preprocessing.channels import generate_channels

from src.court_geometry import NUM_KEYPOINTS


class CourtDataset(Dataset):
    """PyTorch dataset for badminton court detection.

    Each sample returns a dict with:
        image: (7, H, W) float32 tensor — 7-channel preprocessed input
        heatmaps: (14, H/4, W/4) float32 tensor — Gaussian heatmap per keypoint
        keypoints: (14, 2) float32 tensor — normalized [0,1] keypoint coordinates
            (or [-1,-1] where not visible)
        visibility: (14,) float32 tensor — 1.0 if visible, 0.0 if not
        mask: (1, H, W) float32 tensor — binary court line segmentation mask
    """

    def __init__(self, annotations_dir, images_dir, transform=None,
                 image_size=640, heatmap_size=None, sigma=2.0):
        self.images_dir = images_dir
        self.image_size = image_size
        self.heatmap_size = heatmap_size if heatmap_size is not None else image_size // 4
        self.sigma = sigma
        self.transform = transform

        self.annotation_paths = sorted(
            glob.glob(os.path.join(annotations_dir, "*.json"))
        )

    def __len__(self):
        return len(self.annotation_paths)

    def __getitem__(self, idx):
        with open(self.annotation_paths[idx]) as f:
            ann = json.load(f)

        image = self._load_image(ann["image_path"])
        image = cv2.resize(image, (self.image_size, self.image_size))

        keypoints = np.array(ann["keypoints"], dtype=np.float32).copy()
        visibility = np.array(ann["visibility"], dtype=np.float32).copy()

        # Generate 7-channel input from the resized BGR image.
        channels = generate_channels(image)  # (H, W, 7) float32 in [0, 1]

        if self.transform is not None:
            channels, keypoints, visibility = self._apply_transform(
                channels, keypoints, visibility
            )

        heatmaps = self._generate_heatmaps(keypoints, visibility)

        mask = generate_line_mask(
            keypoints,
            visibility.astype(int).tolist(),
            width=self.image_size,
            height=self.image_size,
            line_thickness=3,
        )
        mask = mask.astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(channels.transpose(2, 0, 1))
        )  # (7, H, W)
        heatmap_tensor = torch.from_numpy(heatmaps)  # (14, hm_h, hm_w)
        keypoint_tensor = torch.from_numpy(keypoints)  # (14, 2)
        visibility_tensor = torch.from_numpy(visibility)  # (14,)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return {
            "image": image_tensor,
            "heatmaps": heatmap_tensor,
            "keypoints": keypoint_tensor,
            "visibility": visibility_tensor,
            "mask": mask_tensor,
        }

    def _load_image(self, image_path):
        path = image_path
        if not os.path.isabs(path):
            path = os.path.join(self.images_dir, os.path.basename(path))
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return image

    def _apply_transform(self, channels, keypoints, visibility):
        """Apply an albumentations pipeline to the 7-channel image and keypoints.

        Channels are scaled to uint8 for the augmentation call (albumentations
        pixel-level ops expect uint8/float32 images; uint8 keeps behavior
        consistent with the RGB-oriented transforms) and scaled back to
        [0, 1] afterward. Keypoints that fall outside the frame after a
        geometric transform (e.g. rotation, perspective) are marked
        invisible.

        If a horizontal flip fires, albumentations mirrors each keypoint's
        x-coordinate but leaves it at its original array index. That would
        silently corrupt keypoint semantic identity (e.g. a flipped
        top-left corner would still be reported as index K0 "top-left"
        while actually located at the top-right of the frame). When
        ``self.transform`` is an ``A.ReplayCompose``, we inspect the replay
        log to detect whether HorizontalFlip applied, and if so swap the
        paired left/right keypoint indices defined in
        ``src.court_geometry.FLIP_PAIRS`` to restore correct identity.
        """
        pixel_kps = [
            (float(keypoints[i, 0] * self.image_size),
             float(keypoints[i, 1] * self.image_size))
            for i in range(NUM_KEYPOINTS)
        ]

        transformed = self.transform(
            image=(channels * 255.0).astype(np.uint8),
            keypoints=pixel_kps,
        )
        new_channels = transformed["image"].astype(np.float32) / 255.0
        new_kps = transformed["keypoints"]

        new_keypoints = keypoints.copy()
        new_visibility = visibility.copy()
        for i in range(NUM_KEYPOINTS):
            if new_visibility[i] < 1:
                continue
            kx, ky = new_kps[i][0], new_kps[i][1]
            nx = kx / self.image_size
            ny = ky / self.image_size
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                new_keypoints[i, 0] = nx
                new_keypoints[i, 1] = ny
            else:
                new_visibility[i] = 0.0
                new_keypoints[i] = [-1.0, -1.0]

        if self._replay_has_horizontal_flip(transformed.get("replay")):
            new_keypoints, new_visibility = self._swap_flip_pairs(
                new_keypoints, new_visibility
            )

        return new_channels, new_keypoints, new_visibility

    @staticmethod
    def _replay_has_horizontal_flip(replay_node):
        """Recursively search an A.ReplayCompose replay log for an applied
        HorizontalFlip transform.

        Returns False if ``replay_node`` is None (e.g. a plain A.Compose
        pipeline was used instead of A.ReplayCompose, or no transform ran).
        """
        if not replay_node:
            return False
        name = replay_node.get("__class_fullname__", "")
        if name.endswith("HorizontalFlip") and replay_node.get("applied"):
            return True
        for child in replay_node.get("transforms", None) or []:
            if CourtDataset._replay_has_horizontal_flip(child):
                return True
        return False

    @staticmethod
    def _swap_flip_pairs(keypoints, visibility):
        """Swap left/right keypoint pairs (see court_geometry.FLIP_PAIRS)
        to restore correct semantic identity after a horizontal flip.
        """
        keypoints = keypoints.copy()
        visibility = visibility.copy()
        for i, j in FLIP_PAIRS:
            keypoints[[i, j]] = keypoints[[j, i]]
            visibility[[i, j]] = visibility[[j, i]]
        return keypoints, visibility

    def _generate_heatmaps(self, keypoints, visibility):
        """Generate a Gaussian heatmap for each visible keypoint.

        Invisible keypoints get an all-zero heatmap.
        """
        hm_size = self.heatmap_size
        heatmaps = np.zeros((NUM_KEYPOINTS, hm_size, hm_size), dtype=np.float32)

        xx, yy = np.meshgrid(
            np.arange(hm_size, dtype=np.float32),
            np.arange(hm_size, dtype=np.float32),
        )

        for i in range(NUM_KEYPOINTS):
            if visibility[i] < 1:
                continue
            cx = keypoints[i, 0] * hm_size
            cy = keypoints[i, 1] * hm_size
            heatmaps[i] = np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.sigma ** 2)
            )

        return heatmaps

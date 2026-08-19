"""Inference pipeline for CourtVisionNet.

Loads a trained checkpoint, runs the model on a BGR image, and turns the
raw heatmap/offset/visibility/segmentation outputs into a `CourtDetection`:
court keypoints in normalized [0, 1] image coordinates, a per-keypoint
visibility score, an estimated homography (court template -> image), a
binary segmentation mask, and the projected court line segments.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

from src.court_geometry import (
    COURT_KEYPOINTS_TEMPLATE, FLIP_PAIRS, NUM_KEYPOINTS,
    compute_homography, get_court_lines, project_points,
)
from src.models.courtvisionnet import CourtVisionNet
from src.preprocessing.channels import generate_channels
VISIBILITY_THRESHOLD = 0.5
MIN_POINTS_FOR_HOMOGRAPHY = 4


@dataclass
class CourtDetection:
    """Result of running `CourtPredictor.predict()` on a single image."""

    keypoints: np.ndarray                  # (30, 2) normalized [0, 1] (x, y) image coords
    visibility: np.ndarray                 # (30,) sigmoid visibility probability per keypoint
    confidence: float                      # overall detection confidence in [0, 1]
    homography: Optional[np.ndarray] = None    # (3, 3) template(meters) -> image pixels, or None
    seg_mask: Optional[np.ndarray] = None      # (H, W) uint8 binary court-line mask
    projected_lines: Optional[list] = None     # list of ((x1, y1), (x2, y2)) pixel segments


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def extract_keypoints_from_heatmaps(heatmaps, offsets):
    """Extract (x, y) keypoint locations from soft-argmax coordinates.

    With the soft-argmax keypoint head, `offsets` already contains the full
    normalized [0, 1] (x, y) coordinates (not sub-pixel corrections), so
    they are returned directly.

    Args:
        heatmaps: (K, hm_h, hm_w) array of per-keypoint heatmaps (unused,
            kept for API compatibility)
        offsets: (K, 2) array of normalized [0, 1] (x, y) coordinates from
            soft-argmax

    Returns:
        (K, 2) float64 array of normalized [0, 1] (x, y) coordinates.
    """
    return offsets.astype(np.float64)


def estimate_homography_and_fill(keypoints, visibility, image_w, image_h,
                                  visibility_threshold=VISIBILITY_THRESHOLD):
    """Estimate a template->image homography from visible keypoints.

    Uses the visible detected keypoints as correspondences against the
    known court template (in meters) to solve for a homography, then
    projects the full template through it to fill in keypoints that were
    not confidently detected. Detected (visible) keypoints are left as-is.

    Args:
        keypoints: (30, 2) normalized [0, 1] (x, y) coordinates
        visibility: (30,) visibility probabilities in [0, 1]
        image_w, image_h: original image dimensions in pixels
        visibility_threshold: minimum probability to trust a keypoint

    Returns:
        (homography, filled_keypoints, projected_lines) where homography is
        a (3, 3) array or None if fewer than 4 points are visible or the
        solve fails, filled_keypoints is a (30, 2) normalized array, and
        projected_lines is a list of ((x1, y1), (x2, y2)) pixel-space line
        segments (or None if no homography was found).
    """
    filled = keypoints.copy()
    visible_mask = visibility > visibility_threshold

    if visible_mask.sum() < MIN_POINTS_FOR_HOMOGRAPHY:
        return None, filled, None

    src_pts = COURT_KEYPOINTS_TEMPLATE[visible_mask]
    dst_pts_px = keypoints[visible_mask] * np.array([image_w, image_h])

    try:
        homography = compute_homography(src_pts, dst_pts_px)
    except ValueError:
        return None, filled, None

    all_projected_px = project_points(homography, COURT_KEYPOINTS_TEMPLATE)
    all_projected_norm = all_projected_px / np.array([image_w, image_h])

    for i in range(NUM_KEYPOINTS):
        if not visible_mask[i]:
            filled[i] = all_projected_norm[i]

    projected_lines = [
        (
            (float(all_projected_px[start, 0]), float(all_projected_px[start, 1])),
            (float(all_projected_px[end, 0]), float(all_projected_px[end, 1])),
        )
        for start, end in get_court_lines()
    ]

    return homography, filled, projected_lines


class CourtPredictor:
    """Loads a CourtVisionNet checkpoint and runs end-to-end court detection."""

    def __init__(self, checkpoint_path, device="cuda", image_size=640, heatmap_size=160):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.image_size = image_size
        self.heatmap_size = heatmap_size

        self.model = CourtVisionNet(
            in_channels=7,
            image_size=image_size,
            heatmap_size=heatmap_size,
            pretrained=False,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        """Run inference on a single BGR image.

        Args:
            image: (H, W, 3) BGR uint8 image, any resolution

        Returns:
            CourtDetection with keypoints/visibility/confidence/homography/
            seg_mask/projected_lines populated from the model's outputs.
        """
        h_orig, w_orig = image.shape[:2]
        resized = cv2.resize(image, (self.image_size, self.image_size))
        channels = generate_channels(resized)
        tensor = torch.from_numpy(channels.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        out = self.model(tensor)

        heatmaps = out["heatmaps"][0].cpu().numpy()        # (30, hm_h, hm_w)
        offsets = out["offsets"][0].cpu().numpy()          # (30, 2)
        vis_logits = out["visibility"][0].cpu().numpy()    # (30,)
        seg_logits = out["seg_logits"][0, 0].cpu().numpy()  # (image_size, image_size)

        vis_probs = _sigmoid(vis_logits)
        seg_mask = (_sigmoid(seg_logits) > 0.5).astype(np.uint8)

        keypoints = extract_keypoints_from_heatmaps(heatmaps, offsets)
        homography, keypoints, projected_lines = estimate_homography_and_fill(
            keypoints, vis_probs, w_orig, h_orig,
        )

        visible_mask = vis_probs > VISIBILITY_THRESHOLD
        confidence = float(vis_probs[visible_mask].mean()) if visible_mask.any() else 0.0

        return CourtDetection(
            keypoints=keypoints,
            visibility=vis_probs,
            confidence=confidence,
            homography=homography,
            seg_mask=seg_mask,
            projected_lines=projected_lines,
        )

    @torch.no_grad()
    def _predict_raw(self, image, scale=1.0, flip=False):
        """Run model and return raw keypoints + visibility (no homography fill).

        Args:
            image: (H, W, 3) BGR uint8 image
            scale: resize factor applied before inference
            flip: if True, horizontally flip the image before inference
        """
        img = image
        if flip:
            img = cv2.flip(img, 1)

        size = int(self.image_size * scale)
        resized = cv2.resize(img, (size, size))
        channels = generate_channels(resized)
        tensor = torch.from_numpy(channels.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        # Pad/interpolate features to match expected heatmap size
        if scale != 1.0:
            tensor = torch.nn.functional.interpolate(
                tensor, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )

        out = self.model(tensor)

        offsets = out["offsets"][0].cpu().numpy()       # (30, 2)
        vis_logits = out["visibility"][0].cpu().numpy()  # (30,)
        vis_probs = _sigmoid(vis_logits)
        keypoints = offsets.astype(np.float64)

        if flip:
            keypoints[:, 0] = 1.0 - keypoints[:, 0]
            # Swap left/right keypoint identities
            keypoints_swapped = keypoints.copy()
            vis_swapped = vis_probs.copy()
            for i, j in FLIP_PAIRS:
                keypoints_swapped[i] = keypoints[j]
                keypoints_swapped[j] = keypoints[i]
                vis_swapped[i] = vis_probs[j]
                vis_swapped[j] = vis_probs[i]
            keypoints = keypoints_swapped
            vis_probs = vis_swapped

        return keypoints, vis_probs

    @torch.no_grad()
    def predict_tta(self, image, scales=(0.85, 1.0, 1.15), use_flip=True):
        """Run test-time augmentation: multi-scale + optional flip.

        Averages keypoint predictions and visibility across all augmented
        views for more robust detection.

        Args:
            image: (H, W, 3) BGR uint8 image
            scales: tuple of scale factors to run inference at
            use_flip: whether to also run horizontally-flipped inference
        """
        h_orig, w_orig = image.shape[:2]
        all_kps = []
        all_vis = []

        for scale in scales:
            kps, vis = self._predict_raw(image, scale=scale, flip=False)
            all_kps.append(kps)
            all_vis.append(vis)

            if use_flip:
                kps_f, vis_f = self._predict_raw(image, scale=scale, flip=True)
                all_kps.append(kps_f)
                all_vis.append(vis_f)

        keypoints = np.mean(all_kps, axis=0)
        vis_probs = np.mean(all_vis, axis=0)

        seg_out = self._predict_raw_seg(image)
        seg_mask = seg_out

        homography, keypoints, projected_lines = estimate_homography_and_fill(
            keypoints, vis_probs, w_orig, h_orig,
        )

        visible_mask = vis_probs > VISIBILITY_THRESHOLD
        confidence = float(vis_probs[visible_mask].mean()) if visible_mask.any() else 0.0

        return CourtDetection(
            keypoints=keypoints,
            visibility=vis_probs,
            confidence=confidence,
            homography=homography,
            seg_mask=seg_mask,
            projected_lines=projected_lines,
        )

    @torch.no_grad()
    def _predict_raw_seg(self, image):
        """Get segmentation mask from the original-scale image."""
        resized = cv2.resize(image, (self.image_size, self.image_size))
        channels = generate_channels(resized)
        tensor = torch.from_numpy(channels.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        out = self.model(tensor)
        seg_logits = out["seg_logits"][0, 0].cpu().numpy()
        return (_sigmoid(seg_logits) > 0.5).astype(np.uint8)

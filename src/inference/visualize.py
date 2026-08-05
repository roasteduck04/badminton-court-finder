"""Visualization utilities for CourtVisionNet inference results."""

import cv2
import numpy as np

from src.court_geometry import get_court_lines
from src.inference.predict import VISIBILITY_THRESHOLD as VISIBLE_THRESHOLD

KEYPOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),      # K0-K3 corners
    (255, 128, 0), (128, 255, 0), (0, 128, 255), (128, 0, 255),  # K4-K7
    (255, 0, 128), (0, 255, 128),                              # K8-K9
    (200, 200, 0), (0, 200, 200),                              # K10-K11
    (200, 0, 200), (100, 100, 100),                            # K12-K13
]

KEYPOINT_NAMES = [
    "K0:TL", "K1:TR", "K2:BR", "K3:BL",
    "K4", "K5", "K6", "K7",
    "K8:NetT", "K9:NetB",
    "K10", "K11", "K12", "K13",
]

EXTRAPOLATED_THRESHOLD = 0.3


def _keypoints_to_pixels(detection, width, height):
    """Convert a detection's normalized [0, 1] keypoints to pixel coords."""
    pts = np.asarray(detection.keypoints, dtype=np.float64)
    px = pts.copy()
    px[:, 0] *= width
    px[:, 1] *= height
    return px


def draw_keypoints(image, detection, radius=6):
    """Draw detected keypoints with labels on a copy of `image`.

    Solid, filled circles mark confidently-detected keypoints (visibility
    above `VISIBLE_THRESHOLD`); hollow circles mark keypoints that were only
    extrapolated from the homography (visibility above `EXTRAPOLATED_THRESHOLD`
    but below `VISIBLE_THRESHOLD`). Keypoints below both thresholds are not drawn.

    Args:
        image: (H, W, 3) BGR uint8 image
        detection: CourtDetection
        radius: circle radius in pixels

    Returns:
        (H, W, 3) BGR uint8 image with keypoints drawn
    """
    result = image.copy()
    h, w = image.shape[:2]
    pixel_kpts = _keypoints_to_pixels(detection, w, h)

    for i in range(len(pixel_kpts)):
        vis = detection.visibility[i]
        if vis <= EXTRAPOLATED_THRESHOLD:
            continue

        x, y = int(round(pixel_kpts[i, 0])), int(round(pixel_kpts[i, 1]))
        color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]

        if vis > VISIBLE_THRESHOLD:
            cv2.circle(result, (x, y), radius, color, -1)
            cv2.circle(result, (x, y), radius, (255, 255, 255), 1)
        else:
            cv2.circle(result, (x, y), radius, color, 1)  # hollow = extrapolated

        name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"K{i}"
        cv2.putText(result, name, (x + 8, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return result


def draw_court_overlay(image, detection, alpha=0.3):
    """Draw the full detected court (segmentation, lines, keypoints) on `image`.

    Args:
        image: (H, W, 3) BGR uint8 image
        detection: CourtDetection
        alpha: transparency for the segmentation mask overlay

    Returns:
        (H, W, 3) BGR uint8 image with the overlay drawn
    """
    result = image.copy()
    h, w = image.shape[:2]

    # Segmentation mask overlay.
    if detection.seg_mask is not None and np.asarray(detection.seg_mask).any():
        mask_resized = cv2.resize(
            detection.seg_mask.astype(np.uint8),
            (w, h), interpolation=cv2.INTER_NEAREST,
        )
        overlay = result.copy()
        overlay[mask_resized > 0] = (0, 255, 255)  # yellow
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

    # Court lines between keypoints that are at least extrapolated.
    pixel_kpts = _keypoints_to_pixels(detection, w, h)
    for start_idx, end_idx in get_court_lines():
        if start_idx == end_idx:
            continue
        if detection.visibility[start_idx] > EXTRAPOLATED_THRESHOLD and \
                detection.visibility[end_idx] > EXTRAPOLATED_THRESHOLD:
            pt1 = tuple(pixel_kpts[start_idx].astype(int))
            pt2 = tuple(pixel_kpts[end_idx].astype(int))
            cv2.line(result, pt1, pt2, (0, 255, 0), 2)

    # Homography-projected lines, drawn in a distinct color when available.
    if detection.projected_lines:
        for (x1, y1), (x2, y2) in detection.projected_lines:
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)

    result = draw_keypoints(result, detection)

    cv2.putText(result, f"Conf: {detection.confidence:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return result


def _seg_mask_panel(image, detection):
    """Render the raw segmentation mask, resized to `image`'s size, as BGR."""
    h, w = image.shape[:2]
    if detection.seg_mask is not None:
        mask = (np.asarray(detection.seg_mask).astype(np.uint8) * 255)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        panel = np.zeros((h, w, 3), dtype=np.uint8)
    return panel


def _label_panel(panel, text):
    """Stamp a label bar across the top of a panel (in place on a copy)."""
    labeled = panel.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(labeled, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return labeled


def create_debug_visualization(image, detection):
    """Build a side-by-side debug view: overlay | keypoints | segmentation.

    Args:
        image: (H, W, 3) BGR uint8 image
        detection: CourtDetection

    Returns:
        (H, W*3, 3) BGR uint8 image combining all three panels
    """
    overlay = draw_court_overlay(image, detection)
    keypoints_panel = draw_keypoints(image, detection)
    seg_panel = _seg_mask_panel(image, detection)

    panels = [
        _label_panel(overlay, "Overlay"),
        _label_panel(keypoints_panel, "Keypoints"),
        _label_panel(seg_panel, "Segmentation"),
    ]
    return np.hstack(panels)

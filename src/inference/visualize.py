"""Visualization utilities for CourtVisionNet inference results."""

import cv2
import numpy as np

from src.court_geometry import get_court_lines
from src.inference.predict import VISIBILITY_THRESHOLD as VISIBLE_THRESHOLD
from src.inference.predict import projected_court_lines

KEYPOINT_COLORS = (
    [(255, 0, 0)] * 5         # K0-K4:   Baseline L (red)
    + [(255, 136, 0)] * 5     # K5-K9:   Long Svc L (orange)
    + [(34, 204, 34)] * 5     # K10-K14: Short Svc L (green)
    + [(51, 136, 255)] * 5    # K15-K19: Short Svc R (blue)
    + [(204, 68, 255)] * 5    # K20-K24: Long Svc R (purple)
    + [(255, 68, 136)] * 5    # K25-K29: Baseline R (pink)
)

KEYPOINT_NAMES = [f"K{i}" for i in range(30)]

EXTRAPOLATED_THRESHOLD = 0.3

INLIER_COLOR = (0, 220, 0)
OUTLIER_COLOR = (0, 0, 255)
RAW_COLOR = (200, 200, 200)
PROJECTED_LINE_COLOR = (255, 0, 255)


def _to_pixels(points, width, height):
    px = np.asarray(points, dtype=np.float64).copy()
    px[:, 0] *= width
    px[:, 1] *= height
    return px


def _keypoints_to_pixels(detection, width, height):
    """Convert a detection's normalized [0, 1] keypoints to pixel coords."""
    return _to_pixels(detection.keypoints, width, height)


def _ui_scale(width, height):
    """Drawing scale so markers stay visible on large photos (1.0 up to 800 px)."""
    return max(1.0, max(width, height) / 800.0)


def _in_bounds(pt, width, height, margin=1000):
    return -margin <= pt[0] <= width + margin and -margin <= pt[1] <= height + margin


def _projected_lines(detection):
    lines = getattr(detection, "projected_lines", None)
    if lines is None and getattr(detection, "homography", None) is not None:
        lines = projected_court_lines(np.asarray(detection.homography))
    return lines or []


def draw_keypoints(image, detection, radius=6):
    """Draw detected (refined) keypoints with labels on a copy of `image`.

    Solid, filled circles mark confidently-detected keypoints (visibility
    above `VISIBLE_THRESHOLD`); hollow circles mark keypoints with visibility
    between `EXTRAPOLATED_THRESHOLD` and `VISIBLE_THRESHOLD`. Keypoints
    below both thresholds are not drawn.

    Args:
        image: (H, W, 3) BGR uint8 image
        detection: CourtDetection
        radius: circle radius in pixels

    Returns:
        (H, W, 3) BGR uint8 image with keypoints drawn
    """
    result = image.copy()
    h, w = image.shape[:2]
    s = _ui_scale(w, h)
    radius = int(round(radius * s))
    t = max(1, int(round(s)))
    pixel_kpts = _keypoints_to_pixels(detection, w, h)

    for i in range(len(pixel_kpts)):
        vis = detection.visibility[i]
        if vis <= EXTRAPOLATED_THRESHOLD or not _in_bounds(pixel_kpts[i], w, h):
            continue

        x, y = int(round(pixel_kpts[i, 0])), int(round(pixel_kpts[i, 1]))
        color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]

        if vis > VISIBLE_THRESHOLD:
            cv2.circle(result, (x, y), radius, color, -1)
            cv2.circle(result, (x, y), radius, (255, 255, 255), t)
        else:
            cv2.circle(result, (x, y), radius, color, t)

        name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"K{i}"
        cv2.putText(result, name, (x + radius + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * s, color, t)

    return result


def draw_refinement(image, detection, radius=6):
    """Raw decoder output vs homography-refined keypoints.

    - raw keypoints (visibility above EXTRAPOLATED_THRESHOLD): small grey crosses
    - refined keypoints: circles in the keypoint colour, ringed green for
      RANSAC inliers and red for outliers (no ring if no homography)
    - a thin line joins raw and refined where refinement moved the point
    - projected court lines from the homography in magenta

    Args:
        image: (H, W, 3) BGR uint8 image
        detection: CourtDetection (raw_keypoints/inlier_mask optional)

    Returns:
        (H, W, 3) BGR uint8 image
    """
    result = image.copy()
    h, w = image.shape[:2]
    s = _ui_scale(w, h)
    radius = int(round(radius * s))
    t = max(1, int(round(s)))

    for (x1, y1), (x2, y2) in _projected_lines(detection):
        if _in_bounds((x1, y1), w, h) and _in_bounds((x2, y2), w, h):
            cv2.line(result, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                     PROJECTED_LINE_COLOR, t, cv2.LINE_AA)

    raw = getattr(detection, "raw_keypoints", None)
    raw_px = _to_pixels(raw if raw is not None else detection.keypoints, w, h)
    ref_px = _keypoints_to_pixels(detection, w, h)
    inliers = getattr(detection, "inlier_mask", None)
    vis = np.asarray(detection.visibility)

    for i in range(len(ref_px)):
        rx, ry = int(round(raw_px[i, 0])), int(round(raw_px[i, 1]))
        fx, fy = int(round(ref_px[i, 0])), int(round(ref_px[i, 1]))
        raw_shown = vis[i] > EXTRAPOLATED_THRESHOLD and _in_bounds(raw_px[i], w, h)
        moved = np.hypot(rx - fx, ry - fy) > 2
        if raw_shown:
            cv2.drawMarker(result, (rx, ry), RAW_COLOR, cv2.MARKER_TILTED_CROSS, radius + 2, t)
        if not _in_bounds(ref_px[i], w, h) or not (raw_shown or moved):
            continue
        if raw_shown and moved:
            cv2.line(result, (rx, ry), (fx, fy), RAW_COLOR, t, cv2.LINE_AA)
        color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
        cv2.circle(result, (fx, fy), radius, color, -1 if vis[i] > VISIBLE_THRESHOLD else t)
        if inliers is not None:
            ring = INLIER_COLOR if inliers[i] else (OUTLIER_COLOR if raw_shown and vis[i] > VISIBLE_THRESHOLD else None)
            if ring is not None:
                cv2.circle(result, (fx, fy), radius + 2 * t, ring, 2 * t)
        cv2.putText(result, KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"K{i}",
                    (fx + radius + 2, fy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * s, color, t)

    status = getattr(detection, "homography_status", None)
    if status is not None:
        n_in = int(np.sum(inliers)) if inliers is not None else 0
        cv2.putText(result, f"H: {status} ({n_in} inliers)", (10, h - int(10 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 255, 255), t)
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

    mask_resized = _image_seg_mask(detection, w, h)
    if mask_resized is not None and mask_resized.any():
        overlay = result.copy()
        overlay[mask_resized > 0] = (0, 255, 255)  # yellow
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

    # Court lines between keypoints that are at least extrapolated.
    pixel_kpts = _keypoints_to_pixels(detection, w, h)
    for start_idx, end_idx in get_court_lines():
        if start_idx == end_idx:
            continue
        if detection.visibility[start_idx] > EXTRAPOLATED_THRESHOLD and \
                detection.visibility[end_idx] > EXTRAPOLATED_THRESHOLD and \
                _in_bounds(pixel_kpts[start_idx], w, h) and _in_bounds(pixel_kpts[end_idx], w, h):
            pt1 = tuple(int(v) for v in np.round(pixel_kpts[start_idx]))
            pt2 = tuple(int(v) for v in np.round(pixel_kpts[end_idx]))
            cv2.line(result, pt1, pt2, (0, 255, 0), 2)

    for (x1, y1), (x2, y2) in _projected_lines(detection):
        if _in_bounds((x1, y1), w, h) and _in_bounds((x2, y2), w, h):
            cv2.line(result, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                     PROJECTED_LINE_COLOR, 1)

    result = draw_keypoints(result, detection)

    cv2.putText(result, f"Conf: {detection.confidence:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return result


def _image_seg_mask(detection, width, height):
    """Binary (h, w) uint8 seg mask aligned with the original image, or None.

    Prefers ``seg_mask_image`` (already in image space). Otherwise falls
    back to ``seg_mask`` (canvas probabilities or a binary mask), resized;
    note a letterbox canvas is then only approximately aligned.
    """
    mask = getattr(detection, "seg_mask_image", None)
    if mask is None:
        mask = getattr(detection, "seg_mask", None)
        if mask is None:
            return None
        mask = np.asarray(mask)
        mask = (mask > 0.5) if mask.dtype.kind == "f" else (mask > 0)
    mask = np.asarray(mask).astype(np.uint8)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def _seg_mask_panel(image, detection):
    """Render the segmentation mask, at `image`'s size, as BGR."""
    h, w = image.shape[:2]
    mask = _image_seg_mask(detection, w, h)
    if mask is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)


def _label_panel(panel, text):
    """Stamp a label bar across the top of a panel (on a copy)."""
    labeled = panel.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(labeled, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return labeled


def create_debug_visualization(image, detection):
    """Build a side-by-side debug view: overlay | raw vs refined | segmentation.

    Args:
        image: (H, W, 3) BGR uint8 image
        detection: CourtDetection

    Returns:
        (H, W*3, 3) BGR uint8 image combining all three panels
    """
    panels = [
        _label_panel(draw_court_overlay(image, detection), "Overlay"),
        _label_panel(draw_refinement(image, detection), "Raw (x) vs refined (o)"),
        _label_panel(_seg_mask_panel(image, detection), "Segmentation"),
    ]
    return np.hstack(panels)

"""Evaluation metrics for CourtVisionNet."""

import numpy as np
from shapely.geometry import Polygon

from src.court_geometry import (
    CORNER_INDICES,
    COURT_KEYPOINTS_TEMPLATE,
    NUM_KEYPOINTS,
    compute_homography,
    project_points,
)

_MIN_POINTS = 4


def pck_at_k(pred_kps, gt_kps, visibility, k, image_size=640):
    """Percentage of Correct Keypoints at threshold k pixels.

    Args:
        pred_kps: (30, 2) normalized [0,1] predicted keypoints
        gt_kps: (30, 2) normalized [0,1] ground truth keypoints
        visibility: (30,) ground truth visibility (1.0=visible)
        k: pixel distance threshold
        image_size: image dimension for converting to pixels

    Returns:
        (per_keypoint, mean_accuracy) where per_keypoint is (30,) bool
        and mean_accuracy is float in [0, 1].
    """
    pred_px = np.asarray(pred_kps, dtype=np.float64) * image_size
    gt_px = np.asarray(gt_kps, dtype=np.float64) * image_size
    vis = np.asarray(visibility, dtype=np.float64)

    dists = np.linalg.norm(pred_px - gt_px, axis=1)
    correct = (dists < k) & (vis > 0.5)
    visible_mask = vis > 0.5

    n_visible = visible_mask.sum()
    if n_visible == 0:
        return np.zeros(NUM_KEYPOINTS, dtype=bool), 0.0

    mean_acc = float(correct.sum() / n_visible)
    return correct, mean_acc


def mean_reprojection_error(pred_kps, gt_kps, visibility, image_w, image_h):
    """Mean reprojection error in pixels.

    Estimates homography from visible predicted keypoints, projects
    the full template, and measures pixel distance to ground truth
    for visible points.

    Returns None if homography estimation fails.
    """
    pred = np.asarray(pred_kps, dtype=np.float64)
    gt = np.asarray(gt_kps, dtype=np.float64)
    vis = np.asarray(visibility, dtype=np.float64)
    visible_mask = vis > 0.5

    if visible_mask.sum() < _MIN_POINTS:
        return None

    src_pts = COURT_KEYPOINTS_TEMPLATE[visible_mask]
    dst_pts_px = pred[visible_mask] * np.array([image_w, image_h])

    try:
        H = compute_homography(src_pts, dst_pts_px)
    except ValueError:
        return None

    projected_px = project_points(H, COURT_KEYPOINTS_TEMPLATE)
    gt_px = gt * np.array([image_w, image_h])

    errors = np.linalg.norm(projected_px[visible_mask] - gt_px[visible_mask], axis=1)
    return float(errors.mean())


def court_iou(pred_kps, gt_kps, pred_vis, gt_vis):
    """IoU of the quadrilateral formed by the 4 outer court corners.

    Returns 0.0 if fewer than 4 corners are visible on either side.
    """
    pred = np.asarray(pred_kps, dtype=np.float64)
    gt = np.asarray(gt_kps, dtype=np.float64)
    p_vis = np.asarray(pred_vis, dtype=np.float64)
    g_vis = np.asarray(gt_vis, dtype=np.float64)

    for vis in (p_vis, g_vis):
        if not all(vis[i] > 0.5 for i in CORNER_INDICES):
            return 0.0

    pred_corners = pred[CORNER_INDICES]
    gt_corners = gt[CORNER_INDICES]

    try:
        poly_pred = Polygon(pred_corners)
        poly_gt = Polygon(gt_corners)
        if not poly_pred.is_valid or not poly_gt.is_valid:
            return 0.0
        intersection = poly_pred.intersection(poly_gt).area
        union = poly_pred.union(poly_gt).area
        if union == 0:
            return 0.0
        return float(intersection / union)
    except Exception:
        return 0.0


def segmentation_iou(pred_mask, gt_mask, threshold=0.5):
    """Binary IoU between predicted and ground truth segmentation masks.

    Args:
        pred_mask: (H, W) float array (probabilities or raw values)
        gt_mask: (H, W) float array (binary ground truth)
        threshold: binarization threshold for pred_mask

    Returns:
        float in [0, 1].
    """
    pred_bin = (np.asarray(pred_mask) >= threshold).astype(bool)
    gt_bin = (np.asarray(gt_mask) >= threshold).astype(bool)

    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)

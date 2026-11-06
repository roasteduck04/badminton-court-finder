"""Evaluation metrics for CourtVisionNet.

Metric definitions
==================

Notation. An image has size (h, w) in ORIGINAL pixels. Ground-truth keypoints
g_k and predictions p_k (k = 0..29) are normalized [0,1] coordinates of the
original image and are mapped to pixels by multiplying with (w, h). The scored
set is V = {k : GT visibility_k > 0.5}; predictions for k outside V are never
scored for localisation. Images with V empty contribute no points and are
excluded from every per-image average (their count is reported as
``n_images_no_visible``). "Pooled" metrics pool all (image, k in V) pairs;
"img_" metrics first average over V within an image, then over images.
All PCK-style comparisons are ``error <= threshold``.

Keypoint error (no homography involved):
  e_k            = ||p_k - g_k||_2 in original pixels.
  e_k^ref        = e_k * L_ref / max(h, w), the error at the reference
                   resolution (L_ref = 640 = side of the model's letterbox
                   canvas, so e^ref is "letterbox-canvas pixels").
  err_px_mean / err_px_median     pooled mean / median of e_k.
  err_ref_mean / err_ref_median   pooled mean / median of e_k^ref.
  img_err_px_mean / img_err_ref_mean   per-image mean, averaged over images.
  pck@{5,10,20}px                 fraction of points with e_k^ref <= t.

Normalized PCK:
  D_court        = diagonal of the axis-aligned bbox of the 4 outer court
                   corners projected by H_gt, where H_gt is the DLT homography
                   template(metres)->image fitted to ALL GT-visible points.
                   Fallback (H_gt undefined: no 4 visible points in general
                   position, a corner projects behind the camera, or the
                   diagonal exceeds 20 * D_img): bbox diagonal of the GT-visible
                   points. Images whose D_court is 0 (one visible point) are
                   excluded from normalized metrics.
  D_img          = sqrt(h^2 + w^2).
  pck_norm@a     fraction with e_k <= a * D_court, a in {0.01, 0.02, 0.05}.
  kla@a          fraction with e_k <= a * D_img, a in {0.05, 0.1}
                 (keypoint localisation accuracy as in CourtKeyNet).
  auc_norm       area under the pooled pck_norm curve for a in [0, 0.1]
                 (101 evenly spaced thresholds, trapezoidal rule) divided by
                 0.1, so in [0, 1].
  pcr@...        percentage of images, among those with >= 1 GT-visible outer
                 corner {0, 4, 25, 29}, in which ALL GT-visible outer corners
                 are within the threshold (normalized a, or px at L_ref).

Court-plane error:
  err_m_*        ||H_gt^-1(p_k) - T_k||_2 in metres, T_k the template point,
                 for k in V of images where H_gt exists. A prediction beyond
                 the GT horizon (wrong side of the vanishing line) gets error
                 +inf: it counts in the median and is reported as
                 ``n_beyond_horizon``; err_m_mean averages finite errors only.

Symmetry-aware variant (prefix ``sym``): per image the prediction is relabeled
  with whichever of {identity, rot180} (SYMMETRY_PERMS) minimises the mean
  e_k over V, then every keypoint metric above is recomputed. It measures
  localisation independent of the 180-degree label ambiguity and is reported
  next to, never instead of, the strict metric. The chosen perm is stored per
  image as ``sym_perm``.

Homography from predictions (no GT oracle):
  H_pred         RANSAC homography template->image fitted on predicted points
                 whose predicted visibility prob > 0.5, reprojection threshold
                 1% of D_img. Success = the predicted-visible points contain 4
                 template points in general position, RANSAC returns a finite
                 H, its inliers again contain 4 points in general position and
                 all inliers lie in front of the camera.
  homography_success_rate   successes / images with |V| > 0.
  reproj_*       ||H_pred(T_k) - g_k|| for k in V (original px and at L_ref),
                 over successful images only.
  court_iou      IoU of the full court quadrilateral projected by H_gt and by
                 H_pred, each clipped to the camera's front half-space and to
                 the image rectangle. Computed for images with H_gt whose
                 projected court intersects the image; failed H_pred counts
                 as IoU 0 (``court_iou_mean``); ``court_iou_mean_success``
                 averages successful fits only.

Visibility: pooled precision / recall / F1 / accuracy over all 30 keypoints of
  all images; predicted visible = prob > 0.5, positive = GT-visible.

Segmentation: seg_iou = binary IoU between the predicted line mask
  (prob >= 0.5) and the GT line mask rendered in the 640 x 640 letterbox canvas
  with the training line thickness, drawing only segments whose endpoints are
  both GT-visible and excluding the virtual net centre segment (12, 17), both
  restricted to the letterbox content region. Images where both masks are
  empty are skipped.

Confidence intervals: 95% percentile bootstrap over images (resampling images
  with replacement, ``n_boot`` resamples, fixed seed) of the headline numbers.

Legacy functions ``pck_at_k``, ``mean_reprojection_error``, ``court_iou`` and
``segmentation_iou`` keep their old behaviour for backward compatibility; they
are NOT the paper metrics (see their docstrings).
"""

import math

import cv2
import numpy as np

from src.court_geometry import (
    CORNER_INDICES,
    COURT_KEYPOINTS_TEMPLATE,
    COURT_LENGTH,
    COURT_WIDTH_DOUBLES,
    NUM_KEYPOINTS,
    SYMMETRY_PERMS,
    compute_homography,
    court_orientation,
    get_court_lines,
    has_homography_support,
    project_points,
)

_MIN_POINTS = 4

REF_LONG_SIDE = 640
PX_THRESHOLDS = (5, 10, 20)
NORM_ALPHAS = (0.01, 0.02, 0.05)
KLA_ALPHAS = (0.05, 0.1)
AUC_MAX = 0.1
AUC_STEPS = 101
VIS_THRESHOLD = 0.5
RANSAC_FRAC = 0.01
SYM_PERM_NAMES = ("identity", "rot180")
NET_SEGMENT = (12, 17)
N_VISIBLE_BUCKETS = ((1, 3), (4, 7), (8, 14), (15, 22), (23, 30))
_MAX_DIAG_FACTOR = 20.0

# Court outline in template metres, cyclic order matching CORNER_INDICES.
COURT_QUAD_TEMPLATE = np.array(
    [[0.0, 0.0], [COURT_LENGTH, 0.0],
     [COURT_LENGTH, COURT_WIDTH_DOUBLES], [0.0, COURT_WIDTH_DOUBLES]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _np(x, dtype=np.float64):
    """numpy array from numpy / list / (CPU or GPU) torch tensor."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else float("nan")


def _nanmedian(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size else float("nan")


def _frac(mask_num, mask_den):
    den = int(np.sum(mask_den))
    return float(np.sum(mask_num) / den) if den else float("nan")


def _key(x):
    """Threshold as a short string key: 10 -> '10', 0.05 -> '0.05'."""
    return f"{x:g}"


def to_pixels(kps, image_size_hw):
    """Normalized (..., 2) coords -> pixels of an image of size [h, w]."""
    h, w = image_size_hw
    return _np(kps)[..., :2] * np.array([w, h], dtype=np.float64)


# ---------------------------------------------------------------------------
# Batch API (used by training validation and by evaluation)
# ---------------------------------------------------------------------------

def keypoint_errors(pred_kps, gt_kps, gt_vis, image_size_hw):
    """Per-keypoint Euclidean pixel errors, NaN where GT is not visible.

    Args:
        pred_kps: (K, 2) or (B, K, 2) normalized predictions.
        gt_kps: same shape, normalized ground truth.
        gt_vis: (K,) or (B, K) GT visibility (> 0.5 = visible).
        image_size_hw: [h, w] shared by all samples, or (B, 2) per sample.
            Pass the canvas size (e.g. (640, 640)) to get letterbox-canvas
            pixels, or the original size to get original pixels.

    Returns:
        (K,) or (B, K) float64 errors in pixels. Non-finite predictions give
        +inf (always a miss); GT-invisible points give NaN.
    """
    pred = _np(pred_kps)
    gt = _np(gt_kps)
    vis = _np(gt_vis)
    size = _np(image_size_hw)
    scale = size[..., ::-1]  # (w, h)
    if size.ndim == 2:
        scale = scale[:, None, :]
    d = (pred[..., :2] - gt[..., :2]) * scale
    err = np.hypot(d[..., 0], d[..., 1])
    err = np.where(np.isfinite(err), err, np.inf)
    err[~(vis > VIS_THRESHOLD)] = np.nan
    return err


def summarize_errors(errors, thresholds=PX_THRESHOLDS):
    """Pooled and per-image statistics of a (B, K) (or (K,)) error array.

    NaN entries are unscored (GT-invisible). Rows with no scored entry are
    excluded from the per-image averages and counted in n_images_no_visible.
    Values that cannot be computed (no points) are NaN.

    Returns dict with keys: n_points, n_images, n_images_no_visible,
    err_mean, err_median (pooled), img_err_mean (per-image mean of means),
    pck@<t> (pooled fraction with err <= t) and img_pck@<t> (per-image).
    """
    E = np.atleast_2d(_np(errors))
    valid = ~np.isnan(E)
    n_vis = valid.sum(axis=1)
    has = n_vis > 0
    pts = E[valid]
    out = {
        "n_points": int(pts.size),
        "n_images": int(E.shape[0]),
        "n_images_no_visible": int((~has).sum()),
        "err_mean": float(pts.mean()) if pts.size else float("nan"),
        "err_median": float(np.median(pts)) if pts.size else float("nan"),
    }
    with np.errstate(invalid="ignore"):
        row_mean = np.where(has, np.nansum(np.where(valid, E, 0.0), axis=1) / np.maximum(n_vis, 1), np.nan)
    out["img_err_mean"] = _nanmean(row_mean[has])
    for t in thresholds:
        correct = valid & (np.nan_to_num(E, nan=np.inf) <= t)
        out[f"pck@{_key(t)}"] = _frac(correct, valid)
        per_img = correct.sum(axis=1)[has] / n_vis[has]
        out[f"img_pck@{_key(t)}"] = float(per_img.mean()) if per_img.size else float("nan")
    return out


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------

def _w(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ H[2, :2] + H[2, 2]


def _orient_homography(H, ref_template_pts):
    """Scale H by +-1 so that the reference template points have w > 0.

    Returns None if H is non-finite or the reference points lie on both sides
    of the vanishing line (an impossible camera)."""
    if H is None or not np.all(np.isfinite(H)):
        return None
    w = _w(H, ref_template_pts)
    if np.all(w > 0):
        return H
    if np.all(w < 0):
        return -H
    return None


def fit_gt_homography(gt_px, gt_vis):
    """DLT homography template->image on all GT-visible points, or None."""
    vis = _np(gt_vis) > VIS_THRESHOLD
    gt_px = _np(gt_px)
    if not has_homography_support(vis):
        return None
    H, _ = cv2.findHomography(COURT_KEYPOINTS_TEMPLATE[vis], gt_px[vis], 0)
    return _orient_homography(H, COURT_KEYPOINTS_TEMPLATE[vis])


def fit_pred_homography(pred_px, pred_vis_prob, image_size_hw,
                        vis_threshold=VIS_THRESHOLD, ransac_frac=RANSAC_FRAC):
    """RANSAC homography template->image from predictions only.

    Uses predicted points whose visibility probability exceeds
    ``vis_threshold``; reprojection threshold = ransac_frac * image diagonal.

    Returns (H, inlier_mask (30,) bool) or (None, None) on failure.
    """
    pred_px = _np(pred_px)
    vis = (_np(pred_vis_prob) > vis_threshold) & np.all(np.isfinite(pred_px), axis=1)
    if not has_homography_support(vis):
        return None, None
    h, w = image_size_hw
    thr = ransac_frac * math.hypot(h, w)
    H, mask = cv2.findHomography(
        COURT_KEYPOINTS_TEMPLATE[vis], pred_px[vis], cv2.RANSAC, thr,
        maxIters=2000, confidence=0.995,
    )
    if H is None or mask is None:
        return None, None
    inliers = np.zeros(NUM_KEYPOINTS, dtype=bool)
    inliers[np.flatnonzero(vis)] = mask.ravel() > 0
    if not has_homography_support(inliers):
        return None, None
    H = _orient_homography(H, COURT_KEYPOINTS_TEMPLATE[inliers])
    if H is None:
        return None, None
    return H, inliers


def court_diagonal(gt_px, gt_vis, image_size_hw, H_gt=None):
    """Court-scale normaliser D_court (see module docstring).

    Returns (diagonal_px, source) with source in {"homography", "bbox", None};
    diagonal is NaN when undefined.
    """
    gt_px = _np(gt_px)
    vis = _np(gt_vis) > VIS_THRESHOLD
    h, w = image_size_hw
    d_img = math.hypot(h, w)
    if H_gt is not None and np.all(_w(H_gt, COURT_QUAD_TEMPLATE) > 0):
        corners = project_points(H_gt, COURT_QUAD_TEMPLATE)
        ext = corners.max(axis=0) - corners.min(axis=0)
        diag = float(np.hypot(ext[0], ext[1]))
        if np.isfinite(diag) and 0 < diag <= _MAX_DIAG_FACTOR * d_img:
            return diag, "homography"
    if vis.sum() >= 1:
        p = gt_px[vis]
        ext = p.max(axis=0) - p.min(axis=0)
        diag = float(np.hypot(ext[0], ext[1]))
        if diag > 0:
            return diag, "bbox"
    return float("nan"), None


def court_plane_errors(pred_px, gt_px, gt_vis, H_gt):
    """Errors in metres on the court plane via the inverse GT homography.

    Returns (30,) array: NaN for GT-invisible points (or all NaN if H_gt is
    None), +inf for predictions beyond the GT horizon.
    """
    out = np.full(NUM_KEYPOINTS, np.nan)
    if H_gt is None:
        return out
    vis = _np(gt_vis) > VIS_THRESHOLD
    pred_px = _np(pred_px)
    try:
        Hinv = np.linalg.inv(H_gt)
    except np.linalg.LinAlgError:
        return out
    # Template points have w > 0 under H_gt, so their images map back with a
    # common sign of w under Hinv; a prediction with the other sign lies past
    # the vanishing line.
    ref_sign = np.sign(np.median(_w(Hinv, _np(gt_px)[vis])))
    p = pred_px[vis]
    wq = _w(Hinv, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = project_points(Hinv, p)
    err = np.linalg.norm(q - COURT_KEYPOINTS_TEMPLATE[vis], axis=1)
    bad = (np.sign(wq) != ref_sign) | ~np.isfinite(err)
    err[bad] = np.inf
    out[vis] = err
    return out


# ---------------------------------------------------------------------------
# Polygon clipping (convex, shapely-free)
# ---------------------------------------------------------------------------

def _clip_halfplane(poly, a, b, c):
    """Keep the part of polygon ``poly`` with a*x + b*y + c >= 0."""
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    n = len(poly)
    if n == 0:
        return poly
    f = poly @ np.array([a, b]) + c
    out = []
    for i in range(n):
        j = (i + 1) % n
        if f[i] >= 0:
            out.append(poly[i])
        if (f[i] >= 0) != (f[j] >= 0):
            t = f[i] / (f[i] - f[j])
            out.append(poly[i] + t * (poly[j] - poly[i]))
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def polygon_area(poly):
    """Absolute shoelace area of a simple polygon."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0


def clip_convex(subject, clip):
    """Sutherland-Hodgman: ``subject`` polygon clipped by convex ``clip``."""
    clip = np.asarray(clip, dtype=np.float64).reshape(-1, 2)
    out = np.asarray(subject, dtype=np.float64).reshape(-1, 2)
    if len(clip) < 3 or len(out) < 3:
        return np.zeros((0, 2))
    if _signed_area(clip) < 0:
        clip = clip[::-1]
    for i in range(len(clip)):
        p, q = clip[i], clip[(i + 1) % len(clip)]
        a = -(q[1] - p[1])
        b = q[0] - p[0]
        c = -(a * p[0] + b * p[1])
        out = _clip_halfplane(out, a, b, c)
        if len(out) == 0:
            break
    return out


def convex_polygon_iou(poly_a, poly_b):
    """IoU of two convex polygons; NaN if both are empty."""
    area_a, area_b = polygon_area(poly_a), polygon_area(poly_b)
    if area_a <= 0 and area_b <= 0:
        return float("nan")
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter = polygon_area(clip_convex(poly_a, poly_b))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else float("nan")


def projected_court_polygon(H, image_size_hw):
    """The court outline projected by H (template->image, w>0 in front),
    clipped to the front half-space and to the image rectangle."""
    if H is None:
        return np.zeros((0, 2))
    w_scale = np.abs(_w(H, COURT_QUAD_TEMPLATE)).max()
    eps = 1e-6 * max(w_scale, 1e-12)
    front = _clip_halfplane(COURT_QUAD_TEMPLATE, H[2, 0], H[2, 1], H[2, 2] - eps)
    if len(front) < 3:
        return np.zeros((0, 2))
    img_poly = project_points(H, front)
    h, w = image_size_hw
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    return clip_convex(img_poly, rect)


def court_iou_homography(H_gt, H_pred, image_size_hw):
    """Court IoU of the full court quad under two homographies (see module
    docstring). NaN if H_gt is None or the GT court misses the image; 0 if
    H_pred is None."""
    if H_gt is None:
        return float("nan")
    gt_poly = projected_court_polygon(H_gt, image_size_hw)
    if polygon_area(gt_poly) <= 0:
        return float("nan")
    if H_pred is None:
        return 0.0
    return convex_polygon_iou(gt_poly, projected_court_polygon(H_pred, image_size_hw))


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def render_line_mask(kps, visibility, size, thickness, exclude=(NET_SEGMENT,)):
    """(size, size) uint8 {0,1} line mask from normalized canvas keypoints.

    Pixel-centre convention: normalized x maps to pixel x*size - 0.5. Draws
    every court segment whose two endpoints are visible, except ``exclude``.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    kps = _np(kps)
    vis = _np(visibility) > VIS_THRESHOLD
    shift = 4
    px = np.round((kps * size - 0.5) * (1 << shift)).astype(np.int64)
    skip = {tuple(sorted(s)) for s in exclude}
    for a, b in get_court_lines():
        if tuple(sorted((a, b))) in skip or not (vis[a] and vis[b]):
            continue
        if not (np.all(np.isfinite(kps[a])) and np.all(np.isfinite(kps[b]))):
            continue
        cv2.line(mask, (int(px[a, 0]), int(px[a, 1])), (int(px[b, 0]), int(px[b, 1])),
                 1, int(thickness), cv2.LINE_8, shift)
    return mask


def segmentation_iou(pred_mask, gt_mask, threshold=0.5, empty_value=0.0):
    """Binary IoU between two masks, both binarized at ``threshold``.

    Returns ``empty_value`` when both masks are empty (legacy: 0.0; the
    evaluation passes NaN so such images are skipped).
    """
    pred_bin = np.asarray(pred_mask) >= threshold
    gt_bin = np.asarray(gt_mask) >= threshold
    union = (pred_bin | gt_bin).sum()
    if union == 0:
        return empty_value
    return float((pred_bin & gt_bin).sum() / union)


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

def symmetric_relabel(pred_kps, gt_kps, gt_vis, image_size_hw, perm_names=SYM_PERM_NAMES):
    """Relabel the prediction with the symmetry that minimises mean error.

    Returns (perm_name, permuted_pred_kps, permuted_pred_index_order).
    Ties keep the earlier name (identity first).
    """
    pred = _np(pred_kps)
    best = None
    for name in perm_names:
        perm = SYMMETRY_PERMS[name]
        e = keypoint_errors(pred[perm], gt_kps, gt_vis, image_size_hw)
        m = _nanmean(e)
        m = np.inf if np.isnan(m) else m
        if best is None or m < best[0]:
            best = (m, name, perm)
    _, name, perm = best
    return name, pred[perm], perm


def _list(a):
    """JSON-friendly list: NaN -> None (unscored), +-inf -> "inf" (a miss)."""
    out = []
    for v in np.ravel(a):
        if v is None or np.isnan(v):
            out.append(None)
        elif np.isinf(v):
            out.append("inf")
        else:
            out.append(float(v))
    return out


def _num(v):
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def evaluate_image(pred_kps, pred_vis, gt_kps, gt_vis, image_size_hw,
                   ref_long_side=REF_LONG_SIDE, ransac_frac=RANSAC_FRAC, **extra):
    """All per-image quantities used by ``aggregate``.

    Args:
        pred_kps: (30, 2) normalized ORIGINAL-image predictions.
        pred_vis: (30,) predicted visibility probabilities (or None = all 1).
        gt_kps, gt_vis: GT in the same convention.
        image_size_hw: [h, w] of the original image.
        **extra: stored verbatim in the row (name, source, group, seg_iou, ...).

    Returns a JSON-serialisable dict (NaN/inf stored as None).
    """
    h, w = int(image_size_hw[0]), int(image_size_hw[1])
    size = (h, w)
    pred = _np(pred_kps)[:, :2]
    gt = _np(gt_kps)[:, :2]
    gvis = _np(gt_vis) > VIS_THRESHOLD
    pvis = np.ones(NUM_KEYPOINTS) if pred_vis is None else _np(pred_vis)
    pred_px, gt_px = to_pixels(pred, size), to_pixels(gt, size)

    ref_scale = ref_long_side / max(h, w)
    d_img = math.hypot(h, w)
    H_gt = fit_gt_homography(gt_px, gvis)
    d_court, d_src = court_diagonal(gt_px, gvis, size, H_gt)

    err = keypoint_errors(pred, gt, gvis, size)
    err_m = court_plane_errors(pred_px, gt_px, gvis, H_gt)

    sym_perm, pred_sym, _ = symmetric_relabel(pred, gt, gvis, size)
    err_sym = keypoint_errors(pred_sym, gt, gvis, size)
    err_m_sym = court_plane_errors(to_pixels(pred_sym, size), gt_px, gvis, H_gt)

    H_pred, inliers = fit_pred_homography(pred_px, pvis, size, ransac_frac=ransac_frac)
    reproj = np.full(NUM_KEYPOINTS, np.nan)
    if H_pred is not None:
        proj = project_points(H_pred, COURT_KEYPOINTS_TEMPLATE)
        r = np.linalg.norm(proj - gt_px, axis=1)
        reproj[gvis] = np.where(np.isfinite(r[gvis]), r[gvis], np.inf)
    iou = court_iou_homography(H_gt, H_pred, size)

    pv = pvis > VIS_THRESHOLD
    orient_gt = court_orientation(gt, gvis, size)
    orient_pred = court_orientation(pred, pv & np.all(np.isfinite(pred), axis=1), size)

    row = {
        "image_size": [h, w],
        "n_visible": int(gvis.sum()),
        "ref_scale": ref_scale,
        "img_diag_px": d_img,
        "court_diag_px": _num(d_court),
        "court_diag_source": d_src,
        "gt_homography": H_gt is not None,
        "err_px": _list(err),
        "err_m": _list(err_m),
        "sym_perm": sym_perm,
        "err_sym_px": _list(err_sym),
        "err_sym_m": _list(err_m_sym),
        "h_success": H_pred is not None,
        "h_inliers": int(inliers.sum()) if inliers is not None else 0,
        "h_inlier_mask": inliers.astype(int).tolist() if inliers is not None else None,
        "reproj_px": _list(reproj),
        "court_iou": _num(iou),
        "vis_tp": int((pv & gvis).sum()),
        "vis_fp": int((pv & ~gvis).sum()),
        "vis_fn": int((~pv & gvis).sum()),
        "vis_tn": int((~pv & ~gvis).sum()),
        "gt_handedness": orient_gt[0] if orient_gt else None,
        "pred_handedness": orient_pred[0] if orient_pred else None,
        "gt_row_alignment": orient_gt[1] if orient_gt else None,
        "pred_row_alignment": orient_pred[1] if orient_pred else None,
        "pred_keypoints": [[float(x), float(y)] for x, y in pred],
        "pred_visibility": [float(v) for v in pvis],
    }
    for k, v in extra.items():
        row[k] = v
    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _decode(v):
    """Inverse of the row encoding: None -> NaN, "inf" -> +inf."""
    return np.nan if v is None else float(v)


def _arr(rows, key, shape_tail=()):
    vals = []
    for r in rows:
        v = r.get(key)
        if shape_tail:
            vals.append(np.full(shape_tail, np.nan) if v is None else [_decode(x) for x in v])
        else:
            vals.append(_decode(v))
    if not vals:
        return np.zeros((0,) + shape_tail)
    return np.array(vals, dtype=np.float64)


def stack_rows(rows):
    """Per-image rows -> dict of arrays (one entry per image)."""
    K = (NUM_KEYPOINTS,)
    return {
        "err_px": _arr(rows, "err_px", K),
        "err_m": _arr(rows, "err_m", K),
        "err_sym_px": _arr(rows, "err_sym_px", K),
        "err_sym_m": _arr(rows, "err_sym_m", K),
        "reproj_px": _arr(rows, "reproj_px", K),
        "pred_vis": _arr(rows, "pred_visibility", K),
        "ref_scale": _arr(rows, "ref_scale"),
        "img_diag": _arr(rows, "img_diag_px"),
        "court_diag": _arr(rows, "court_diag_px"),
        "h_success": np.array([bool(r.get("h_success")) for r in rows], dtype=bool),
        "court_iou": _arr(rows, "court_iou"),
        "vis": np.array([[r.get("vis_tp", 0), r.get("vis_fp", 0), r.get("vis_fn", 0),
                          r.get("vis_tn", 0)] for r in rows], dtype=np.float64).reshape(-1, 4),
        "seg_iou": _arr(rows, "seg_iou"),
    }


def _subset(arr, idx):
    return {k: v[idx] for k, v in arr.items()}


def _keypoint_block(E, ref_scale, court_diag, img_diag, Em, curve=True):
    """Every keypoint metric for an (N, 30) original-pixel error array."""
    valid = ~np.isnan(E)
    n_vis = valid.sum(axis=1)
    has = n_vis > 0
    Einf = np.where(valid, E, np.inf)
    Eref = E * ref_scale[:, None]

    out = {
        "n_images": int(len(E)),
        "n_images_with_visible": int(has.sum()),
        "n_images_no_visible": int((~has).sum()),
        "n_points": int(valid.sum()),
    }
    pts, pts_ref = E[valid], Eref[valid]
    out["err_px_mean"] = _nanmean(pts)
    out["err_px_median"] = _nanmedian(pts)
    out["err_ref_mean"] = _nanmean(pts_ref)
    out["err_ref_median"] = _nanmedian(pts_ref)

    def img_mean(X):
        with np.errstate(invalid="ignore"):
            m = np.where(valid, X, 0.0).sum(axis=1) / np.maximum(n_vis, 1)
        return _nanmean(m[has]) if has.any() else float("nan")

    out["img_err_px_mean"] = img_mean(E)
    out["img_err_ref_mean"] = img_mean(Eref)

    def pck(correct, mask):
        """pooled, per-image."""
        rows_ok = mask.sum(axis=1) > 0
        pooled = _frac(correct & mask, mask)
        per = (correct & mask).sum(axis=1)[rows_ok] / mask.sum(axis=1)[rows_ok]
        return pooled, (float(per.mean()) if per.size else float("nan"))

    Eref_inf = np.where(valid, Eref, np.inf)
    for t in PX_THRESHOLDS:
        out[f"pck@{_key(t)}px"], out[f"img_pck@{_key(t)}px"] = pck(Eref_inf <= t, valid)

    norm_ok = np.isfinite(court_diag) & (court_diag > 0)
    nvalid = valid & norm_ok[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        En = np.where(nvalid, Einf / court_diag[:, None], np.inf)
        Ei = np.where(valid, Einf / img_diag[:, None], np.inf)
    out["n_points_norm"] = int(nvalid.sum())
    for a in NORM_ALPHAS:
        out[f"pck_norm@{_key(a)}"], out[f"img_pck_norm@{_key(a)}"] = pck(En <= a, nvalid)
    for a in KLA_ALPHAS:
        out[f"kla@{_key(a)}"], out[f"img_kla@{_key(a)}"] = pck(Ei <= a, valid)

    ts = np.linspace(0.0, AUC_MAX, AUC_STEPS)
    n_norm = nvalid.sum()
    if n_norm:
        en = En[nvalid]
        curve_vals = (en[None, :] <= ts[:, None]).mean(axis=1)
        out["auc_norm"] = float(np.trapezoid(curve_vals, ts) / AUC_MAX) if hasattr(np, "trapezoid") \
            else float(np.trapz(curve_vals, ts) / AUC_MAX)
    else:
        curve_vals = np.full_like(ts, np.nan)
        out["auc_norm"] = float("nan")
    if curve:
        out["pck_curve"] = {"thresholds": [float(t) for t in ts], "pck_norm": _list(curve_vals)}

    corners = list(CORNER_INDICES)
    cv = valid[:, corners]
    cimg = cv.any(axis=1)
    out["pcr_n_images"] = int(cimg.sum())
    for a in NORM_ALPHAS:
        ok = (En[:, corners] <= a) | ~cv
        ok_img = ok.all(axis=1) & cimg & norm_ok
        den = (cimg & norm_ok).sum()
        out[f"pcr_norm@{_key(a)}"] = float(ok_img.sum() / den) if den else float("nan")
    for t in PX_THRESHOLDS:
        ok = (Eref_inf[:, corners] <= t) | ~cv
        out[f"pcr@{_key(t)}px"] = _frac(ok.all(axis=1) & cimg, cimg)

    m_valid = ~np.isnan(Em)
    m = Em[m_valid]
    out["n_points_m"] = int(m.size)
    out["n_beyond_horizon"] = int(np.isinf(m).sum())
    out["err_m_median"] = float(np.median(m)) if m.size else float("nan")
    out["err_m_mean"] = _nanmean(m[np.isfinite(m)])
    return out


def _core_summary(arr, curve=True):
    """Summary without breakdowns or CIs."""
    out = {
        "n_images": int(len(arr["err_px"])),
        "strict": _keypoint_block(arr["err_px"], arr["ref_scale"], arr["court_diag"],
                                  arr["img_diag"], arr["err_m"], curve),
        "sym": _keypoint_block(arr["err_sym_px"], arr["ref_scale"], arr["court_diag"],
                               arr["img_diag"], arr["err_sym_m"], curve),
    }

    # Homography from predictions
    has_vis = (~np.isnan(arr["err_px"])).any(axis=1)
    succ = arr["h_success"] & has_vis
    R = arr["reproj_px"][succ]
    Rref = R * arr["ref_scale"][succ][:, None]
    rv = ~np.isnan(R)
    hom = {
        "n_attempted": int(has_vis.sum()),
        "n_success": int(succ.sum()),
        "success_rate": float(succ.sum() / has_vis.sum()) if has_vis.any() else float("nan"),
        "reproj_px_mean": _nanmean(R[rv]),
        "reproj_px_median": _nanmedian(R[rv]),
        "reproj_ref_mean": _nanmean(Rref[rv]),
        "reproj_ref_median": _nanmedian(Rref[rv]),
    }
    if rv.any():
        with np.errstate(invalid="ignore"):
            per = np.where(rv, Rref, 0).sum(1) / np.maximum(rv.sum(1), 1)
        hom["img_reproj_ref_mean"] = _nanmean(per[rv.any(1)])
        Rinf = np.where(rv, Rref, np.inf)
        for t in PX_THRESHOLDS:
            hom[f"reproj_pck@{_key(t)}px"] = _frac((Rinf <= t) & rv, rv)
    else:
        hom["img_reproj_ref_mean"] = float("nan")
        for t in PX_THRESHOLDS:
            hom[f"reproj_pck@{_key(t)}px"] = float("nan")
    out["homography"] = hom

    iou = arr["court_iou"]
    iv = ~np.isnan(iou)
    out["court_iou"] = {
        "n": int(iv.sum()),
        "mean": _nanmean(iou[iv]),
        "median": _nanmedian(iou[iv]),
        "mean_success": _nanmean(iou[iv & arr["h_success"]]),
        "n_success": int((iv & arr["h_success"]).sum()),
    }

    tp, fp, fn, tn = arr["vis"].sum(axis=0) if len(arr["vis"]) else (0, 0, 0, 0)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp and prec + rec) else (0.0 if tp + fp + fn else float("nan"))
    tot = tp + fp + fn + tn
    out["visibility"] = {
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "accuracy": float((tp + tn) / tot) if tot else float("nan"),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

    seg = arr["seg_iou"]
    sv = ~np.isnan(seg)
    out["segmentation"] = {"n": int(sv.sum()), "seg_iou_mean": _nanmean(seg[sv])}
    return out


# (label, path in core summary). The first block is the paper's headline set.
HEADLINE_METRICS = [
    ("pck@5px", ("strict", "pck@5px")),
    ("pck@10px", ("strict", "pck@10px")),
    ("pck@20px", ("strict", "pck@20px")),
    ("pck_norm@0.01", ("strict", "pck_norm@0.01")),
    ("pck_norm@0.02", ("strict", "pck_norm@0.02")),
    ("pck_norm@0.05", ("strict", "pck_norm@0.05")),
    ("kla@0.05", ("strict", "kla@0.05")),
    ("kla@0.1", ("strict", "kla@0.1")),
    ("auc_norm", ("strict", "auc_norm")),
    ("err_ref_mean", ("strict", "err_ref_mean")),
    ("err_ref_median", ("strict", "err_ref_median")),
    ("img_err_ref_mean", ("strict", "img_err_ref_mean")),
    ("img_pck@10px", ("strict", "img_pck@10px")),
    ("pcr_norm@0.02", ("strict", "pcr_norm@0.02")),
    ("err_m_median", ("strict", "err_m_median")),
    ("sym_pck@10px", ("sym", "pck@10px")),
    ("sym_pck_norm@0.02", ("sym", "pck_norm@0.02")),
    ("sym_auc_norm", ("sym", "auc_norm")),
    ("sym_err_ref_median", ("sym", "err_ref_median")),
    ("homography_success_rate", ("homography", "success_rate")),
    ("reproj_ref_median", ("homography", "reproj_ref_median")),
    ("court_iou_mean", ("court_iou", "mean")),
    ("vis_f1", ("visibility", "f1")),
    ("seg_iou_mean", ("segmentation", "seg_iou_mean")),
]


def headline(core):
    """Flat {label: value} of HEADLINE_METRICS from a core summary."""
    out = {}
    for label, (sec, key) in HEADLINE_METRICS:
        out[label] = core.get(sec, {}).get(key, float("nan"))
    return out


def bootstrap_ci(n_items, stat_fn, n_boot=1000, seed=0, level=0.95):
    """Percentile bootstrap CI over items.

    ``stat_fn(idx)`` computes a dict {name: float} (or a float) on the items
    selected by integer index array ``idx``. Returns {name: [lo, hi]} (or
    [lo, hi]); NaN statistics are ignored, all-NaN gives [nan, nan].
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items) if n_items else np.zeros(0, dtype=int)
        samples.append(stat_fn(idx))
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2

    def ci(vals):
        v = np.asarray(vals, dtype=np.float64)
        v = v[~np.isnan(v)]
        if not v.size:
            return [float("nan"), float("nan")]
        return [float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))]

    if samples and isinstance(samples[0], dict):
        return {k: ci([s[k] for s in samples]) for k in samples[0]}
    return ci(samples)


def _bucket_label(lo, hi):
    return f"{lo}-{hi}"


def aggregate(rows, n_boot=1000, seed=0, ci_level=0.95, breakdowns=True):
    """Summary dict from per-image rows produced by ``evaluate_image``.

    Keys: n_images, strict, sym, homography, court_iou, visibility,
    segmentation, headline (flat), ci (headline label -> [lo, hi]) and, with
    ``breakdowns``: by_source, by_group, by_keypoint, by_n_visible.
    """
    arr = stack_rows(rows)
    core = _core_summary(arr)
    core["headline"] = headline(core)
    n = len(rows)
    if n_boot and n:
        core["ci"] = bootstrap_ci(
            n, lambda idx: headline(_core_summary(_subset(arr, idx), curve=False)),
            n_boot=n_boot, seed=seed, level=ci_level,
        )
        core["ci_meta"] = {"method": "percentile bootstrap over images", "n_boot": n_boot,
                           "seed": seed, "level": ci_level}
    if not breakdowns:
        return core

    def by_field(field):
        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault(str(r.get(field) or "unknown"), []).append(i)
        res = {}
        for g, idx in sorted(groups.items()):
            sub = _core_summary(_subset(arr, np.array(idx)), curve=False)
            res[g] = dict(n_images=len(idx), **headline(sub))
        return res

    core["by_source"] = by_field("source")
    core["by_group"] = by_field("group")

    E, Es = arr["err_px"], arr["err_sym_px"]
    Eref = E * arr["ref_scale"][:, None]
    Esref = Es * arr["ref_scale"][:, None]
    cd = arr["court_diag"]
    pv = arr["pred_vis"] > VIS_THRESHOLD
    gv = ~np.isnan(E)
    per_kp = []
    for k in range(NUM_KEYPOINTS):
        v = gv[:, k]
        tp = int((pv[:, k] & v).sum())
        nv = v & np.isfinite(cd) & (cd > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            en = E[:, k] / cd
            esn = Es[:, k] / cd
        per_kp.append({
            "index": k,
            "n": int(v.sum()),
            "err_ref_mean": _nanmean(Eref[v, k]),
            "err_ref_median": _nanmedian(Eref[v, k]),
            "pck@10px": _frac(v & (np.nan_to_num(Eref[:, k], nan=np.inf) <= 10), v),
            "pck_norm@0.02": _frac(nv & (np.nan_to_num(en, nan=np.inf) <= 0.02), nv),
            "sym_pck@10px": _frac(v & (np.nan_to_num(Esref[:, k], nan=np.inf) <= 10), v),
            "sym_pck_norm@0.02": _frac(nv & (np.nan_to_num(esn, nan=np.inf) <= 0.02), nv),
            "vis_precision": tp / int(pv[:, k].sum()) if pv[:, k].any() else float("nan"),
            "vis_recall": tp / int(v.sum()) if v.any() else float("nan"),
        })
    core["by_keypoint"] = per_kp

    n_vis = (~np.isnan(E)).sum(axis=1)
    buckets = {"0": {"n_images": int((n_vis == 0).sum())}}
    for lo, hi in N_VISIBLE_BUCKETS:
        idx = np.flatnonzero((n_vis >= lo) & (n_vis <= hi))
        if idx.size:
            sub = _core_summary(_subset(arr, idx), curve=False)
            buckets[_bucket_label(lo, hi)] = dict(n_images=int(idx.size), **headline(sub))
        else:
            buckets[_bucket_label(lo, hi)] = {"n_images": 0}
    core["by_n_visible"] = buckets
    return core


# ---------------------------------------------------------------------------
# Legacy API (kept importable; NOT the paper metrics)
# ---------------------------------------------------------------------------

def pck_at_k(pred_kps, gt_kps, visibility, k, image_size=640):
    """LEGACY: PCK at k pixels of a square ``image_size`` space, per image.

    Kept for backward compatibility. Returns (per_keypoint (30,) bool,
    accuracy); an image with no visible point returns accuracy 0.0, which
    biases per-image averages. Use keypoint_errors + summarize_errors.
    """
    pred_px = _np(pred_kps) * image_size
    gt_px = _np(gt_kps) * image_size
    vis = _np(visibility)

    dists = np.linalg.norm(pred_px - gt_px, axis=1)
    visible_mask = vis > 0.5
    correct = (dists < k) & visible_mask

    n_visible = visible_mask.sum()
    if n_visible == 0:
        return np.zeros(NUM_KEYPOINTS, dtype=bool), 0.0
    return correct, float(correct.sum() / n_visible)


def mean_reprojection_error(pred_kps, gt_kps, visibility, image_w, image_h):
    """LEGACY: homography reprojection error using GT visibility as an oracle.

    Fits a DLT homography on predicted points selected by GT visibility and
    measures the reprojection error of the template at GT-visible points.
    Returns None if fewer than 4 points or the fit fails. Superseded by
    ``fit_pred_homography`` (prediction-only RANSAC) in ``evaluate_image``.
    """
    pred = _np(pred_kps)
    gt = _np(gt_kps)
    visible_mask = _np(visibility) > 0.5

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
    """LEGACY: IoU of the quadrilateral of the 4 labelled outer corners.

    Returns 0.0 unless all 4 corners are visible on both sides, or a quad is
    not convex. Superseded by ``court_iou_homography``.
    """
    pred = _np(pred_kps)
    gt = _np(gt_kps)
    p_vis = _np(pred_vis)
    g_vis = _np(gt_vis)

    for vis in (p_vis, g_vis):
        if not all(vis[i] > 0.5 for i in CORNER_INDICES):
            return 0.0

    pred_corners = pred[CORNER_INDICES]
    gt_corners = gt[CORNER_INDICES]
    for quad in (pred_corners, gt_corners):
        if polygon_area(quad) <= 0 or not _is_convex(quad):
            return 0.0
    iou = convex_polygon_iou(pred_corners, gt_corners)
    return 0.0 if np.isnan(iou) else iou


def _is_convex(poly):
    p = np.asarray(poly, dtype=np.float64)
    n = len(p)
    signs = []
    for i in range(n):
        a, b, c = p[i], p[(i + 1) % n], p[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-12:
            signs.append(cross > 0)
    return len(set(signs)) <= 1

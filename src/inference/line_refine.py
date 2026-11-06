"""Line-aware homography refinement (C1) and multi-hypothesis verification (C2).

Both use the model's court-line segmentation, which is otherwise unused at
inference. Everything runs in the model's square canvas (the space of
``seg_mask``) in continuous pixel coordinates: canvas pixel (i, j) has centre
(j + 0.5, i + 0.5), which matches normalized canvas keypoints times the canvas
size. ``canvas_affine`` maps original-image pixels into this space, so a
template->image homography H becomes ``A @ H`` on the canvas. Distances and
thresholds below are therefore canvas pixels (the 640-px reference scale of
the evaluation for the default model).

C1, ``refine_homography``: starting from the MAGSAC fit, minimise over the 8
DoF of H

    (1 - alpha) / n_kp   * sum_k  rho(||H(T_k) - p_k||)        k: MAGSAC inliers
    +     alpha / n_line * sum_i  rho(d(q_i, H(L(i))))          i: seg line pixels

where p_k are the decoded keypoints, q_i the line pixels of ``seg_mask``,
L(i) the template court line assigned to q_i and d the point-line distance in
the image. Each term is averaged so ``alpha`` sets their balance regardless of
how many line pixels there are (PnLCalib uses alpha = 0.5 on a comparable
cost). rho is Huber or Cauchy with scale ``delta``; the minimiser is
Levenberg-Marquardt with iteratively reweighted residuals. The assignment
L(i) (nearest projected painted segment) is redone ``iterations`` times;
pixels farther than the gate from every projected segment (other courts,
players, false positives) are dropped. The result is kept only if it still
passes the sanity checks of the MAGSAC fit (handedness, convex corners) and
does not move the inlier reprojections far from the keypoints.

C2, ``select_homography``: MAGSAC on several keypoint subsets (by default
all candidates and the more confident half) and inlier thresholds,
each fit scored by ``line_agreement``: the F1 of seg line pixels lying within
``tol`` of the projected template lines and projected template line pixels
lying within ``tol`` of seg line pixels. The best-scoring fit is kept (the
default fit unless another beats it by ``margin``). A score below a threshold
lets the predictor reject the image, which makes completeness a measured
quantity rather than "MAGSAC returned something".
"""

import cv2
import numpy as np

from src.court_geometry import (
    CORNER_INDICES, COURT_KEYPOINTS_TEMPLATE, COURT_LENGTH, COURT_WIDTH_DOUBLES, get_court_lines,
    has_homography_support, validate_quadrilateral,
)

# The centre line across the net (K12-K17) is virtual: it is not painted and
# the segmentation targets exclude it.
NET_SEGMENT = (12, 17)
PAINTED_LINES = np.array(
    [s for s in get_court_lines() if tuple(sorted(s)) != NET_SEGMENT], dtype=np.int64,
)
LOSSES = ("huber", "cauchy", "linear")
SUBSETS = ("all", "confident", "rows0-2", "rows3-5")

# Template normalisation for a well-conditioned parameterisation of H.
_T_CENTRE = np.array([COURT_LENGTH / 2, COURT_WIDTH_DOUBLES / 2])
_T_SCALE = COURT_LENGTH / 2
NUM_KP = len(COURT_KEYPOINTS_TEMPLATE)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def canvas_affine(image_hw, size, mode="letterbox"):
    """3x3 affine mapping original-image pixels to canvas pixels.

    ``mode`` "letterbox" reproduces ``src.preprocessing.letterbox`` (and
    ``keypoints_to_letterbox``); "squash" is a plain resize to size x size.
    """
    h, w = image_hw
    if mode == "squash":
        return np.diag([size / w, size / h, 1.0])
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    return np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y], [0.0, 0.0, 1.0]])


def project(H, pts):
    """Project (N, 2) points; returns (pixels (N, 2), w (N,)) with w > 0 in front."""
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = ph[:, 2]
    safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return ph[:, :2] / safe[:, None], w


def handedness(H, template_pts):
    """Sign of the Jacobian determinant of H at the centroid of ``template_pts``
    (+1 for a real camera, -1 for a mirrored court), 0 if undefined."""
    c = np.asarray(template_pts, dtype=np.float64).mean(axis=0)
    w_c = H[2, 0] * c[0] + H[2, 1] * c[1] + H[2, 2]
    if abs(w_c) < 1e-12:
        return 0
    p = project(H, c[None])[0][0]
    J = (H[:2, :2] - np.outer(p, H[2, :2])) / w_c
    det = np.linalg.det(J)
    return 0 if abs(det) < 1e-12 else (1 if det > 0 else -1)


def _segments(Hc):
    """Projected painted segments with both endpoints in front: (a, b, keep)."""
    proj, depth = project(Hc, COURT_KEYPOINTS_TEMPLATE)
    keep = (depth[PAINTED_LINES[:, 0]] > 0) & (depth[PAINTED_LINES[:, 1]] > 0)
    return proj[PAINTED_LINES[:, 0]], proj[PAINTED_LINES[:, 1]], keep


def _segment_distances(pts, a, b):
    """(N, L) distances from points to finite segments a_l-b_l."""
    dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    inv_len2 = 1.0 / np.maximum(dx * dx + dy * dy, 1e-12)
    rx = pts[:, :1] - a[:, 0]                        # (N, L)
    ry = pts[:, 1:] - a[:, 1]
    t = np.clip((rx * dx + ry * dy) * inv_len2, 0.0, 1.0)
    ex, ey = rx - t * dx, ry - t * dy
    return np.sqrt(ex * ex + ey * ey)


def assign_lines(Hc, pts, gate):
    """Nearest projected painted segment for each point, -1 if none is within
    ``gate`` pixels. Returns (index into PAINTED_LINES (N,), distance (N,))."""
    a, b, keep = _segments(Hc)
    if not keep.any() or len(pts) == 0:
        return np.full(len(pts), -1, dtype=np.int64), np.full(len(pts), np.inf)
    dist = _segment_distances(pts, a[keep], b[keep])
    j = dist.argmin(axis=1)
    best = dist[np.arange(len(pts)), j]
    idx = np.flatnonzero(keep)[j]
    idx[best > gate] = -1
    return idx, best


def line_pixels(seg, threshold=0.5, valid=None, max_points=3000, seed=0):
    """Canvas coordinates (N, 2) of segmentation pixels >= threshold.

    ``valid`` (same shape, bool) restricts to the letterbox content region.
    At most ``max_points`` pixels are kept (deterministic subsample).
    """
    m = np.asarray(seg) >= threshold
    if valid is not None:
        m &= np.asarray(valid, dtype=bool)
    ys, xs = np.nonzero(m)
    pts = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)
    if max_points and len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[np.sort(rng.choice(len(pts), max_points, replace=False))]
    return pts


def content_region(A, image_hw, size):
    """(size, size) bool mask of the canvas pixels that hold image content."""
    h, w = image_hw
    x0, y0 = int(round(A[0, 2])), int(round(A[1, 2]))
    x1, y1 = int(round(A[0, 2] + A[0, 0] * w)), int(round(A[1, 2] + A[1, 1] * h))
    valid = np.zeros((size, size), dtype=bool)
    valid[max(y0, 0):min(y1, size), max(x0, 0):min(x1, size)] = True
    return valid


# ----------------------------------------------------------------------
# Verification score (C2)
# ----------------------------------------------------------------------

def render_model_lines(Hc, size, thickness=1):
    """(size, size) uint8 {0,1} mask of the painted court lines under Hc
    (canvas homography). Segments with an endpoint behind the camera are
    skipped; the rest are clipped to the canvas before drawing."""
    mask = np.zeros((size, size), dtype=np.uint8)
    a, b, keep = _segments(Hc)
    shift = 4
    rect = (-2 * size, -2 * size, 5 * size, 5 * size)
    for pa, pb in zip(a[keep], b[keep]):
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        # Clip in float first: far-away endpoints would overflow cv2's ints.
        ok, pa, pb = _clip_segment(pa, pb, rect)
        if not ok:
            continue
        p1 = tuple(int(v) for v in np.round((pa - 0.5) * (1 << shift)))
        p2 = tuple(int(v) for v in np.round((pb - 0.5) * (1 << shift)))
        cv2.line(mask, p1, p2, 1, int(thickness), cv2.LINE_8, shift)
    return mask


def _clip_segment(pa, pb, rect):
    """Liang-Barsky clip of segment pa-pb to rect (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = rect
    d = pb - pa
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], pa[0] - x0), (d[0], x1 - pa[0]), (-d[1], pa[1] - y0), (d[1], y1 - pa[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False, pa, pb
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False, pa, pb
    return True, pa + t0 * d, pa + t1 * d


def line_agreement(Hc, seg, threshold=0.5, valid=None, tol=3.0):
    """Agreement between the projected template lines and the segmentation.

    precision: share of seg line pixels within ``tol`` px of a projected line
    recall:    share of projected line pixels (1-px lines, inside ``valid``)
               within ``tol`` px of a seg line pixel
    f1:        their harmonic mean (0 when either set is empty)

    Returns a dict with precision, recall, f1, n_seg, n_model.
    """
    seg = np.asarray(seg)
    size = seg.shape[0]
    seg_bin = seg >= threshold
    model = render_model_lines(Hc, size, 1).astype(bool)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        seg_bin &= valid
        model &= valid
    n_seg, n_model = int(seg_bin.sum()), int(model.sum())
    out = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_seg": n_seg, "n_model": n_model}
    if n_seg == 0 or n_model == 0:
        return out
    # distanceTransform gives the distance to the nearest zero pixel.
    dt_model = cv2.distanceTransform((~model).astype(np.uint8), cv2.DIST_L2, 5)
    dt_seg = cv2.distanceTransform((~seg_bin).astype(np.uint8), cv2.DIST_L2, 5)
    prec = float((dt_model[seg_bin] <= tol).mean())
    rec = float((dt_seg[model] <= tol).mean())
    out["precision"], out["recall"] = prec, rec
    out["f1"] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return out


# ----------------------------------------------------------------------
# Refinement (C1)
# ----------------------------------------------------------------------

def _norm_mats(size):
    """N_img (canvas px -> ~[-1, 1]) and N_tpl (metres -> ~[-1, 1])."""
    h = size / 2.0
    N_img = np.array([[1 / h, 0, -1.0], [0, 1 / h, -1.0], [0, 0, 1]])
    N_tpl = np.array([[1 / _T_SCALE, 0, -_T_CENTRE[0] / _T_SCALE],
                      [0, 1 / _T_SCALE, -_T_CENTRE[1] / _T_SCALE], [0, 0, 1]])
    return N_img, N_tpl


def _robust(r, loss, delta):
    """(rho(r), IRLS weight rho'(r)/r) elementwise."""
    a = np.abs(r)
    if loss == "huber":
        rho = np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))
        wgt = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))
    elif loss == "cauchy":
        z = (r / delta) ** 2
        rho = 0.5 * delta ** 2 * np.log1p(z)
        wgt = 1.0 / (1.0 + z)
    else:
        rho, wgt = 0.5 * r ** 2, np.ones_like(r)
    return rho, wgt


def _project_jac(g, s22, Xn, half):
    """Canvas projections of normalized template points under the parameters
    g (G with G[2, 2] = s22) and their Jacobian d p / d g: (n, 2), (n, 2, 8)."""
    G = np.append(g, s22).reshape(3, 3)
    u = Xn @ G.T
    inv = 1.0 / u[:, 2]
    pi = u[:, :2] * inv[:, None]
    J = np.zeros((len(Xn), 2, 8))
    J[:, 0, 0:3] = Xn * inv[:, None]
    J[:, 1, 3:6] = Xn * inv[:, None]
    J[:, 0, 6:8] = -pi[:, :1] * Xn[:, :2] * inv[:, None]
    J[:, 1, 6:8] = -pi[:, 1:] * Xn[:, :2] * inv[:, None]
    return (pi + 1.0) * half, J * half


def _residuals_jac(g, s22, Xn, half, kp_rows, kp_px, line_pts, ia, ib):
    """Stacked residuals [keypoint x/y (2 n_kp), point-line distance (n_line)]
    and their Jacobian (n_res, 8).

    The line residual is the signed distance of q from the line through the
    projected segment ends a, b: r = (d x e) / |d|, d = b - a, e = q - a.
    """
    P, J = _project_jac(g, s22, Xn, half)
    r_kp = (P[kp_rows] - kp_px).ravel()
    J_kp = J[kp_rows].reshape(-1, 8)
    if len(line_pts) == 0:
        return r_kp, J_kp
    a, b = P[ia], P[ib]
    d, e = b - a, line_pts - a
    L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    cr = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0]
    r_line = cr / L
    # d(cr)/da = (d_y - e_y, e_x - d_x), d(cr)/db = (e_y, -e_x); dL/db = -dL/da = d / L
    k = (cr / L ** 3)[:, None]
    dra = np.stack([d[:, 1] - e[:, 1], e[:, 0] - d[:, 0]], axis=1) / L[:, None] + k * d
    drb = np.stack([e[:, 1], -e[:, 0]], axis=1) / L[:, None] - k * d
    J_line = np.einsum("ni,nij->nj", dra, J[ia]) + np.einsum("ni,nij->nj", drb, J[ib])
    return np.concatenate([r_kp, r_line]), np.vstack([J_kp, J_line])


def refine_homography(H, keypoints, inliers, image_hw, seg, A, alpha=0.5, loss="huber", delta=2.0,
                      gates=(12.0, 6.0, 4.0), lm_iterations=15, seg_threshold=0.5,
                      max_line_points=3000, max_shift=None, valid=None):
    """C1: refine a template->image homography with keypoints and seg lines.

    Args:
        H: (3, 3) template (metres) -> original pixels, e.g. the MAGSAC fit
        keypoints: (30, 2) normalized original-image keypoints
        inliers: (30,) bool, keypoints used in the keypoint term
        image_hw: (h, w) of the original image
        seg: (S, S) seg probabilities on the canvas (``CourtDetection.seg_mask``)
        A: canvas affine (``canvas_affine``)
        alpha: weight of the line term (0 = keypoints only, 1 = lines only)
        loss: "huber", "cauchy" or "linear"
        delta: robust-loss scale in canvas px
        gates: line-assignment gate (canvas px) per assignment round; its
            length is the number of rounds
        lm_iterations: Levenberg-Marquardt steps per round
        max_shift: revert if the median inlier reprojection moves more than
            this (canvas px) away from the keypoints; default 3 * gates[-1]
        valid: optional (S, S) bool content region of the canvas

    Returns:
        (H_refined, info). ``info["status"]`` is "ok", or "reverted: <why>"
        (H is then returned unchanged) or "skipped: <why>".
    """
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
    seg = np.asarray(seg, dtype=np.float32)
    size = seg.shape[0]
    inliers = np.asarray(inliers, dtype=bool)
    info = {"status": "skipped", "n_kp": int(inliers.sum()), "n_line": 0}
    if inliers.sum() < 4:
        info["status"] = "skipped: too few inliers"
        return H, info

    Hc0 = A @ H
    kp_tpl = COURT_KEYPOINTS_TEMPLATE[inliers]
    h, w = image_hw
    kp_px, _ = project(A, np.asarray(keypoints, dtype=np.float64)[inliers] * np.array([w, h]))
    pts = line_pixels(seg, seg_threshold, valid, max_line_points) if alpha > 0 else np.zeros((0, 2))

    N_img, N_tpl = _norm_mats(size)
    G = N_img @ Hc0 @ np.linalg.inv(N_tpl)
    if abs(G[2, 2]) < 1e-6 * np.abs(G).max():
        info["status"] = "skipped: ill-conditioned"
        return H, info
    # Fix G[2, 2] = +-1; keeping its sign keeps "in front" = positive w.
    s22 = np.sign(G[2, 2])
    g = (G / abs(G[2, 2])).ravel()[:8]
    N_img_inv = np.linalg.inv(N_img)
    half = size / 2.0
    Xn = np.hstack([COURT_KEYPOINTS_TEMPLATE, np.ones((NUM_KP, 1))]) @ N_tpl.T
    kp_rows = np.flatnonzero(inliers)

    def to_Hc(params):
        return N_img_inv @ np.append(params, s22).reshape(3, 3) @ N_tpl

    n_kp = len(kp_tpl)
    h_sign = handedness(Hc0, kp_tpl)
    rounds = []
    for gate in gates:
        line_idx, _ = assign_lines(to_Hc(g), pts, gate)
        sel = line_idx >= 0
        n_line = int(sel.sum())
        if alpha >= 1 and n_line < 8:
            info["status"] = "skipped: too few line pixels"
            return H, info
        q = pts[sel]
        ia, ib = PAINTED_LINES[line_idx[sel], 0], PAINTED_LINES[line_idx[sel], 1]
        # Per-residual term weights: each term is a mean, balanced by alpha.
        c = np.concatenate([
            np.full(2 * n_kp, (1 - alpha) / n_kp),
            np.full(n_line, alpha / n_line if n_line else 0.0),
        ])

        def cost(params, q=q, ia=ia, ib=ib, c=c):
            r, J = _residuals_jac(params, s22, Xn, half, kp_rows, kp_px, q, ia, ib)
            rho, wgt = _robust(r, loss, delta)
            return float((c * rho).sum()), r, J, wgt

        lam = 1e-3
        f, r, J, wgt = cost(g)
        for _ in range(lm_iterations):
            f_prev = f
            JtW = J.T * (c * wgt)
            Hm = JtW @ J
            grad = JtW @ r
            if not np.all(np.isfinite(Hm)):
                break
            improved = False
            for _ in range(8):
                step = np.linalg.solve(Hm + lam * np.diag(np.diag(Hm) + 1e-12), -grad)
                f_new, r_new, J_new, wgt_new = cost(g + step)
                if np.isfinite(f_new) and f_new < f:
                    g, f, r, J, wgt = g + step, f_new, r_new, J_new, wgt_new
                    lam = max(lam / 3, 1e-9)
                    improved = True
                    break
                lam *= 4
            if not improved or f_prev - f < 1e-6 * f_prev:
                break
        rounds.append({"gate": float(gate), "n_line": n_line, "cost": f})
    info["rounds"] = rounds
    info["n_line"] = rounds[-1]["n_line"] if rounds else 0

    Hc = to_Hc(g)
    reason = _sanity(Hc, Hc0, kp_tpl, kp_px, max_shift if max_shift is not None else 3 * gates[-1],
                     h_sign)
    if reason:
        info["status"] = f"reverted: {reason}"
        return H, info
    H_new = np.linalg.inv(A) @ Hc
    H_new *= np.linalg.norm(H) / np.linalg.norm(H_new)
    info["status"] = "ok"
    return H_new, info


def _sanity(Hc, Hc0, kp_tpl, kp_px, max_shift, h_sign):
    if not np.all(np.isfinite(Hc)):
        return "non-finite"
    _, depth = project(Hc, kp_tpl)
    if np.any(depth * np.sign(project(Hc0, kp_tpl)[1]) <= 0):
        return "inliers behind camera"
    if handedness(Hc, kp_tpl) != h_sign:
        return "handedness flipped"
    corners, cdepth = project(Hc, COURT_KEYPOINTS_TEMPLATE[CORNER_INDICES])
    if np.all(cdepth > 0) and not validate_quadrilateral(corners):
        return "non_convex"
    shift = np.median(np.linalg.norm(project(Hc, kp_tpl)[0] - kp_px, axis=1))
    if shift > max_shift:
        return f"keypoints moved {shift:.1f}px"
    return None


# ----------------------------------------------------------------------
# Multi-hypothesis fit (C2)
# ----------------------------------------------------------------------

def hypothesis_subsets(candidates, conf, names=SUBSETS):
    """Named keypoint subsets to fit, among SUBSETS:
        all        every candidate
        confident  candidates with decoder confidence >= their median
                   (needs ``conf`` and >= 8 candidates)
        rows0-2 / rows3-5   one half of the court
    Subsets that cannot determine a homography are left out."""
    candidates = np.asarray(candidates, dtype=bool)
    rows = np.arange(len(candidates)) // 5
    subsets = {}
    for name in names:
        if name == "all":
            subsets[name] = candidates
        elif name == "confident":
            if conf is None or candidates.sum() < 8:
                continue
            c = np.asarray(conf, dtype=np.float64)
            subsets[name] = candidates & (c >= np.median(c[candidates]))
        elif name == "rows0-2":
            subsets[name] = candidates & (rows <= 2)
        elif name == "rows3-5":
            subsets[name] = candidates & (rows >= 3)
        else:
            raise ValueError(f"unknown subset {name!r}, expected one of {SUBSETS}")
    return {k: v for k, v in subsets.items() if has_homography_support(v)}


def select_homography(keypoints, candidates, conf, image_hw, seg, A, fit_fn,
                      thresholds=(0.005, 0.01, 0.02), default_threshold=0.01, min_inliers=6,
                      subsets=("all", "confident"), seg_threshold=0.5, tol=3.0, margin=0.05,
                      valid=None):
    """C2: fit H on several subsets / thresholds and keep the one whose
    projected template best agrees with the segmentation.

    ``fit_fn(keypoints, mask, image_hw, threshold_frac, min_inliers)`` is the
    single-fit routine (``predict.fit_court_homography``), returning
    (H, inlier_mask, status). The default hypothesis is all candidates at
    ``default_threshold`` (the single-fit result); another hypothesis replaces
    it only if its line F1 is more than ``margin`` higher. Inliers of a subset
    fit are re-evaluated on all candidates (reprojection within the fit
    threshold) so every hypothesis reports a full inlier mask. Near-duplicate
    fits (outer corners within 1 px) are scored once.

    Returns (H, inliers, status, score_dict, hypotheses). ``hypotheses`` is a
    list of (name, threshold, f1, n_inliers); H is None when nothing fits.
    """
    h, w = image_hw
    seg = np.asarray(seg)
    masks = hypothesis_subsets(candidates, conf, subsets)
    if "all" not in masks:
        masks = {"all": np.asarray(candidates, dtype=bool), **masks}
    kp_px = np.asarray(keypoints, dtype=np.float64) * np.array([w, h])
    cands = np.asarray(candidates, dtype=bool)
    # The default hypothesis goes first so it is never dropped as a duplicate.
    jobs = [("all", default_threshold)] + [
        (name, thr) for name in masks for thr in sorted(set(thresholds))
        if (name, thr) != ("all", default_threshold)
    ]
    hyps, seen = [], []
    default_status = None
    for name, thr in jobs:
        H, inl, status = fit_fn(keypoints, masks[name], image_hw, thr, min_inliers)
        if default_status is None:
            default_status = status
        if H is None:
            continue
        corners = project(H, COURT_KEYPOINTS_TEMPLATE[CORNER_INDICES])[0]
        if any(np.abs(corners - s).max() < 1.0 for s in seen):
            continue
        seen.append(corners)
        if name != "all":
            proj, depth = project(H, COURT_KEYPOINTS_TEMPLATE)
            err = np.linalg.norm(proj - kp_px, axis=1)
            inl = cands & (depth > 0) & (err <= thr * max(h, w))
            if not has_homography_support(inl):
                continue
        score = line_agreement(A @ H, seg, seg_threshold, valid, tol)
        hyps.append({"name": name, "threshold": thr, "H": H, "inliers": inl, "score": score,
                     "default": (name, thr) == ("all", default_threshold)})
    if not hyps:
        return None, None, default_status, None, []
    best = max(hyps, key=lambda d: (d["score"]["f1"], int(d["inliers"].sum())))
    default = hyps[0] if hyps[0]["default"] else None
    if default is not None and best["score"]["f1"] - default["score"]["f1"] <= margin:
        best = default
    summary = [(d["name"], d["threshold"], d["score"]["f1"], int(d["inliers"].sum())) for d in hyps]
    status = "ok" if best["default"] else f"ok ({best['name']}@{best['threshold']:g})"
    return best["H"], best["inliers"], status, best["score"], summary


__all__ = [
    "LOSSES", "PAINTED_LINES", "SUBSETS", "assign_lines", "canvas_affine", "content_region", "line_agreement",
    "line_pixels", "refine_homography", "render_model_lines", "select_homography",
]

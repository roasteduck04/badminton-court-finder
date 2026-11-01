"""Classical (non-learned) court-detection baseline.

Farin et al., *Robust Camera Calibration for Sport Videos using Court Models*
(SPIE 5307, 2004) with the MonoTrack line-family split (Liu & Wang, CVPRW
2022). Parameters follow the open implementation
github.com/gchlebus/tennis-court-detection; see docs/improvement-plan.md,
"B2 recipe". Nothing here is trained: the same code and parameters run on
every image, and the result is scored with exactly the same metric code as
the model (``src.evaluation.evaluate.run_evaluation``).

Pipeline (all pixel parameters are at the working resolution: the image is
resized so its long side is ``work_long_side`` = 1280 px, which is how the
line width tau and the other pixel constants scale with the resolution):

1. Line pixels: response >= 80 and more than 20 brighter than both
   neighbours tau px away, horizontally or vertically. The response is the
   luminance (``--line-mode lum``), or for yellow / coloured lines the Lab
   chroma distance to the floor colour (``chroma``; floor colour = mode of the
   a*b* histogram of the lower two thirds of the image), or both. Line width
   is unknown and varies with distance, so tau runs over (3, 6, 12, 24) px
   and the masks are OR-ed.
2. Texture filter: keep pixels whose structure-tensor eigenvalue ratio
   lambda1 / lambda2 > 4 (Sobel 3, 41 x 41 box window).
3. Line candidates: standard Hough transform, non-maximum suppression in
   (angle, offset), 5 rounds of total-least-squares refit on line pixels
   within 8 px whose structure-tensor orientation runs along the line
   (+-15 deg; drops Hough lines that only join crossings of other lines),
   duplicate removal (lines sharing > 50% of their pixels).
4. Two line families: MonoTrack's max-weight bipartite partition (max cut,
   exhaustive over the <= 18 strongest lines) with edge weight
   (|theta_ij - pi/2| + eps)^-2, theta_ij the angle between lines i and j.
5. Hypotheses: every pair of image lines of family 1 x every pair of family 2
   (each family ordered across its lines) x every pair of model transverse
   lines (6, the keypoint rows) x every pair of model longitudinal lines (5,
   the keypoint columns), both family-to-direction assignments. H comes from
   the 4 intersections (unit square -> quad, closed form). Only one ordering
   per family is needed: reversing it is a court symmetry, which the final
   canonicalisation undoes anyway.
6. Score: walk every projected painted model segment in the image: +1 per
   line pixel, -0.5 per non-line pixel (pixels outside the image count 0 so
   partial views are not punished for what the camera cannot see). A
   vectorised coarse score (``coarse_samples`` points per segment, weighted
   by their image spacing; a 4-sample pass first keeps the best 4000) ranks
   the hypotheses; the best ``n_fine`` are rescored by the exact pixel walk
   (each line pixel counted once). Degenerate H are rejected: the 4 used
   intersections must form a convex quad in front of the camera, and every
   projected outer court side (with both ends in front) must be >= 30 px.
7. Refinement: Levenberg-Marquardt on the 8 homography parameters,
   minimising the (Huber-weighted) distances of line pixels to the projected
   model line of their nearest painted segment (within 8 px), reassigning
   pixels each round. The refined H is kept unless its walk score drops by
   more than 5% (the score is quantised to whole pixels). The 30 keypoints
   are projected and relabelled with ``court_geometry.canonicalize``.
8. Failure: if the best score / work_long_side < ``score_threshold`` (2.3,
   chosen on the val split; see docs/classical-baseline.md), or there are
   not 2 lines per family, or no valid hypothesis, no prediction is made: keypoints NaN (every GT-visible point counts as a miss), visibility
   0, homography None.

Predicted visibility = projected keypoint in front of the camera and inside
the image.

Usage:
  python scripts/classical_baseline.py --manifest data/colab_data_v2/manifest.json \
      --eval-split test_real --out results/classical_test_real.json [--debug-dir DIR]
  python scripts/classical_baseline.py --eval-annotations A --eval-images I --out X.json
  python scripts/classical_baseline.py --image photo.jpg --debug-dir out/
"""

import argparse
import dataclasses
import itertools
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.court_geometry import (  # noqa: E402
    COURT_KEYPOINTS_TEMPLATE,
    COURT_LENGTH,
    COURT_WIDTH_DOUBLES,
    NUM_KEYPOINTS,
    canonicalize,
    get_court_lines,
    permute_keypoints,
)
from src.evaluation.evaluate import (  # noqa: E402
    load_items,
    print_summary,
    run_evaluation,
    write_outputs,
)
from src.evaluation.metrics import NET_SEGMENT  # noqa: E402


@dataclasses.dataclass
class ClassicalParams:
    work_long_side: int = 1280
    line_mode: str = "lum"            # lum | chroma | both
    lum_threshold: float = 80.0
    chroma_threshold: float = 30.0    # Lab a*b* distance to the floor colour
    diff_threshold: float = 20.0
    taus: tuple = (3, 6, 12, 24)
    texture_window: int = 41
    eig_ratio: float = 4.0
    hough_votes: int = 64
    hough_max_lines: int = 60
    nms_angle_deg: float = 2.0
    nms_dist_px: float = 10.0
    refit_dist_px: float = 8.0
    refit_iters: int = 5
    orientation_tol_deg: float = 15.0
    min_line_span_px: float = 40.0
    max_lines: int = 18
    max_per_family: int = 8
    partition_eps: float = 0.05
    quad_margin: float = 1.5          # intersections within this many image sizes
    min_side_px: float = 30.0
    prefilter_samples: int = 4       # cheap first pass over all hypotheses ...
    n_prefilter: int = 4000           # ... keeps this many for the coarse score
    coarse_samples: int = 12
    n_fine: int = 100
    n_refine: int = 3
    refine_rounds: int = 3
    refine_dist_px: float = 8.0
    huber_px: float = 2.0
    refine_max_pixels: int = 15000
    refine_score_tol: float = 0.05    # keep a refined H unless its score drops > 5%
    score_threshold: float = 2.3      # walk score / work_long_side; chosen on val (see docs)


# ---------------------------------------------------------------------------
# Court model
# ---------------------------------------------------------------------------

ROW_X = np.unique(COURT_KEYPOINTS_TEMPLATE[:, 0])   # 6 transverse lines x = const
COL_Y = np.unique(COURT_KEYPOINTS_TEMPLATE[:, 1])   # 5 longitudinal lines y = const
# 11 model lines a*x + b*y + c = 0 in template metres: rows first, then columns.
MODEL_LINES = np.array([[1.0, 0.0, -x] for x in ROW_X] + [[0.0, 1.0, -y] for y in COL_Y])
PAINTED = [s for s in get_court_lines() if tuple(s) != NET_SEGMENT]
SEG_P = COURT_KEYPOINTS_TEMPLATE[[a for a, _ in PAINTED]]
SEG_Q = COURT_KEYPOINTS_TEMPLATE[[b for _, b in PAINTED]]
SEG_LINE = np.array([
    int(np.flatnonzero(ROW_X == p[0])[0]) if p[0] == q[0]
    else len(ROW_X) + int(np.flatnonzero(COL_Y == p[1])[0])
    for p, q in zip(SEG_P, SEG_Q)
])
COURT_CORNERS = np.array([[0.0, 0.0], [COURT_LENGTH, 0.0],
                          [COURT_LENGTH, COURT_WIDTH_DOUBLES], [0.0, COURT_WIDTH_DOUBLES]])

# Template symmetries as template->template maps, matching SYMMETRY_PERMS:
# relabelling with perm p turns H into H @ SYM_MATRIX[p].
SYM_MATRIX = {
    "identity": np.eye(3),
    "col_flip": np.array([[1.0, 0, 0], [0, -1, COURT_WIDTH_DOUBLES], [0, 0, 1]]),
    "row_flip": np.array([[-1.0, 0, COURT_LENGTH], [0, 1, 0], [0, 0, 1]]),
    "rot180": np.array([[-1.0, 0, COURT_LENGTH], [0, -1, COURT_WIDTH_DOUBLES], [0, 0, 1]]),
}


def _hom(pts):
    pts = np.asarray(pts, dtype=np.float64)
    return np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1)


# ---------------------------------------------------------------------------
# 1-2. Line pixels
# ---------------------------------------------------------------------------

def _ridge_mask(resp, threshold, diff, taus):
    """resp >= threshold and > diff above both neighbours tau px away
    (horizontally or vertically), OR-ed over taus."""
    r = resp.astype(np.float32)
    h, w = r.shape
    out = np.zeros((h, w), dtype=bool)
    for t in taus:
        p = cv2.copyMakeBorder(r, t, t, t, t, cv2.BORDER_REPLICATE)
        left, right = p[t:t + h, :w], p[t:t + h, 2 * t:2 * t + w]
        up, down = p[:h, t:t + w], p[2 * t:2 * t + h, t:t + w]
        out |= ((r - left > diff) & (r - right > diff)) | ((r - up > diff) & (r - down > diff))
    return out & (r >= threshold)


def _structure_tensor(resp, window, ratio):
    """Structure-tensor test lambda1 > ratio * lambda2 (oriented structure),
    and the dominant gradient angle (= the local line normal) per pixel."""
    g = resp.astype(np.float32)
    ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    k = (window, window)
    a = cv2.boxFilter(ix * ix, -1, k)
    b = cv2.boxFilter(ix * iy, -1, k)
    c = cv2.boxFilter(iy * iy, -1, k)
    half_tr = 0.5 * (a + c)
    s = np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b * b, 0))
    l1, l2 = half_tr + s, np.maximum(half_tr - s, 0)
    return l1 > ratio * l2, 0.5 * np.arctan2(2 * b, a - c)


def floor_chroma(bgr):
    """Floor colour (a*, b*) in 8-bit Lab: mode of the a*b* histogram of the
    lower two thirds of the image."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    low = lab[lab.shape[0] // 3:]
    hist = cv2.calcHist([low], [1, 2], None, [32, 32], [0, 256, 0, 256])
    i, j = np.unravel_index(int(np.argmax(hist)), hist.shape)
    return np.array([8 * i + 4, 8 * j + 4], dtype=np.float32)


def detect_line_pixels(bgr, p):
    """Line-pixel mask (steps 1 and 2) and the local line-normal angle map."""
    responses = []
    if p.line_mode in ("lum", "both"):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        responses.append((gray, p.lum_threshold))
    if p.line_mode in ("chroma", "both"):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ab0 = floor_chroma(bgr)
        chroma = np.hypot(lab[..., 1] - ab0[0], lab[..., 2] - ab0[1])
        responses.append((chroma, p.chroma_threshold))
    if not responses:
        raise ValueError(f"unknown line_mode {p.line_mode!r}")
    mask = np.zeros(bgr.shape[:2], dtype=bool)
    normal = np.zeros(bgr.shape[:2], dtype=np.float32)
    for resp, thr in responses:
        keep, ang = _structure_tensor(resp, p.texture_window, p.eig_ratio)
        m = _ridge_mask(resp, thr, p.diff_threshold, p.taus) & keep & ~mask
        normal[m] = ang[m]
        mask |= m
    return mask, normal


# ---------------------------------------------------------------------------
# 3. Line candidates
# ---------------------------------------------------------------------------

def _same_line(l1, l2, center, angle_tol, dist_tol):
    """Near-identical lines (normals within angle_tol, offsets at the image
    centre within dist_tol)."""
    cos = float(l1[:2] @ l2[:2])
    if abs(cos) < np.cos(angle_tol):
        return False
    s = 1.0 if cos > 0 else -1.0
    return abs((l1[:2] @ center + l1[2]) - s * (l2[:2] @ center + l2[2])) < dist_tol


def _aligned(line, normals, angle_tol):
    """Pixels whose local line normal is within angle_tol of the line's."""
    return np.abs(np.cos(normals - np.arctan2(line[1], line[0]))) > np.cos(angle_tol)


def refit_line(line, pts, dist, iters, normals=None, angle_tol=np.deg2rad(15)):
    """Total-least-squares refit on points within ``dist`` (and, if
    ``normals`` is given, locally running along the line); returns
    (line, inlier mask) or (None, mask) if fewer than 2 inliers."""
    sel = np.zeros(len(pts), dtype=bool)
    for _ in range(iters):
        sel = np.abs(pts @ line[:2] + line[2]) < dist
        if normals is not None:
            sel &= _aligned(line, normals, angle_tol)
        if sel.sum() < 2:
            return None, sel
        q = pts[sel]
        mu = q.mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov((q - mu).T))
        n = vecs[:, 0]
        line = np.array([n[0], n[1], -n @ mu])
    sel = np.abs(pts @ line[:2] + line[2]) < dist
    if normals is not None:
        sel &= _aligned(line, normals, angle_tol)
    return line, sel


def find_lines(mask, normal, p):
    """Refined candidate lines (N, 3) with a^2 + b^2 = 1, strongest first, and
    their support (inlier pixel counts). Only pixels whose structure-tensor
    orientation runs along a line support it, so Hough lines that merely
    connect crossings of other lines lose their support."""
    h, w = mask.shape
    yx = np.argwhere(mask)
    pts = yx[:, ::-1].astype(np.float64)
    normals = normal[yx[:, 0], yx[:, 1]].astype(np.float64)
    if len(pts) < p.hough_votes:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)
    hl = cv2.HoughLines(mask.astype(np.uint8), 1, np.pi / 180, p.hough_votes)
    if hl is None:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)
    center = np.array([w / 2.0, h / 2.0])
    angle_tol = np.deg2rad(p.nms_angle_deg)
    kept = []
    for rho, theta in hl[:, 0]:  # OpenCV returns lines sorted by votes
        line = np.array([np.cos(theta), np.sin(theta), -rho])
        if any(_same_line(line, k, center, angle_tol, p.nms_dist_px) for k in kept):
            continue
        kept.append(line)
        if len(kept) >= p.hough_max_lines:
            break

    refined = []
    for line in kept:
        line, sel = refit_line(line, pts, p.refit_dist_px, p.refit_iters, normals,
                               np.deg2rad(p.orientation_tol_deg))
        if line is None or sel.sum() < p.hough_votes:
            continue
        along = pts[sel] @ np.array([-line[1], line[0]])
        if np.ptp(along) < p.min_line_span_px:
            continue
        refined.append((int(sel.sum()), line, sel))
    refined.sort(key=lambda t: -t[0])
    lines, support, sels = [], [], []
    for n, line, sel in refined:
        if any((sel & s).sum() > 0.5 * n for s in sels):
            continue
        lines.append(line)
        support.append(n)
        sels.append(sel)
    return (np.array(lines).reshape(-1, 3), np.array(support, dtype=int))


# ---------------------------------------------------------------------------
# 4. Two families (max cut)
# ---------------------------------------------------------------------------

def split_families(lines, eps=0.05):
    """MonoTrack partition: the 2-colouring maximising the summed weight
    (|theta_ij - pi/2| + eps)^-2 of edges between the two groups. Returns a
    boolean array (True = family 1); line 0 is always in family 0."""
    n = len(lines)
    if n < 2:
        return np.zeros(n, dtype=bool)
    ang = np.arctan2(lines[:, 1], lines[:, 0]) % np.pi
    d = np.abs(ang[:, None] - ang[None, :])
    d = np.minimum(d, np.pi - d)                       # angle between lines, [0, pi/2]
    W = (np.abs(d - np.pi / 2) + eps) ** -2.0
    np.fill_diagonal(W, 0.0)
    codes = np.arange(2 ** (n - 1), dtype=np.int64)
    bits = ((codes[:, None] >> np.arange(n - 1)) & 1).astype(np.float64)
    S = np.concatenate([np.ones((len(codes), 1)), 1 - 2 * bits], axis=1)   # +-1, line 0 = +1
    cut = 0.25 * (W.sum() - np.einsum("ki,ij,kj->k", S, W, S))
    return S[int(np.argmax(cut))] < 0


def order_family(lines, center):
    """Sort a family across its lines: by where each line crosses the line
    through ``center`` perpendicular to the family's mean direction."""
    dirs = np.stack([-lines[:, 1], lines[:, 0]], axis=1)
    dirs = dirs * np.sign(dirs @ dirs[0] + 1e-12)[:, None]
    m = dirs.mean(axis=0)
    m = np.array([-m[1], m[0]]) / max(np.linalg.norm(m), 1e-12)   # across the family
    denom = lines[:, :2] @ m
    t = -(lines[:, :2] @ center + lines[:, 2]) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    return lines[np.argsort(t)]


# ---------------------------------------------------------------------------
# 5. Hypotheses
# ---------------------------------------------------------------------------

def square_to_quad(q):
    """Homographies mapping the unit square (0,0),(1,0),(1,1),(0,1) to quads
    q (N, 4, 2) (Heckbert's closed form). Returns (N, 3, 3)."""
    x0, x1, x2, x3 = (q[:, i, 0] for i in range(4))
    y0, y1, y2, y3 = (q[:, i, 1] for i in range(4))
    dx1, dx2, dx3 = x1 - x2, x3 - x2, x0 - x1 + x2 - x3
    dy1, dy2, dy3 = y1 - y2, y3 - y2, y0 - y1 + y2 - y3
    with np.errstate(divide="ignore", invalid="ignore"):
        den = dx1 * dy2 - dx2 * dy1
        g = (dx3 * dy2 - dx2 * dy3) / den
        hh = (dx1 * dy3 - dx3 * dy1) / den
    H = np.empty((len(q), 3, 3))
    H[:, 0] = np.stack([x1 - x0 + g * x1, x3 - x0 + hh * x3, x0], axis=1)
    H[:, 1] = np.stack([y1 - y0 + g * y1, y3 - y0 + hh * y3, y0], axis=1)
    H[:, 2] = np.stack([g, hh, np.ones_like(g)], axis=1)
    return H


def _model_rect_inverses():
    """Inverse affine maps model rectangle -> unit square for every
    (row pair, column pair) under both family assignments. (M, 3, 3)."""
    out = []
    for swap in (False, True):
        for r1, r2 in itertools.combinations(range(len(ROW_X)), 2):
            for c1, c2 in itertools.combinations(range(len(COL_Y)), 2):
                X1, dX = ROW_X[r1], ROW_X[r2] - ROW_X[r1]
                Y1, dY = COL_Y[c1], COL_Y[c2] - COL_Y[c1]
                if not swap:   # family 1 = transverse (u along x), family 2 = longitudinal
                    A = [[1 / dX, 0, -X1 / dX], [0, 1 / dY, -Y1 / dY], [0, 0, 1]]
                else:          # family 1 = longitudinal (u along y)
                    A = [[0, 1 / dY, -Y1 / dY], [1 / dX, 0, -X1 / dX], [0, 0, 1]]
                out.append(A)
    return np.array(out, dtype=np.float64)


MODEL_RECT_INV = _model_rect_inverses()


def image_quads(fam1, fam2, shape, margin):
    """Valid quads [p_ik, p_jk, p_jl, p_il] for ordered families: finite,
    within ``margin`` image sizes of the image, convex, consistently
    oriented. Returns (N, 4, 2)."""
    h, w = shape
    P = np.cross(fam1[:, None, :], fam2[None, :, :])          # (n1, n2, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = P[..., :2] / P[..., 2:3]
    p1 = np.array(list(itertools.combinations(range(len(fam1)), 2)))
    p2 = np.array(list(itertools.combinations(range(len(fam2)), 2)))
    i, j = np.repeat(p1[:, 0], len(p2)), np.repeat(p1[:, 1], len(p2))
    k, m = np.tile(p2[:, 0], len(p1)), np.tile(p2[:, 1], len(p1))
    q = np.stack([X[i, k], X[j, k], X[j, m], X[i, m]], axis=1)
    lo = np.array([-margin * w, -margin * h])
    hi = np.array([(1 + margin) * w, (1 + margin) * h])
    ok = np.all(np.isfinite(q), axis=(1, 2)) & np.all((q > lo) & (q < hi), axis=(1, 2))
    e = np.roll(q, -1, axis=1) - q
    cross = e[:, :, 0] * np.roll(e, -1, axis=1)[:, :, 1] - e[:, :, 1] * np.roll(e, -1, axis=1)[:, :, 0]
    ok &= np.all(cross > 1e-6, axis=1) | np.all(cross < -1e-6, axis=1)
    return q[ok]


def hypotheses(fam1, fam2, shape, p):
    """All template->image homographies (N, 3, 3) passing the degeneracy
    checks; w > 0 at the 4 used intersections."""
    q = image_quads(fam1, fam2, shape, p.quad_margin)
    if len(q) == 0:
        return np.zeros((0, 3, 3))
    Q = square_to_quad(q)
    # w at the unit-square corners: 1, g+1, g+h+1, h+1 (all > 0 = in front).
    g, hh = Q[:, 2, 0], Q[:, 2, 1]
    Q = Q[(g + 1 > 0) & (hh + 1 > 0) & (g + hh + 1 > 0) & np.all(np.isfinite(Q), axis=(1, 2))]
    H = np.einsum("qab,mbc->qmac", Q, MODEL_RECT_INV).reshape(-1, 3, 3)
    return H[_not_degenerate(H, p.min_side_px)]


def _not_degenerate(H, min_side):
    """Projected outer court sides with both ends in front must be >= min_side."""
    c = np.einsum("nab,kb->nka", H, _hom(COURT_CORNERS))      # (N, 4, 3)
    front = c[..., 2] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = c[..., :2] / c[..., 2:3]
    side = np.linalg.norm(np.roll(xy, -1, axis=1) - xy, axis=2)
    both = front & np.roll(front, -1, axis=1)
    short = both & ~(side >= min_side)
    return ~short.any(axis=1) & front.any(axis=1)


# ---------------------------------------------------------------------------
# 6. Scoring
# ---------------------------------------------------------------------------

def coarse_scores(H, support, S, chunk=4000):
    """Vectorised approximation of the walk score: S model points per painted
    segment, each weighted by its image spacing."""
    t = (np.arange(S) + 0.5) / S
    M = SEG_P[:, None, :] + t[None, :, None] * (SEG_Q - SEG_P)[:, None, :]   # (nseg, S, 2)
    Mh = _hom(M.reshape(-1, 2)).T.astype(np.float32)                        # (3, K)
    h, w = support.shape
    cap = np.float32(4.0 * max(h, w) / S)
    flat = support.ravel()
    out = np.empty(len(H))
    with np.errstate(divide="ignore", invalid="ignore"):
        for s in range(0, len(H), chunk):
            P = np.matmul(H[s:s + chunk].astype(np.float32), Mh)            # (n, 3, K)
            z = P[:, 2]
            front = z > 1e-9
            inv = np.where(front, 1.0 / z, np.nan).astype(np.float32)
            x = (P[:, 0] * inv).reshape(-1, len(SEG_P), S)
            y = (P[:, 1] * inv).reshape(-1, len(SEG_P), S)
            dist = np.hypot(np.diff(x, axis=2), np.diff(y, axis=2))
            spacing = np.concatenate([dist, dist[:, :, -1:]], axis=2)
            spacing = np.where(np.isfinite(spacing), np.minimum(spacing, cap), 0)
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)                # NaN -> False
            idx = np.where(inside, y.astype(np.int64) * w + x.astype(np.int64), 0)
            val = np.where(flat[idx], 1.0, -0.5) * inside
            out[s:s + chunk] = (val * spacing).sum(axis=(1, 2))
    return out


def _clip_segment(a, b, w, h):
    """Liang-Barsky clip of segment a->b to [0, w) x [0, h); None if outside."""
    t0, t1 = 0.0, 1.0
    d = b - a
    for pv, qv in ((-d[0], a[0]), (d[0], w - 1 - a[0]), (-d[1], a[1]), (d[1], h - 1 - a[1])):
        if abs(pv) < 1e-12:
            if qv < 0:
                return None
            continue
        r = qv / pv
        if pv < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return a + t0 * d, a + t1 * d


def projected_segments(H, shape):
    """Image segments (seg index, a, b) of the painted model segments: parts
    in front of the camera, clipped to the image."""
    h, w = shape
    out = []
    wp = _hom(SEG_P) @ H[2]
    wq = _hom(SEG_Q) @ H[2]
    for k in range(len(SEG_P)):
        lo, hi = 0.0, 1.0
        eps = 1e-6 * max(abs(wp[k]), abs(wq[k]))
        if wp[k] <= eps and wq[k] <= eps:
            continue
        if wp[k] <= eps or wq[k] <= eps:
            tz = (eps - wp[k]) / (wq[k] - wp[k])
            lo, hi = (tz, 1.0) if wp[k] <= eps else (0.0, tz)
        ends = []
        for t in (lo, hi):
            m = SEG_P[k] + t * (SEG_Q[k] - SEG_P[k])
            v = H @ np.array([m[0], m[1], 1.0])
            ends.append(v[:2] / v[2])
        c = _clip_segment(ends[0], ends[1], w, h)
        if c is not None:
            out.append((k, c[0], c[1]))
    return out


def walk_stats(H, mask):
    """(distinct line pixels, non-line samples) along every projected
    painted segment; see walk_score."""
    h, w = mask.shape
    on_idx, n_off = [], 0
    for _, a, b in projected_segments(H, (h, w)):
        n = int(np.ceil(np.linalg.norm(b - a))) + 1
        t = np.linspace(0.0, 1.0, n)
        x = np.clip(np.rint(a[0] + t * (b[0] - a[0])).astype(int), 0, w - 1)
        y = np.clip(np.rint(a[1] + t * (b[1] - a[1])).astype(int), 0, h - 1)
        on = mask[y, x]
        on_idx.append(np.unique(y[on] * w + x[on]))
        n_off += int((~on).sum())
    n_on = len(np.unique(np.concatenate(on_idx))) if on_idx else 0
    return n_on, n_off


def walk_score(H, mask):
    """+1 per line pixel, -0.5 per non-line pixel along every projected
    painted segment (in-image part only). A line pixel counts once even if
    several projected segments cross it, so a degenerate H that stacks model
    lines onto one image line gains nothing from the stacking."""
    n_on, n_off = walk_stats(H, mask)
    return n_on - 0.5 * n_off


def _dedupe_key(H):
    """Hypotheses equal up to a court symmetry give the same set of projected
    keypoints."""
    v = _hom(COURT_KEYPOINTS_TEMPLATE) @ H.T
    front = v[:, 2] > 0
    xy = np.rint(v[front, :2] / v[front, 2:3] / 4.0).astype(np.int64)
    return tuple(sorted(map(tuple, xy.tolist())))


# ---------------------------------------------------------------------------
# 7. Refinement
# ---------------------------------------------------------------------------

def _point_segment_dist(pts, a, b):
    d = b - a
    L2 = max(float(d @ d), 1e-12)
    t = np.clip(((pts - a) @ d) / L2, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * d), axis=1)


def refine_homography(H, pts, shape, p):
    """LM fit of H to line pixels (see module docstring, step 7)."""
    def update(q):
        return np.array([[1 + q[0], q[1], q[2]], [q[3], 1 + q[4], q[5]], [q[6], q[7], 1.0]])

    def residuals(Hq, X, lid):
        L = MODEL_LINES @ np.linalg.inv(Hq)                  # image lines, (11, 3)
        L = L / np.linalg.norm(L[:, :2], axis=1, keepdims=True)
        return np.einsum("na,na->n", L[lid], X)

    for _ in range(p.refine_rounds):
        segs = projected_segments(H, shape)
        if not segs:
            break
        D = np.stack([_point_segment_dist(pts, a, b) for _, a, b in segs], axis=1)
        nearest = np.argmin(D, axis=1)
        sel = D[np.arange(len(pts)), nearest] < p.refine_dist_px
        lid = SEG_LINE[np.array([k for k, _, _ in segs])][nearest[sel]]
        n_rows = len(np.unique(lid[lid < len(ROW_X)]))
        n_cols = len(np.unique(lid[lid >= len(ROW_X)]))
        if n_rows < 2 or n_cols < 2 or sel.sum() < 20:
            break
        X = _hom(pts[sel])
        q, lam = np.zeros(8), 1e-3

        def cost(r):
            a = np.abs(r)
            return float(np.where(a < p.huber_px, 0.5 * a * a,
                                  p.huber_px * (a - 0.5 * p.huber_px)).sum())

        r = residuals(H @ update(q), X, lid)
        c0 = cost(r)
        for _ in range(15):
            J = np.empty((len(r), 8))
            for j in range(8):
                dq = np.zeros(8)
                dq[j] = 1e-6
                J[:, j] = (residuals(H @ update(q + dq), X, lid) - r) / 1e-6
            wts = np.where(np.abs(r) < p.huber_px, 1.0, p.huber_px / np.maximum(np.abs(r), 1e-12))
            A = J.T @ (wts[:, None] * J)
            g = J.T @ (wts * r)
            try:
                step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
            except np.linalg.LinAlgError:
                break
            r_new = residuals(H @ update(q + step), X, lid)
            c_new = cost(r_new)
            if np.isfinite(c_new) and c_new < c0:
                q, r, c0, lam = q + step, r_new, c_new, lam / 10
                if np.abs(step).max() < 1e-9:
                    break
            else:
                lam *= 10
                if lam > 1e8:
                    break
        H = H @ update(q)
        H = H / np.linalg.norm(H)
    return H


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ClassicalCourtDetector:
    """``predict(bgr)`` -> dict(keypoints (30,2) normalized original coords or
    NaN, visibility (30,), homography (template->original px) or None,
    status, score, ...)."""

    def __init__(self, params=None):
        self.p = params or ClassicalParams()
        self.last = None

    def _fail(self, status, info):
        info.update(status=status, homography=None,
                    keypoints=np.full((NUM_KEYPOINTS, 2), np.nan),
                    visibility=np.zeros(NUM_KEYPOINTS))
        self.last = info
        return info

    def predict(self, bgr):
        p = self.p
        t0 = time.time()
        h0, w0 = bgr.shape[:2]
        s = p.work_long_side / max(h0, w0)
        img = cv2.resize(bgr, (max(1, round(w0 * s)), max(1, round(h0 * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        shape = img.shape[:2]
        info = {"scale": s, "score": None, "score_norm": None, "n_hypotheses": 0}

        mask, normal = detect_line_pixels(img, p)
        info["n_line_pixels"] = int(mask.sum())
        lines, support = find_lines(mask, normal, p)
        lines, support = lines[:p.max_lines], support[:p.max_lines]
        info["n_lines"] = len(lines)
        info["lines"] = lines
        fam = split_families(lines, p.partition_eps)
        center = np.array([shape[1] / 2.0, shape[0] / 2.0])
        fams = [lines[fam == f][:p.max_per_family] for f in (False, True)]
        info["family_sizes"] = [len(f) for f in fams]
        info["families"] = fams
        if min(len(f) for f in fams) < 2:
            info["time_s"] = time.time() - t0
            return self._fail("too_few_lines", info)
        fam1, fam2 = (order_family(f, center) for f in fams)

        H = hypotheses(fam1, fam2, shape, p)
        info["n_hypotheses"] = len(H)
        if len(H) == 0:
            info["time_s"] = time.time() - t0
            return self._fail("no_hypothesis", info)

        support_mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if len(H) > p.n_prefilter:
            pre = coarse_scores(H, support_mask, p.prefilter_samples)
            H = H[np.argsort(-pre)[:p.n_prefilter]]
        coarse = coarse_scores(H, support_mask, p.coarse_samples)
        cands, seen = [], set()
        for i in np.argsort(-coarse):
            key = _dedupe_key(H[i])
            if key in seen:
                continue
            seen.add(key)
            cands.append(H[i] / np.linalg.norm(H[i]))
            if len(cands) >= p.n_fine:
                break
        fine = np.array([walk_score(Hc, mask) for Hc in cands])
        order = np.argsort(-fine)

        pts = np.argwhere(mask)[:, ::-1].astype(np.float64)
        if len(pts) > p.refine_max_pixels:
            pts = pts[np.random.default_rng(0).choice(len(pts), p.refine_max_pixels, replace=False)]
        best_H, best = None, -np.inf
        for i in order[:p.n_refine]:
            Hr = refine_homography(cands[i], pts, shape, p)
            if not (np.all(np.isfinite(Hr)) and _not_degenerate(Hr[None], p.min_side_px)[0]):
                Hr = cands[i]
            sc = walk_score(Hr, mask)
            # The walk score is quantised to whole pixels, so a slightly-off
            # hypothesis can outscore the sub-pixel fit; only reject
            # refinements that clearly lose support.
            if sc < fine[i] - p.refine_score_tol * abs(fine[i]):
                Hr, sc = cands[i], fine[i]
            if sc > best:
                best_H, best = Hr, sc
        info["score"] = float(best)
        info["score_norm"] = float(best / p.work_long_side)
        info["H_work"] = best_H
        n_on, n_off = walk_stats(best_H, mask)
        info["line_ratio"] = n_on / max(n_on + n_off, 1)
        info["time_s"] = time.time() - t0
        if info["score_norm"] < p.score_threshold:
            return self._fail("low_score", info)

        # Back to original pixels, project, canonicalise.
        Ho = np.diag([1 / s, 1 / s, 1.0]) @ best_H
        v = _hom(COURT_KEYPOINTS_TEMPLATE) @ Ho.T
        front = v[:, 2] > 0
        kps = np.full((NUM_KEYPOINTS, 2), np.nan)
        kps[front] = v[front, :2] / v[front, 2:3] / np.array([w0, h0])
        inside = front & np.all((kps >= 0) & (kps <= 1), axis=1)
        _, _, perm = canonicalize(kps, front.astype(float), image_size=(h0, w0))
        perm = perm or "identity"
        kps, vis = permute_keypoints(kps, inside.astype(float), perm)
        info.update(status="ok", perm=perm, keypoints=kps, visibility=vis,
                    homography=Ho @ SYM_MATRIX[perm])
        self.last = info
        return info


# ---------------------------------------------------------------------------
# Debug drawing
# ---------------------------------------------------------------------------

def draw_debug(bgr, info, params):
    """Working-resolution overlay: line pixels (red), family lines (blue /
    yellow), projected model (green if accepted, magenta if rejected)."""
    s = info["scale"]
    h0, w0 = bgr.shape[:2]
    img = cv2.resize(bgr, (max(1, round(w0 * s)), max(1, round(h0 * s))))
    mask, _ = detect_line_pixels(img, params)
    img[mask] = (0.5 * img[mask] + np.array([0, 0, 127])).astype(np.uint8)
    h, w = img.shape[:2]
    for f, colour in zip(info.get("families", []), ((255, 128, 0), (0, 220, 255))):
        for a, b, c in f:
            if abs(b) > abs(a):
                p1, p2 = (0, -c / b), (w, -(c + a * w) / b)
            else:
                p1, p2 = (-c / a, 0), (-(c + b * h) / a, h)
            ok, q1, q2 = cv2.clipLine((0, 0, w, h), tuple(map(int, p1)), tuple(map(int, p2)))
            if ok:
                cv2.line(img, q1, q2, colour, 1, cv2.LINE_AA)
    H = info.get("H_work")
    if H is not None:
        colour = (0, 255, 0) if info["status"] == "ok" else (255, 0, 255)
        for _, a, b in projected_segments(H, (h, w)):
            cv2.line(img, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                     colour, 2, cv2.LINE_AA)
    label = f"{info['status']} score={info.get('score_norm')}"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Classical line-fitting court baseline")
    ap.add_argument("--manifest")
    ap.add_argument("--eval-split", default="test_real")
    ap.add_argument("--eval-annotations")
    ap.add_argument("--eval-images")
    ap.add_argument("--image", help="run on one image (no scoring)")
    ap.add_argument("--out")
    ap.add_argument("--debug-dir", help="write overlay images here")
    ap.add_argument("--line-mode", choices=["lum", "chroma", "both"], default="lum")
    ap.add_argument("--score-threshold", type=float, default=ClassicalParams.score_threshold)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    params = ClassicalParams(line_mode=args.line_mode, score_threshold=args.score_threshold)
    det = ClassicalCourtDetector(params)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            ap.error(f"cannot read {args.image}")
        info = det.predict(img)
        print(f"{info['status']}: score_norm={info['score_norm']} lines={info['n_lines']} "
              f"families={info['family_sizes']} hypotheses={info['n_hypotheses']} "
              f"({info['time_s']:.1f}s)")
        if args.debug_dir:
            out = os.path.join(args.debug_dir, os.path.basename(args.image))
            cv2.imwrite(os.path.splitext(out)[0] + "_classical.jpg", draw_debug(img, info, params))
        return

    if args.manifest:
        evals = load_items(manifest=args.manifest, split=args.eval_split, images_dir=args.eval_images)
    elif args.eval_annotations:
        evals = load_items(args.eval_annotations, args.eval_images)
    else:
        ap.error("give --manifest, --eval-annotations or --image")

    diags = []

    def predict_fn(img):
        info = det.predict(img)
        diags.append({k: info.get(k) for k in ("status", "score_norm", "line_ratio", "n_lines",
                                                "family_sizes", "n_hypotheses", "perm",
                                                "time_s")})
        if args.debug_dir:
            name = evals[len(diags) - 1]["name"]
            cv2.imwrite(os.path.join(args.debug_dir, os.path.splitext(name)[0] + ".jpg"),
                        draw_debug(img, info, params))
        return info

    rows, summary = run_evaluation(evals, predict_fn, n_boot=args.n_boot, seed=args.seed)
    for r, d in zip(rows, diags):
        r.update({f"classical_{k}": v for k, v in d.items()})
    status = [d["status"] for d in diags]
    summary["meta"] = {
        "baseline": "classical line fitting (Farin 2004 + MonoTrack 2022 family split)",
        "params": dataclasses.asdict(params),
        "n_eval": len(evals),
        "status_counts": {s: status.count(s) for s in sorted(set(status))},
        "completeness": status.count("ok") / len(status) if status else float("nan"),
        "time_s_mean": float(np.mean([d["time_s"] for d in diags])) if diags else float("nan"),
        "manifest": args.manifest,
        "eval_split": args.eval_split if args.manifest else args.eval_annotations,
    }
    print_summary(summary)
    print(f"completeness {summary['meta']['completeness']:.3f}  status {summary['meta']['status_counts']}")
    if args.out:
        write_outputs(args.out, rows, summary)
        print(f"wrote {os.path.splitext(args.out)[0]}.json/_rows.jsonl/.md")


if __name__ == "__main__":
    main()

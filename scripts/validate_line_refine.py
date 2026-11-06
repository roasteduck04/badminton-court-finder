"""Validate line-aware homography refinement (C1) and multi-hypothesis
verification (C2) without a trained model.

Two kinds of input, both on the 640-px letterbox canvas of the model:

  gt         real annotations (``--splits`` of a CVN dataset): the seg mask is
             the GT line mask (``metrics.render_line_mask``, the seg training
             target) corrupted by blur, occlusion blobs, distractor lines and
             speckle; keypoints are GT + Gaussian noise + outliers.
  synthetic  rendered court images with a known homography (main court,
             neighbouring courts, players, noise); the seg mask is a classical
             white-line detector on the letterboxed image, so it is not
             derived from the GT at all; keypoints as above.

Methods (``predict.estimate_court_homography``): magsac (baseline), C1
kp-only (the same robust refinement with alpha = 0, i.e. no line term: the
control for C1), C2 (multi-hypothesis), C1 (line refinement) and C1+C2. Errors are canvas pixels
(= the evaluation's 640-px reference scale), measured on GT-visible points:

  H err    ||H(T_k) - g_k||, the homography itself
  out err  error of the output keypoints with fill="in_frame" (the default:
           inliers keep their decoded position) and fill="all" (every point
           is the template reprojection)
  IoU      court IoU of H vs the GT homography (metrics.court_iou_homography)

It also reports the line-agreement score of correct and of wrong homographies
(labels shifted by one court row, a common failure; another image's court)
for C2's reject flag.

    python scripts/validate_line_refine.py --data data/colab_data_v2 \\
        --splits val test_real test_roboflow --synthetic 200 --out results/line_refine.json
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.court_geometry import (  # noqa: E402
    COURT_KEYPOINTS_TEMPLATE, COURT_WIDTH_DOUBLES, has_homography_support,
)
from src.evaluation.metrics import (  # noqa: E402
    court_iou_homography, fit_gt_homography, render_line_mask,
)
from src.inference.line_refine import (  # noqa: E402
    PAINTED_LINES, canvas_affine, content_region, line_agreement, project,
)
from src.inference.predict import (  # noqa: E402
    estimate_court_homography, fit_court_homography, refine_keypoints,
)
from src.preprocessing.letterbox import letterbox  # noqa: E402

SIZE = 640
SEG_THICKNESS = 5
METHODS = {
    "magsac": {},
    "C1 kp-only": {"line_refine": True, "line_options": {"alpha": 0.0}},
    "C2": {"multi_hypothesis": True},
    "C1": {"line_refine": True},
    "C1+C2": {"multi_hypothesis": True, "line_refine": True},
}


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------

def corrupt_seg(mask, rng, valid):
    """GT line mask -> plausible seg probabilities: blur, occlusions,
    distractor lines (other courts / players / ads), speckle."""
    seg = mask.astype(np.float32)
    for _ in range(rng.integers(2, 7)):                      # occluding players
        c = rng.uniform(0, SIZE, 2).astype(int)
        cv2.ellipse(seg, (int(c[0]), int(c[1])), (int(rng.uniform(6, 20)), int(rng.uniform(15, 45))),
                    0, 0, 360, 0.0, -1)
    ys, xs = np.nonzero(mask)
    for _ in range(rng.integers(1, 4)):                      # parallel distractor lines
        if len(xs) < 2:
            break
        i, j = rng.choice(len(xs), 2, replace=False)
        d = np.array([xs[j] - xs[i], ys[j] - ys[i]], dtype=np.float64)
        if np.linalg.norm(d) < 20:
            continue
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d) * rng.uniform(20, 80) * rng.choice([-1, 1])
        p1 = np.array([xs[i], ys[i]]) + nrm
        p2 = np.array([xs[j], ys[j]]) + nrm
        cv2.line(seg, tuple(int(v) for v in p1), tuple(int(v) for v in p2), 1.0, 3)
    for _ in range(rng.integers(0, 4)):                      # random strokes
        p = rng.uniform(0, SIZE, 4).astype(int)
        cv2.line(seg, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1.0, 2)
    seg = cv2.GaussianBlur(seg, (0, 0), 1.2) * 1.6
    seg += (rng.random(seg.shape) < 0.002).astype(np.float32)  # speckle
    seg *= valid  # the model's seg is zero in the letterbox padding
    return np.clip(seg, 0, 1)


def perturb_keypoints(gt_px, vis, rng, sigma, outlier_rate, scale):
    """Canvas-px noise sigma, outliers displaced 15-60 canvas px; returns
    original-pixel keypoints."""
    kp = gt_px + rng.normal(0, sigma / scale, gt_px.shape)
    out = vis & (rng.random(len(vis)) < outlier_rate)
    ang = rng.uniform(0, 2 * np.pi, out.sum())
    r = rng.uniform(15, 60, out.sum()) / scale
    kp[out] += np.stack([np.cos(ang), np.sin(ang)], 1) * r[:, None]
    return kp


def gt_items(data, splits):
    for split in splits:
        for path in sorted(glob.glob(os.path.join(data, split, "annotations", "*.json"))):
            with open(path, encoding="utf-8") as f:
                ann = json.load(f)
            kps = np.asarray(ann["keypoints"], dtype=np.float64)
            vis = np.asarray(ann["visibility"], dtype=np.float64) > 0.5
            if not has_homography_support(vis):
                continue
            yield split, os.path.basename(path), kps, vis, tuple(ann["image_size"])


def random_camera(rng, w, h):
    """A random plausible template->image homography (court mostly in view)."""
    view = rng.choice(["end", "side", "oblique"])
    if view == "end":
        near_y, far_y = rng.uniform(0.75, 1.05), rng.uniform(0.1, 0.35)
        near_half, far_half = rng.uniform(0.3, 0.5), rng.uniform(0.1, 0.2)
        cx = rng.uniform(0.4, 0.6)
        dst = [[cx - near_half, near_y], [cx + near_half, near_y],
               [cx - far_half, far_y], [cx + far_half, far_y]]
        src = COURT_KEYPOINTS_TEMPLATE[[0, 4, 25, 29]]
    else:
        near_y, far_y = rng.uniform(0.7, 0.95), rng.uniform(0.25, 0.45)
        near_half, far_half = rng.uniform(0.35, 0.55), rng.uniform(0.25, 0.4)
        cx = rng.uniform(0.4, 0.6)
        skew = rng.uniform(-0.15, 0.15) if view == "oblique" else 0.0
        dst = [[cx - near_half, near_y], [cx + near_half, near_y + skew],
               [cx - far_half + skew, far_y], [cx + far_half, far_y - skew]]
        src = COURT_KEYPOINTS_TEMPLATE[[4, 29, 0, 25]]
    dst = np.asarray(dst) * [w, h] + rng.uniform(-10, 10, (4, 2))
    H, _ = cv2.findHomography(src, dst, 0)
    return H


def render_synthetic(H, w, h, rng):
    """BGR image of the court under H, with neighbouring courts and players."""
    floor = np.array(rng.uniform([30, 60, 20], [90, 140, 90]), dtype=np.float32)
    img = np.ones((h, w, 3), np.float32) * floor
    img += cv2.GaussianBlur(rng.normal(0, 12, (h, w, 3)).astype(np.float32), (0, 0), 3)
    # neighbours: copies of the template shifted across the width
    gap = rng.uniform(1.5, 3.0)
    shifts = [0.0] + [s * (COURT_WIDTH_DOUBLES + gap) for s in (-1, 1) if rng.random() < 0.7]
    for dy in shifts:
        Hs = H @ np.array([[1, 0, 0], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        P, depth = project(Hs, COURT_KEYPOINTS_TEMPLATE)
        for a, b in PAINTED_LINES:
            if depth[a] <= 0 or depth[b] <= 0:
                continue
            pa, pb = np.clip(P[a], -1e4, 1e4), np.clip(P[b], -1e4, 1e4)
            cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb),
                     (230, 230, 230), int(rng.integers(2, 4)), cv2.LINE_AA)
    for _ in range(rng.integers(2, 8)):                      # players
        c = rng.uniform([0, 0.2 * h], [w, h])
        s = rng.uniform(15, 50)
        cv2.rectangle(img, (int(c[0] - s / 3), int(c[1] - s * 2)), (int(c[0] + s / 3), int(c[1])),
                      tuple(float(v) for v in rng.uniform(0, 255, 3)), -1)
    img += rng.normal(0, 4, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def classical_seg(image):
    """White-line detector on the letterboxed image: top-hat + threshold."""
    canvas, _ = letterbox(image, SIZE)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    return np.clip((th - 25.0) / 40.0, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def score(H, raw_norm, cands, inliers, gt_px, vis, image_hw, A, H_gt):
    h, w = image_hw
    s = A[0, 0]
    if H is None:
        return {"ok": False, "iou": court_iou_homography(H_gt, None, image_hw) if H_gt is not None
                else np.nan}
    proj, _ = project(H, COURT_KEYPOINTS_TEMPLATE)
    e_h = np.linalg.norm(proj - gt_px, axis=1)[vis] * s
    res = {"ok": True, "h_err": e_h,
           "iou": court_iou_homography(H_gt, H, image_hw) if H_gt is not None else np.nan}
    for fill in ("in_frame", "all"):
        kp, _ = refine_keypoints(raw_norm, H, cands, inliers, image_hw, fill)
        res[f"out_{fill}"] = np.linalg.norm(kp * [w, h] - gt_px, axis=1)[vis] * s
    return res


def run_case(raw_px, cands, gt_px, vis, image_hw, seg, A, H_gt, opts, out):
    h, w = image_hw
    raw_norm = raw_px / [w, h]
    s = A[0, 0]
    out.setdefault("raw", []).append(np.linalg.norm(raw_px - gt_px, axis=1)[vis] * s)
    for name, method in METHODS.items():
        kw = dict(method)
        kw["line_options"] = {**opts["line"], **kw.get("line_options", {})}
        kw["hypothesis_options"] = opts["hypothesis"]
        t0 = time.perf_counter()
        fit = estimate_court_homography(raw_norm, cands, image_hw, seg, A, None, **kw)
        dt = time.perf_counter() - t0
        r = score(fit["H"], raw_norm, cands, fit["inliers"], gt_px, vis, image_hw, A, H_gt)
        r["time"] = dt
        r["line_score"] = fit["line_score"]
        r["refine_status"] = fit["refine_status"]
        out.setdefault(name, []).append(r)


def _score_of(H, image_hw, seg, A):
    if H is None:
        return None
    return line_agreement(A @ H, seg, valid=content_region(A, image_hw, SIZE))["f1"]


def row_shift_fit(raw_px, cands, image_hw):
    """A wrong fit: every label shifted by one court row."""
    h, w = image_hw
    shifted = np.zeros_like(raw_px)
    shifted[5:] = raw_px[:-5]
    mask = np.zeros_like(cands)
    mask[5:] = cands[:-5]
    H, _, _ = fit_court_homography(shifted / [w, h], mask, image_hw, 0.02, 4)
    return H


def summarize(out):
    rows = {}
    raw = np.concatenate(out["raw"])
    rows["raw keypoints"] = {"mean": float(raw.mean()), "median": float(np.median(raw)),
                             "pck@5": float((raw <= 5).mean())}
    # Errors are compared on the images every method fitted.
    common = np.all([[r["ok"] for r in out[name]] for name in METHODS], axis=0)
    rows["n_images"] = int(len(common))
    rows["n_common"] = int(common.sum())
    for name in METHODS:
        rs = out[name]
        d = {"fail": float(np.mean([not r["ok"] for r in rs]))}
        for key in ("h_err", "out_in_frame", "out_all"):
            e = np.concatenate([r[key] for r, c in zip(rs, common) if c]) if common.any() \
                else np.zeros(0)
            d[f"{key}_mean"] = float(e.mean()) if e.size else np.nan
            d[f"{key}_median"] = float(np.median(e)) if e.size else np.nan
            d[f"{key}_pck@5"] = float((e <= 5).mean()) if e.size else np.nan
        d["iou_mean"] = float(np.nanmean([r["iou"] for r in rs]))
        d["ms"] = 1000 * float(np.mean([r["time"] for r in rs]))
        st = [r["refine_status"] for r in rs if r["ok"]]
        d["refine_ok"] = float(np.mean([s == "ok" for s in st])) if "C1" in name and st else np.nan
        rows[name] = d
    return rows


TABLE_COLS = ["fail", "h_err_mean", "h_err_median", "h_err_pck@5", "out_in_frame_mean",
              "out_all_mean", "out_all_pck@5", "iou_mean", "refine_ok", "ms"]


def print_table(title, rows):
    print(f"\n### {title}\n")
    print(f"Errors on the {rows['n_common']} of {rows['n_images']} images that every method "
          "fitted; fail and IoU (failure = 0) on all.\n")
    cols = TABLE_COLS
    print("| method | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    raw = rows["raw keypoints"]
    print(f"| raw keypoints | | {raw['mean']:.2f} | {raw['median']:.2f} | {100 * raw['pck@5']:.1f} |"
          + " |" * (len(cols) - 4))
    for name in METHODS:
        d = rows[name]
        cells = []
        for c in cols:
            v = d[c]
            if np.isnan(v):
                cells.append("")
            elif "pck" in c or c in ("fail", "iou_mean", "refine_ok"):
                cells.append(f"{100 * v:.1f}")
            else:
                cells.append(f"{v:.2f}")
        print(f"| {name} | " + " | ".join(cells) + " |")


def gt_cases(args):
    """(raw_px, cands, gt_px, vis, image_hw, seg, A, H_gt, H_foreign) per annotation."""
    prev = None
    for i, (_split, _name, kps, vis, (h, w)) in enumerate(gt_items(args.data, args.splits)):
        rng = np.random.default_rng([args.seed, i])
        A = canvas_affine((h, w), SIZE)
        gt_px = kps * [w, h]
        gt_canvas = project(A, gt_px)[0] / SIZE
        seg = corrupt_seg(render_line_mask(gt_canvas, vis, SIZE, SEG_THICKNESS), rng,
                          content_region(A, (h, w), SIZE))
        raw = perturb_keypoints(gt_px, vis, rng, args.sigma, args.outliers, A[0, 0])
        H_gt = fit_gt_homography(gt_px, vis)
        # Foreign: the previous annotation's court, stretched to this image.
        foreign = fit_gt_homography(prev[0] * [w, h], prev[1]) if prev is not None else None
        prev = (kps, vis)
        yield raw, vis.copy(), gt_px, vis, (h, w), seg, A, H_gt, foreign


def synthetic_cases(args):
    w, h = 1280, 720
    A = canvas_affine((h, w), SIZE)
    for i in range(args.synthetic):
        rng = np.random.default_rng([args.seed, 10_000 + i])
        H = random_camera(rng, w, h)
        P, depth = project(H, COURT_KEYPOINTS_TEMPLATE)
        vis = (depth > 0) & (P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)
        if not has_homography_support(vis):
            continue
        seg = classical_seg(render_synthetic(H, w, h, rng))
        raw = perturb_keypoints(P, vis, rng, args.sigma, args.outliers, A[0, 0])
        yield raw, vis.copy(), P, vis, (h, w), seg, A, H, random_camera(rng, w, h)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/colab_data_v2")
    ap.add_argument("--splits", nargs="*", default=["test_real", "test_roboflow"])
    ap.add_argument("--synthetic", type=int, default=100, help="number of synthetic images")
    ap.add_argument("--sigma", type=float, default=3.0, help="keypoint noise, canvas px")
    ap.add_argument("--outliers", type=float, default=0.15, help="outlier rate")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--loss", default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--subsets", nargs="*", default=None, help="C2 keypoint subsets")
    ap.add_argument("--margin", type=float, default=None, help="C2 margin over the default fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    args = ap.parse_args(argv)
    opts = {
        "line": {k: getattr(args, k) for k in ("alpha", "loss", "delta")
                 if getattr(args, k) is not None},
        "hypothesis": {k: getattr(args, k) for k in ("subsets", "margin")
                       if getattr(args, k) is not None},
    }

    results = {"options": opts}
    sources = []
    if args.splits:
        sources.append(("gt", f"GT-derived ({', '.join(args.splits)})", gt_cases(args)))
    if args.synthetic:
        sources.append(("synthetic", "Synthetic, known H", synthetic_cases(args)))
    line_scores = {}
    for key, title, cases in sources:
        out, sc = {}, {"correct": [], "row_shift": [], "foreign": []}
        for raw, cands, gt_px, vis, hw, seg, A, H_gt, foreign in cases:
            run_case(raw, cands, gt_px, vis, hw, seg, A, H_gt, opts, out)
            sc["correct"].append(out["magsac"][-1]["line_score"])
            sc["row_shift"].append(_score_of(row_shift_fit(raw, cands, hw), hw, seg, A))
            sc["foreign"].append(_score_of(foreign, hw, seg, A))
        results[key] = summarize(out)
        line_scores[key] = {k: [x for x in v if x is not None] for k, v in sc.items()}
        print_table(title, results[key])

    print("\n### Line score (F1) of correct and wrong homographies\n")
    print("Share (%) below each reject threshold; correct = the MAGSAC fit, row_shift = labels "
          "shifted by one court row, foreign = another image's court.\n")
    thr = (0.2, 0.3, 0.4, 0.5)
    print("| input | fit | n | median | " + " | ".join(f"<{t}" for t in thr) + " |")
    print("|---" * (4 + len(thr)) + "|")
    for key, sc in line_scores.items():
        for kind, vals in sc.items():
            v = np.asarray(vals, dtype=np.float64)
            if not v.size:
                continue
            cells = " | ".join(f"{100 * (v < t).mean():.0f}" for t in thr)
            print(f"| {key} | {kind} | {v.size} | {np.median(v):.2f} | {cells} |")
    results["line_score"] = line_scores

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""Camera-realism augmentation for synthetic (Blender) CVN datasets.

Applies, with per-image probabilities, in this order:
  1. Brown-Conrady lens distortion (k1, k2, p1, p2) - warps the image AND the
     keypoints consistently; keypoints that leave the frame become invisible.
     The output focal length is chosen so every output pixel samples the
     source image (no black borders).
  2. Vignetting
  3. Defocus (disk) or motion blur
  4. Sensor noise (Gaussian read noise + intensity-dependent shot noise)
  5. JPEG compression (quality 30-95)

Runs in the project Python (numpy + OpenCV) on CVN-format images +
annotations and writes to a separate --out directory. Deterministic given
--seed: each output's RNG is derived from (seed, image stem, copy index).

Coordinates: CVN keypoints are normalized with the pixel-centre convention
(pixel i <-> (i + 0.5) / W); OpenCV pixel coordinates are norm * W - 0.5.

Usage:
    python src/tools/synthetic_augment.py --images data/blender/images \
        --annotations data/blender/annotations --out data/blender_aug --seed 0
"""

import argparse
import copy
import hashlib
import json
import os
import sys

import cv2
import numpy as np

DEFAULT_PROBS = {"distort": 0.5, "vignette": 0.4, "blur": 0.4, "noise": 0.6, "jpeg": 0.7}


# --------------------------------------------------------------------------
# Lens distortion
# --------------------------------------------------------------------------

def _radial_monotonic_limit(k1, k2, r_max=3.0, steps=3000):
    """Largest undistorted radius up to which r_d = r(1 + k1 r^2 + k2 r^4) increases."""
    r = np.linspace(0.0, r_max, steps)
    d = 1 + 3 * k1 * r ** 2 + 5 * k2 * r ** 4
    bad = np.nonzero(d <= 0.05)[0]
    return float(r[bad[0] - 1]) if len(bad) else float(r_max)


def _distort_normalized(xy, dist):
    """Brown-Conrady forward model on normalized coords (N, 2)."""
    k1, k2, p1, p2 = dist
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], axis=1)


def _undistort_normalized(xy, dist):
    """Inverse Brown-Conrady via OpenCV's iterative solver (N, 2)."""
    pts = xy.reshape(-1, 1, 2).astype(np.float64)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)
    out = cv2.undistortPointsIter(pts, np.eye(3), np.array(list(dist) + [0.0]), None, None, crit)
    return out.reshape(-1, 2)


def sample_distortion(rng, image_size):
    """Random plausible distortion params for an image of [h, w]."""
    h, w = image_size
    f = float(rng.uniform(0.55, 1.2)) * max(w, h)
    r_corner = 0.5 * np.hypot(w, h) / f
    for _ in range(50):
        if rng.random() < 0.75:  # barrel (wide-angle / phone) is the common case
            k1 = float(rng.uniform(-0.35, -0.03))
        else:
            k1 = float(rng.uniform(0.03, 0.15))
        k2 = float(rng.uniform(-0.04, 0.04))
        if _radial_monotonic_limit(k1, k2) > r_corner * 1.6:
            break
    else:
        k1, k2 = -0.05, 0.0
    p1, p2 = (float(v) for v in rng.uniform(-0.002, 0.002, size=2))
    return {"f": f, "k1": k1, "k2": k2, "p1": p1, "p2": p2}


def _source_coords(u, v, f, s, cx, cy, dist):
    """Source pixel for output pixels (u, v) with output focal f*s."""
    xy_d = np.stack([(u - cx) / (f * s), (v - cy) / (f * s)], axis=1)
    xy_u = _undistort_normalized(xy_d, dist)
    return xy_u[:, 0] * f + cx, xy_u[:, 1] * f + cy


def _border_pixels(w, h, n=64):
    xs = np.linspace(0, w - 1, n)
    ys = np.linspace(0, h - 1, n)
    u = np.concatenate([xs, xs, np.zeros(n), np.full(n, w - 1.0)])
    v = np.concatenate([np.zeros(n), np.full(n, h - 1.0), ys, ys])
    return u, v


def _choose_output_scale(w, h, f, cx, cy, dist):
    """Smallest focal scale s such that every output pixel samples inside the source."""
    u, v = _border_pixels(w, h)

    def inside(s):
        sx, sy = _source_coords(u, v, f, s, cx, cy, dist)
        return bool(np.all((sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)))

    lo, hi = 0.3, 3.0
    if not inside(hi):
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if inside(mid):
            hi = mid
        else:
            lo = mid
    return hi


def distort_brown_conrady(img, kps_norm, vis, params):
    """Warp image + keypoints with Brown-Conrady distortion.

    Args:
        img: (H, W, C) or (H, W) uint8
        kps_norm: (30, 2) normalized keypoints (pixel-centre convention)
        vis: (30,) visibility flags
        params: dict f, k1, k2, p1, p2 (from sample_distortion)

    Returns:
        (img_out, kps_out, vis_out, scale); invisible keypoints are [-1, -1].
    """
    h, w = img.shape[:2]
    f = params["f"]
    dist = (params["k1"], params["k2"], params["p1"], params["p2"])
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    s = _choose_output_scale(w, h, f, cx, cy, dist)

    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    sx, sy = _source_coords(uu.ravel(), vv.ravel(), f, s, cx, cy, dist)
    map_x = sx.reshape(h, w).astype(np.float32)
    map_y = sy.reshape(h, w).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    kps = np.asarray(kps_norm, dtype=np.float64).copy()
    vis = np.asarray(vis).astype(np.int64).copy()
    r_valid = _radial_monotonic_limit(params["k1"], params["k2"])
    kps_out = np.full_like(kps, -1.0)
    for k in np.nonzero(vis > 0)[0]:
        px = kps[k, 0] * w - 0.5
        py = kps[k, 1] * h - 0.5
        xu = np.array([[(px - cx) / f, (py - cy) / f]])
        if np.hypot(*xu[0]) >= r_valid:
            vis[k] = 0
            continue
        xd = _distort_normalized(xu, dist)[0]
        qx, qy = xd[0] * f * s + cx, xd[1] * f * s + cy
        if -0.5 <= qx <= w - 0.5 and -0.5 <= qy <= h - 0.5:
            kps_out[k] = [(qx + 0.5) / w, (qy + 0.5) / h]
        else:
            vis[k] = 0
    return out, kps_out, vis, s


# --------------------------------------------------------------------------
# Photometric degradations
# --------------------------------------------------------------------------

def apply_vignette(img, strength):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2) / (((w - 1) / 2) ** 2 + ((h - 1) / 2) ** 2)
    gain = np.clip(1.0 - strength * r2, 0.0, 1.0)
    if img.ndim == 3:
        gain = gain[..., None]
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def apply_defocus(img, radius):
    k = int(np.ceil(radius)) * 2 + 1
    kernel = np.zeros((k, k), np.float32)
    cv2.circle(kernel, (k // 2, k // 2), max(1, int(round(radius))), 1.0, -1, cv2.LINE_AA)
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def apply_motion_blur(img, length, angle_deg):
    k = int(length) | 1
    kernel = np.zeros((k, k), np.float32)
    c = k // 2
    dx, dy = np.cos(np.radians(angle_deg)) * c, np.sin(np.radians(angle_deg)) * c
    cv2.line(kernel, (int(round(c - dx)), int(round(c - dy))),
             (int(round(c + dx)), int(round(c + dy))), 1.0, 1, cv2.LINE_AA)
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def apply_noise(img, rng, read_sigma, shot_gain):
    x = img.astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    sigma = np.sqrt(read_sigma ** 2 + shot_gain * x)
    return np.clip(x + noise * sigma, 0, 255).astype(np.uint8)


def apply_jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def make_rng(seed, stem, copy_index=0):
    digest = hashlib.sha256(f"{seed}:{stem}:{copy_index}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _bbox(kps, vis):
    idx = [i for i in range(len(vis)) if vis[i]]
    if not idx:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [kps[i][0] for i in idx]
    ys = [kps[i][1] for i in idx]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def augment_sample(img, ann, rng, probs=None):
    """Augment one image + CVN annotation. Returns (img, ann, params)."""
    probs = {**DEFAULT_PROBS, **(probs or {})}
    ann = copy.deepcopy(ann)
    kps = np.asarray(ann["keypoints"], dtype=np.float64)
    vis = np.asarray(ann["visibility"]).astype(np.int64)
    params = {}

    if rng.random() < probs["distort"]:
        d = sample_distortion(rng, img.shape[:2])
        img, kps, vis, s = distort_brown_conrady(img, kps, vis, d)
        params["distortion"] = {**{k: round(v, 6) for k, v in d.items()}, "output_scale": round(s, 6)}
    if rng.random() < probs["vignette"]:
        v = float(rng.uniform(0.1, 0.5))
        img = apply_vignette(img, v)
        params["vignette"] = round(v, 4)
    if rng.random() < probs["blur"]:
        if rng.random() < 0.5:
            r = float(rng.uniform(1.0, 3.5))
            img = apply_defocus(img, r)
            params["defocus_radius"] = round(r, 3)
        else:
            length, angle = int(rng.integers(3, 16)), float(rng.uniform(0, 180))
            img = apply_motion_blur(img, length, angle)
            params["motion_blur"] = {"length": length, "angle": round(angle, 2)}
    if rng.random() < probs["noise"]:
        read, shot = float(rng.uniform(1.0, 8.0)), float(rng.uniform(0.0, 0.5))
        img = apply_noise(img, rng, read, shot)
        params["noise"] = {"read_sigma": round(read, 3), "shot_gain": round(shot, 4)}
    if rng.random() < probs["jpeg"]:
        q = int(rng.integers(30, 96))
        img = apply_jpeg(img, q)
        params["jpeg_quality"] = q

    kps[vis == 0] = -1.0
    ann["keypoints"] = kps.tolist()
    ann["visibility"] = vis.tolist()
    ann["bounding_box"] = _bbox(ann["keypoints"], ann["visibility"])
    ann["image_size"] = [int(img.shape[0]), int(img.shape[1])]
    ann["augment"] = params
    return img, ann, params


def augment_dataset(images_dir, annotations_dir, out_dir, seed=0, copies=1, probs=None):
    out_img = os.path.join(out_dir, "images")
    out_ann = os.path.join(out_dir, "annotations")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_ann, exist_ok=True)
    n = 0
    for fname in sorted(os.listdir(annotations_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(annotations_dir, fname), encoding="utf-8") as f:
            ann = json.load(f)
        img = cv2.imread(os.path.join(images_dir, os.path.basename(ann["image_path"])), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skip {fname}: image not found")
            continue
        stem = os.path.splitext(os.path.basename(ann["image_path"]))[0]
        for c in range(copies):
            out, new_ann, _ = augment_sample(img, ann, make_rng(seed, stem, c), probs)
            name = f"{stem}_aug{c}.png" if copies > 1 else f"{stem}.png"
            new_ann["image_path"] = name
            cv2.imwrite(os.path.join(out_img, name), out)
            with open(os.path.join(out_ann, os.path.splitext(name)[0] + ".json"), "w", encoding="utf-8") as f:
                json.dump(new_ann, f, indent=2)
            n += 1
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--copies", type=int, default=1)
    for k, v in DEFAULT_PROBS.items():
        p.add_argument(f"--p-{k}", type=float, default=v, help=f"probability (default {v})")
    a = p.parse_args(argv)
    if os.path.abspath(a.out) in (os.path.abspath(a.images), os.path.abspath(a.annotations)):
        sys.exit("--out must be a new directory, not the input directory")
    probs = {k: getattr(a, f"p_{k}") for k in DEFAULT_PROBS}
    n = augment_dataset(a.images, a.annotations, a.out, a.seed, a.copies, probs)
    print(f"Wrote {n} augmented images to {a.out}")


if __name__ == "__main__":
    main()

"""PyTorch Dataset for CourtVisionNet badminton court detection.

Pipeline per sample (same geometry as inference):

    BGR image -> letterbox to a square canvas (aspect preserved)
              -> [train] albumentations on the BGR uint8 canvas + keypoints
                 + content-validity mask
              -> canonical relabeling of the keypoints (court symmetry)
              -> 7 input channels (``generate_channels``), sliced to
                 ``in_channels``
              -> Gaussian heatmap targets + court-line segmentation mask

Annotations come either from a directory of CVN JSON files or from a split
of a dataset manifest (JSON: split -> list of {file, source, group, ...}).

Coordinates: keypoints are normalized [0,1] in the square canvas with the
pixel-centre convention (canvas pixel i covers [i/S, (i+1)/S); heatmap index
j <-> normalized (j + 0.5) / W).
"""

import argparse
import glob
import json
import os
from collections import Counter

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.court_geometry import (
    COURT_KEYPOINTS_TEMPLATE,
    NUM_KEYPOINTS,
    canonicalize,
    get_court_lines,
    has_homography_support,
    permute_keypoints,
)
from src.preprocessing.augmentation import reseed_transform
from src.preprocessing.channels import generate_channels
from src.preprocessing.letterbox import content_mask, keypoints_to_letterbox, letterbox

# The K12-K17 centre-line segment crosses the net and is not painted.
VIRTUAL_SEGMENTS = {(12, 17)}
PAINTED_SEGMENTS = [s for s in get_court_lines() if tuple(s) not in VIRTUAL_SEGMENTS]

_LINE_SHIFT = 4  # cv2 fractional bits for sub-pixel line drawing
# A GT homography whose fit residual exceeds this (canvas px) is treated as
# unreliable and the seg mask falls back to visible-segment drawing.
_MAX_FIT_RESIDUAL_PX = 8.0


def infer_source(path):
    """Guess the data source from a file name (used when not recorded)."""
    base = os.path.basename(str(path)).lower()
    if base.startswith("bl_") or "blender" in base or base.startswith("render"):
        return "blender"
    if base.startswith("ds_") or ".rf." in base:
        return "roboflow"
    if base.startswith("mine_"):
        return "manual"
    return "unknown"


def load_manifest_split(manifest_path, split):
    """Read one split of a dataset manifest.

    Accepts ``{"splits": {split: [...]}}`` or ``{split: [...]}``. Entries are
    dicts with ``file`` (annotation JSON, relative to the manifest's
    ``root``/directory, or absolute) plus optional ``image``, ``source``,
    ``group``; a bare string is treated as ``file``.

    Returns:
        list of dicts with keys ann_path, image_candidates, source, group.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    root = manifest.get("root") or manifest_dir
    if not os.path.isabs(root):
        root = os.path.join(manifest_dir, root)
    splits = manifest.get("splits", manifest)
    if split not in splits or not isinstance(splits[split], list):
        available = [k for k, v in splits.items() if isinstance(v, list)]
        raise KeyError(f"Split {split!r} not in {manifest_path}; available: {available}")

    items = []
    for entry in splits[split]:
        if isinstance(entry, str):
            entry = {"file": entry}
        rel = entry.get("file") or entry.get("annotation") or entry.get("ann")
        if rel is None:
            raise ValueError(f"Manifest entry without 'file': {entry}")
        # scripts/prepare_colab.py layout: <root>/<split>/{annotations,images}/
        candidates = [rel] if os.path.isabs(rel) else [
            os.path.join(root, split, "annotations", rel),
            os.path.join(root, rel),
            os.path.join(root, "annotations", rel),
            os.path.join(manifest_dir, rel),
        ]
        ann_path = next((c for c in candidates if os.path.isfile(c)), None)
        if ann_path is None:
            raise FileNotFoundError(f"Annotation {rel!r} of split {split!r} not found; tried {candidates}")
        image = entry.get("image") or entry.get("image_path")
        image_candidates = []
        if image:
            image_candidates = [image] if os.path.isabs(image) else [
                os.path.join(root, split, "images", image),
                os.path.join(root, image),
                os.path.join(root, "images", image),
            ]
        items.append({
            "ann_path": ann_path,
            "image_candidates": image_candidates,
            "source": entry.get("source"),
            "group": entry.get("group"),
        })
    return items


class CourtDataset(Dataset):
    """PyTorch dataset for badminton court detection.

    Each sample is a dict with (S = image_size, h = heatmap_size, K = 30):
        image:      (C, S, S) float32 in [0, 1], C = in_channels (7 or 3)
        heatmaps:   (K, h, h) float32 Gaussian per visible keypoint (peak 1
                    at the keypoint), zeros for invisible keypoints
        keypoints:  (K, 2) float32 normalized canvas coords, [-1, -1] if invisible
        visibility: (K,) float32 {0, 1}
        mask:       (1, S, S) float32 {0, 1} court-line segmentation target
        valid:      (1, S, S) float32 {0, 1} real-image-content mask
        source:     str ("roboflow" | "blender" | "manual" | "unknown")
        group:      str (video/session id, "" if unknown)
        index:      int

    Args:
        annotations_dir: directory of CVN annotation JSONs (ignored when
            ``manifest`` is given).
        images_dir: directory used for image basename lookup.
        transform: albumentations ReplayCompose from
            ``src.preprocessing.augmentation`` (None = no augmentation).
        manifest / split: read the annotation list from a manifest split.
        exclude_sources: drop samples whose source is in this list.
        canonicalize_labels: relabel keypoints into the canonical court
            symmetry convention after augmentation (see
            ``court_geometry.canonicalize``).
    """

    def __init__(self, annotations_dir=None, images_dir=None, transform=None,
                 image_size=640, heatmap_size=None, sigma=2.0, line_thickness=5,
                 in_channels=7, manifest=None, split=None, exclude_sources=None,
                 canonicalize_labels=True):
        self.images_dir = images_dir
        self.image_size = image_size
        self.heatmap_size = heatmap_size if heatmap_size is not None else image_size // 4
        self.sigma = float(sigma)
        self.line_thickness = int(line_thickness)
        self.in_channels = int(in_channels)
        self.transform = transform
        self.canonicalize_labels = canonicalize_labels

        if manifest:
            if split is None:
                raise ValueError("split is required with manifest")
            items = load_manifest_split(manifest, split)
            self._search_dirs = [os.path.join(os.path.dirname(os.path.abspath(manifest)), "images")]
        elif annotations_dir:
            items = [
                {"ann_path": p, "image_candidates": [], "source": None, "group": None}
                for p in sorted(glob.glob(os.path.join(annotations_dir, "*.json")))
            ]
            self._search_dirs = []
        else:
            items = []
            self._search_dirs = []
        if images_dir:
            self._search_dirs.insert(0, images_dir)

        exclude = set(exclude_sources or [])
        self.items = []
        for item in items:
            with open(item["ann_path"], "r", encoding="utf-8") as f:
                ann = json.load(f)
            item = dict(item)
            item["ann"] = ann
            item["source"] = item["source"] or ann.get("source") or infer_source(item["ann_path"])
            item["group"] = item["group"] or ann.get("group") or ""
            if item["source"] in exclude:
                continue
            self.items.append(item)

    # Kept for backward compatibility with code that inspected this list.
    @property
    def annotation_paths(self):
        return [it["ann_path"] for it in self.items]

    @property
    def sources(self):
        return [it["source"] for it in self.items]

    def reseed(self, seed):
        """Reseed the augmentation RNG (used for num_workers=0 loading)."""
        reseed_transform(self.transform, seed)

    def __len__(self):
        return len(self.items)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        item = self.items[idx]
        ann = item["ann"]
        size = self.image_size

        image = self._load_image(ann, item)
        canvas, info = letterbox(image, size)
        valid = content_mask(info)

        keypoints = np.array(ann["keypoints"], dtype=np.float64).reshape(-1, 2)
        visibility = (np.array(ann["visibility"], dtype=np.float64) > 0).astype(np.float32)
        keypoints[visibility < 1] = -1.0
        keypoints = keypoints_to_letterbox(keypoints, info)

        flipped = False
        if self.transform is not None:
            canvas, valid, keypoints, visibility, flipped = self._apply_transform(
                canvas, valid, keypoints, visibility
            )
        keypoints, visibility = self._drop_outside(keypoints, visibility, valid)
        keypoints, visibility = self._relabel(keypoints, visibility, flipped)

        if self.in_channels == 3:
            channels = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            channels = generate_channels(canvas, valid)[..., :self.in_channels]

        heatmaps = self._generate_heatmaps(keypoints, visibility)
        mask = self._generate_line_mask(keypoints, visibility) * valid.astype(np.float32)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(channels.transpose(2, 0, 1))),
            "heatmaps": torch.from_numpy(heatmaps),
            "keypoints": torch.from_numpy(keypoints.astype(np.float32)),
            "visibility": torch.from_numpy(visibility.astype(np.float32)),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "valid": torch.from_numpy(valid.astype(np.float32)).unsqueeze(0),
            "source": item["source"],
            "group": str(item["group"]),
            "index": idx,
        }

    def _load_image(self, ann, item):
        image_path = ann["image_path"]
        base = os.path.basename(image_path.replace("\\", "/"))
        candidates = list(item.get("image_candidates") or [])
        if os.path.isabs(image_path):
            candidates.append(image_path)
        ann_dir = os.path.dirname(item["ann_path"])
        for d in self._search_dirs + [os.path.join(ann_dir, "..", "images"), ann_dir]:
            candidates.append(os.path.join(d, base))
        if not os.path.isabs(image_path):
            candidates.append(image_path)
        for path in candidates:
            if os.path.isfile(path):
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is not None:
                    return image
        raise FileNotFoundError(
            f"Image {image_path!r} for {item['ann_path']} not found; tried {candidates}"
        )

    def _apply_transform(self, canvas, valid, keypoints, visibility):
        """Augment the BGR canvas, validity mask and keypoints together.

        Returns (canvas, valid, keypoints, visibility, flipped).
        """
        size = self.image_size
        vis_idx = np.flatnonzero(visibility > 0)
        # Normalized pixel-centre coords -> albumentations pixel-index coords.
        kps_px = keypoints[vis_idx] * size - 0.5
        out = self.transform(image=canvas, mask=valid, keypoints=kps_px.reshape(-1, 2))

        new_kps = np.full_like(keypoints, -1.0)
        new_vis = np.zeros_like(visibility)
        moved = np.asarray(out["keypoints"], dtype=np.float64).reshape(-1, 2)
        if len(vis_idx):
            new_kps[vis_idx] = (moved[:, :2] + 0.5) / size
            new_vis[vis_idx] = 1.0
        flipped = self._replay_has_horizontal_flip(out.get("replay"))
        return out["image"], out["mask"], new_kps, new_vis, flipped

    def _drop_outside(self, keypoints, visibility, valid):
        """Keypoints outside the canvas or on padding become invisible."""
        size = self.image_size
        keypoints = keypoints.copy()
        visibility = visibility.copy()
        for i in np.flatnonzero(visibility > 0):
            x, y = keypoints[i]
            px, py = int(np.floor(x * size)), int(np.floor(y * size))
            if not (0 <= px < size and 0 <= py < size):
                visibility[i] = 0.0
            elif not valid[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2].any():
                # 1 px tolerance: labels on the image border stay visible.
                visibility[i] = 0.0
        keypoints[visibility < 1] = -1.0
        return keypoints, visibility

    def _relabel(self, keypoints, visibility, flipped):
        """Canonical relabeling (court symmetry) after augmentation.

        If the orientation cannot be determined (too few points), fall back
        to the column-flip permutation when a horizontal flip fired, which
        keeps the labeling's handedness.
        """
        perm = None
        if self.canonicalize_labels:
            keypoints, visibility, perm = canonicalize(
                keypoints, visibility, [self.image_size, self.image_size]
            )
        if perm is None and flipped:
            keypoints, visibility = permute_keypoints(keypoints, visibility, "col_flip")
        return keypoints, visibility

    @staticmethod
    def _replay_has_horizontal_flip(replay_node):
        """Recursively search an A.ReplayCompose replay log for an applied
        HorizontalFlip. Returns False for None (plain Compose / no replay).
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

    # ------------------------------------------------------------------
    def _generate_heatmaps(self, keypoints, visibility):
        """Gaussian heatmap per visible keypoint, centred (in heatmap index
        space) at ``kp * hm - 0.5``; invisible keypoints get all zeros.
        """
        hm = self.heatmap_size
        heatmaps = np.zeros((NUM_KEYPOINTS, hm, hm), dtype=np.float32)
        grid = np.arange(hm, dtype=np.float64)
        denom = 2.0 * self.sigma ** 2
        for i in np.flatnonzero(visibility > 0):
            cx = keypoints[i, 0] * hm - 0.5
            cy = keypoints[i, 1] * hm - 0.5
            gx = np.exp(-((grid - cx) ** 2) / denom)
            gy = np.exp(-((grid - cy) ** 2) / denom)
            heatmaps[i] = np.outer(gy, gx).astype(np.float32)
        return heatmaps

    def _generate_line_mask(self, keypoints, visibility):
        """(S, S) float32 court-line mask.

        If the visible keypoints support a homography, every painted court
        line of the template is projected (so lines whose end keypoints are
        off-screen still appear); otherwise only segments with both end
        points visible are drawn.
        """
        size = self.image_size
        mask = np.zeros((size, size), dtype=np.uint8)
        pts_px = keypoints * size  # continuous pixel coords

        segments = None
        if has_homography_support(visibility):
            segments = self._projected_segments(pts_px, visibility)
        if segments is None:
            segments = [
                (pts_px[a], pts_px[b]) for a, b in PAINTED_SEGMENTS
                if visibility[a] > 0 and visibility[b] > 0
            ]

        scale = 1 << _LINE_SHIFT
        for p0, p1 in segments:
            clipped = _clip_segment(p0, p1, size)
            if clipped is None:
                continue
            q0, q1 = clipped
            # Continuous coords -> cv2 pixel-index coords (centre at i).
            a = tuple(int(round((v - 0.5) * scale)) for v in q0)
            b = tuple(int(round((v - 0.5) * scale)) for v in q1)
            ok, a, b = cv2.clipLine((0, 0, size * scale, size * scale), a, b)
            if ok:
                cv2.line(mask, a, b, 255, self.line_thickness, cv2.LINE_AA, _LINE_SHIFT)
        return (mask.astype(np.float32) / 255.0 >= 0.5).astype(np.float32)

    def _projected_segments(self, pts_px, visibility):
        vis = visibility > 0
        tpl = COURT_KEYPOINTS_TEMPLATE
        H, _ = cv2.findHomography(tpl[vis], pts_px[vis], 0)
        if H is None:
            return None
        proj = _apply_h(H, tpl[vis])
        if proj is None or np.max(np.linalg.norm(proj - pts_px[vis], axis=1)) > _MAX_FIT_RESIDUAL_PX:
            return None
        # Orient H so the visible points have positive depth (w > 0).
        w_vis = (np.c_[tpl[vis], np.ones(vis.sum())] @ H[2])
        if np.all(w_vis < 0):
            H = -H
        elif not np.all(w_vis > 0):
            return None

        segments = []
        for a, b in PAINTED_SEGMENTS:
            seg = _segment_in_front(H, tpl[a], tpl[b])
            if seg is not None:
                segments.append(seg)
        return segments


def _apply_h(H, pts):
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    if np.any(np.abs(ph[:, 2]) < 1e-12):
        return None
    return ph[:, :2] / ph[:, 2:3]


def _segment_in_front(H, t0, t1, eps=1e-6):
    """Project template segment t0-t1, keeping only the part in front of
    the camera (homogeneous w > eps). Returns (p0, p1) image points or None.
    """
    w0 = H[2] @ np.r_[t0, 1.0]
    w1 = H[2] @ np.r_[t1, 1.0]
    scale = max(abs(w0), abs(w1))
    if scale <= 0:
        return None
    eps = eps * scale + 1e-12
    if w0 <= eps and w1 <= eps:
        return None
    if w0 <= eps or w1 <= eps:
        # Move the behind-camera end to where w = eps (w is affine in t).
        t = (eps - w0) / (w1 - w0)
        tc = t0 + t * (t1 - t0)
        if w0 <= eps:
            t0 = tc
        else:
            t1 = tc
    p = _apply_h(H, np.array([t0, t1]))
    if p is None:
        return None
    return p[0], p[1]


def _clip_segment(p0, p1, size, margin=4.0):
    """Liang-Barsky clip of a float segment to [-margin, size + margin]^2.

    Keeps coordinates bounded before the integer conversion for cv2.
    """
    lo, hi = -margin, size + margin
    x0, y0 = float(p0[0]), float(p0[1])
    dx, dy = float(p1[0]) - x0, float(p1[1]) - y0
    t_min, t_max = 0.0, 1.0
    for p, q in ((-dx, x0 - lo), (dx, hi - x0), (-dy, y0 - lo), (dy, hi - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)
        if t_min > t_max:
            return None
    return (x0 + t_min * dx, y0 + t_min * dy), (x0 + t_max * dx, y0 + t_max * dy)


def _main():
    parser = argparse.ArgumentParser(description="Inspect a CourtDataset split")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--images", default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--check-images", action="store_true",
                        help="load every sample to verify images resolve")
    args = parser.parse_args()
    ds = CourtDataset(args.annotations, args.images, image_size=args.image_size,
                      manifest=args.manifest, split=args.split)
    print(f"{len(ds)} samples; sources: {dict(Counter(ds.sources))}")
    print(f"groups: {len(set(it['group'] for it in ds.items))}")
    if args.check_images:
        missing = 0
        for i in range(len(ds)):
            try:
                ds._load_image(ds.items[i]["ann"], ds.items[i])
            except FileNotFoundError as e:
                missing += 1
                print(e)
        print(f"missing images: {missing}")


if __name__ == "__main__":
    _main()

"""Shared data-pipeline helpers: canonical labels, label QA, split groups, manifests.

Used by scripts/prepare_colab.py, scripts/qa_annotations.py and
scripts/canonicalize_annotations.py. Pure Python + numpy/cv2/PIL, no torch.

Annotation dicts follow the CVN format (see CLAUDE.md): normalized keypoints,
``image_size`` = [height, width], optional per-point lists ``occluded`` and
``keypoint_origin`` and optional ``source`` / ``group`` / ``canonical_perm``.
"""

import datetime as _dt
import hashlib
import json
import os
import re

import cv2
import numpy as np

from src.court_geometry import (
    COURT_KEYPOINTS_TEMPLATE,
    NUM_KEYPOINTS,
    SYMMETRY_PERMS,
    canonicalize,
    court_orientation,
    has_homography_support,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Per-keypoint list fields that must be relabeled together with the keypoints.
PER_POINT_FIELDS = ("occluded", "keypoint_origin")

# The symmetry group is the Klein four-group: encode each element as
# (col_flip bit, row_flip bit) so composition is XOR.
_PERM_BITS = {"identity": 0, "col_flip": 1, "row_flip": 2, "rot180": 3}
_BITS_PERM = {v: k for k, v in _PERM_BITS.items()}


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data, indent=2):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path_basename(p):
    """Basename that also handles Windows separators on any OS."""
    return re.split(r"[\\/]", str(p))[-1]


def image_shape(path):
    """(height, width) of an image as cv2.imread would load it (EXIF-rotated).

    Reads only the header via PIL; returns None if the file cannot be opened.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            try:
                orient = im.getexif().get(274, 1)
            except Exception:
                orient = 1
    except Exception:
        img = cv2.imread(path)
        return None if img is None else tuple(img.shape[:2])
    if orient in (5, 6, 7, 8):  # 90-degree rotations swap the axes
        w, h = h, w
    return int(h), int(w)


# ---------------------------------------------------------------------------
# Canonical labels
# ---------------------------------------------------------------------------

def compose_perm_names(first, then):
    """Name of applying symmetry ``first`` and then ``then`` (both names).

    None propagates (unknown orientation). The group is abelian, so order
    does not matter, but the signature documents intent.
    """
    if first is None or then is None:
        return then if first is None else first
    return _BITS_PERM[_PERM_BITS[first] ^ _PERM_BITS[then]]


def apply_perm_to_annotation(ann, perm_name):
    """Return a copy of ``ann`` relabeled by the symmetry ``perm_name``."""
    out = dict(ann)
    perm = SYMMETRY_PERMS[perm_name]
    out["keypoints"] = [list(map(float, ann["keypoints"][p])) for p in perm]
    out["visibility"] = [int(ann["visibility"][p]) for p in perm]
    for field in PER_POINT_FIELDS:
        vals = ann.get(field)
        if isinstance(vals, list) and len(vals) == NUM_KEYPOINTS:
            out[field] = [vals[p] for p in perm]
    return out


def canonicalize_annotation(ann):
    """Relabel an annotation into the canonical convention.

    Returns (new_ann, applied) where ``applied`` is the symmetry applied now
    ("identity" if already canonical) or None if the orientation cannot be
    determined (annotation returned unchanged apart from ``canonical_perm``).
    ``new_ann["canonical_perm"]`` is the cumulative relabeling relative to the
    original source labels (composed with any earlier ``canonical_perm``).
    """
    _, _, applied = canonicalize(ann["keypoints"], ann["visibility"], ann.get("image_size"))
    if applied is None:
        out = dict(ann)
        out.setdefault("canonical_perm", None)
        return out, None
    out = apply_perm_to_annotation(ann, applied)
    prev = ann.get("canonical_perm", "identity")
    out["canonical_perm"] = applied if prev is None else compose_perm_names(prev, applied)
    return out, applied


def is_canonical(ann):
    """True / False, or None if the orientation is undeterminable."""
    o = court_orientation(ann["keypoints"], ann["visibility"], ann.get("image_size"))
    if o is None:
        return None
    return o[0] > 0 and o[1] >= 0


# ---------------------------------------------------------------------------
# Label QA
# ---------------------------------------------------------------------------

def homography_residual(keypoints, visibility, image_size):
    """Reprojection residual of the best template->image homography, in pixels.

    Fits cv2.findHomography (least squares, all visible points) at the
    original resolution. Returns dict(max, rms, n, dof) or None when the
    visible points cannot determine a homography. With exactly 4 points the
    fit is exact (dof = 0) so the residual carries no information.
    """
    vis = np.asarray(visibility) > 0
    if not has_homography_support(vis):
        return None
    h, w = image_size
    pts = np.asarray(keypoints, dtype=np.float64) * np.array([w, h], dtype=np.float64)
    H, _ = cv2.findHomography(COURT_KEYPOINTS_TEMPLATE[vis], pts[vis], 0)
    if H is None:
        return None
    proj = cv2.perspectiveTransform(COURT_KEYPOINTS_TEMPLATE[vis][None], H)[0]
    err = np.linalg.norm(proj - pts[vis], axis=1)
    return {
        "max": float(err.max()),
        "rms": float(np.sqrt((err ** 2).mean())),
        "n": int(vis.sum()),
        "dof": int(2 * vis.sum() - 8),
        "worst_kp": int(np.flatnonzero(vis)[int(err.argmax())]),
    }


def qa_annotation(ann, image_path=None):
    """Per-annotation QA record (no thresholds applied)."""
    vis = np.asarray(ann.get("visibility", []))
    rec = {
        "n_visible": int((vis > 0).sum()),
        "image_size": list(ann.get("image_size", [])),
        "actual_size": None,
        "size_status": "unchecked",
        "handedness": None,
        "row_alignment": None,
        "perm_to_canonical": None,
        "residual_max": None,
        "residual_rms": None,
        "worst_kp": None,
    }
    if image_path is not None:
        if not os.path.isfile(image_path):
            rec["size_status"] = "missing_image"
        else:
            shape = image_shape(image_path)
            rec["actual_size"] = list(shape) if shape else None
            rec["size_status"] = size_status(rec["image_size"], shape)
    orient = court_orientation(ann["keypoints"], ann["visibility"], ann.get("image_size"))
    if orient is not None:
        rec["handedness"], rec["row_alignment"] = orient[0], round(orient[1], 4)
        _, _, rec["perm_to_canonical"] = canonicalize(
            ann["keypoints"], ann["visibility"], ann.get("image_size"))
    if ann.get("image_size"):
        r = homography_residual(ann["keypoints"], ann["visibility"], ann["image_size"])
        if r is not None:
            rec["residual_max"] = round(r["max"], 3)
            rec["residual_rms"] = round(r["rms"], 3)
            rec["worst_kp"] = r["worst_kp"]
    return rec


def size_status(declared, actual):
    """Compare a declared [h, w] with the actual (h, w)."""
    if actual is None:
        return "unreadable"
    if not declared or len(declared) != 2:
        return "missing"
    dh, dw = int(declared[0]), int(declared[1])
    ah, aw = int(actual[0]), int(actual[1])
    if (dh, dw) == (ah, aw):
        return "ok"
    if (dh, dw) == (aw, ah):
        return "transposed"
    if abs(dh / dw - ah / aw) < 0.01:
        return "rescaled"
    return "mismatch"


# ---------------------------------------------------------------------------
# Split groups
# ---------------------------------------------------------------------------

_RF_VIDEO_RE = re.compile(r"^(.+?)_image_\d+")


def roboflow_group(filename):
    """Video id of a Roboflow export file (prefix before ``_image_``)."""
    name = path_basename(filename)
    m = _RF_VIDEO_RE.match(name)
    return m.group(1) if m else os.path.splitext(name)[0]


_NAME_TIME_PATTERNS = [
    # IMG_20260815_201556(.jpg|_1.jpg), VID_20260815_200026_03.jpg
    (re.compile(r"(?:IMG|VID|PXL|MVIMG)[_-](\d{8})[_-](\d{6})"), "%Y%m%d%H%M%S"),
    # Screenshot 2026-08-13 130730
    (re.compile(r"(\d{4}-\d{2}-\d{2})[ _](\d{6})"), "%Y-%m-%d%H%M%S"),
    # photo_2026-09-10_17-37-33
    (re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})"), "%Y-%m-%d%H-%M-%S"),
]
# Camera-counter names without a timestamp (iPhone IMG_3850_01, IMG_3736_frame_00000)
_COUNTER_RE = re.compile(r"^IMG_(\d{4})(?:_|\.|$)")
_WHATSAPP_RE = re.compile(r"(\d{8})-WA\d+")


def capture_time(image_path, name=None):
    """Capture datetime from EXIF DateTimeOriginal/DateTime, else the filename."""
    if image_path and os.path.isfile(image_path):
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                exif = im.getexif()
                val = None
                try:
                    val = exif.get_ifd(0x8769).get(36867)
                except Exception:
                    val = None
                val = val or exif.get(306)
            if val:
                return _dt.datetime.strptime(str(val).strip()[:19], "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass
    name = name or path_basename(image_path or "")
    for rx, fmt in _NAME_TIME_PATTERNS:
        m = rx.search(name)
        if m:
            try:
                return _dt.datetime.strptime(m.group(1) + m.group(2), fmt)
            except ValueError:
                continue
    return None


def hand_label_groups(items, gap_minutes=10.0, counter_gap=20):
    """Assign session groups to hand-labeled images.

    Args:
        items: list of (key, image_path) pairs (key is returned in the map).
        gap_minutes: timestamped photos closer than this (in capture order)
            belong to the same session.
        counter_gap: images named IMG_NNNN* without a timestamp (phone video
            frames) are clustered by camera counter; a jump larger than this
            starts a new session.

    Returns:
        dict key -> group id. Groups are prefixed by how they were derived:
        ``session_<YYYYMMDD_HHMM>`` (timestamp), ``counter_IMG_<a>-<b>``
        (camera counter), ``single_<stem>`` (no usable metadata).
    """
    timed, counted, single = [], [], []
    for key, path in items:
        name = path_basename(path)
        t = capture_time(path, name)
        if t is not None:
            timed.append((t, key))
            continue
        m = _COUNTER_RE.match(name)
        if m:
            counted.append((int(m.group(1)), key))
            continue
        single.append((os.path.splitext(name)[0], key))

    groups = {}
    timed.sort()
    cur, last = None, None
    for t, key in timed:
        if last is None or (t - last).total_seconds() > gap_minutes * 60:
            cur = "session_" + t.strftime("%Y%m%d_%H%M")
        groups[key] = cur
        last = t

    counted.sort()
    clusters = []
    for n, key in counted:
        if clusters and n - clusters[-1][-1][0] <= counter_gap:
            clusters[-1].append((n, key))
        else:
            clusters.append([(n, key)])
    for cl in clusters:
        gid = f"counter_IMG_{cl[0][0]}-{cl[-1][0]}"
        for _, key in cl:
            groups[key] = gid

    for stem, key in single:
        groups[key] = "single_" + re.sub(r"\s+", "_", stem)
    return groups


def assign_groups_to_splits(group_sizes, ratios, seed=42):
    """Deterministically assign whole groups to splits.

    Greedy largest-first: each group goes to the split with the largest
    remaining deficit (target - current). Ties in size are broken by a
    seeded shuffle, ties in deficit by split order.

    Args:
        group_sizes: dict group -> number of samples
        ratios: dict split -> fraction (need not sum to 1; zeros allowed)
        seed: RNG seed

    Returns:
        dict group -> split
    """
    import random

    total = sum(group_sizes.values())
    norm = sum(ratios.values()) or 1.0
    target = {s: total * r / norm for s, r in ratios.items()}
    current = {s: 0 for s in ratios}
    order = sorted(group_sizes)  # stable base order independent of dict order
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -group_sizes[g])  # stable: shuffle breaks size ties
    out = {}
    split_names = list(ratios)
    for g in order:
        best = max(split_names, key=lambda s: (target[s] - current[s], -split_names.index(s))
                   if ratios[s] > 0 else (-float("inf"), 0))
        out[g] = best
        current[best] += group_sizes[g]
    return out


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

def thumb_vector(image, size=32):
    """Zero-mean, unit-norm vector of a size x size grayscale thumbnail."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 1e-9 else t


def cross_split_near_duplicates(vectors, splits, threshold=0.98):
    """Pairs (i, j, ncc) with i < j, different splits and NCC > threshold.

    Args:
        vectors: (N, D) array of thumb_vector rows
        splits: length-N list of split names
    """
    V = np.asarray(vectors, dtype=np.float64)
    splits = np.asarray(splits)
    pairs = []
    names = sorted(set(splits.tolist()))
    for a_i, a in enumerate(names):
        ia = np.flatnonzero(splits == a)
        for b in names[a_i + 1:]:
            ib = np.flatnonzero(splits == b)
            if len(ia) == 0 or len(ib) == 0:
                continue
            S = V[ia] @ V[ib].T
            for r, c in zip(*np.nonzero(S > threshold)):
                i, j = int(ia[r]), int(ib[c])
                pairs.append((min(i, j), max(i, j), float(S[r, c])))
    return sorted(pairs)

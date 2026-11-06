"""Convert Blender render metadata to CVN annotation format.

Reads from data/blender/raw/ (images + metadata JSONs), writes
CVN-format annotations and copies images to the output directories
(default data/blender/annotations/ and data/blender/images/).

Labels are canonicalized (see src.court_geometry.canonicalize) from the
in-frame keypoints when they determine a homography (so the output is stable
under re-canonicalization), otherwise from ALL projected keypoints that lie in
front of the camera, including those outside the frame, so the orientation is
always determinable for Blender renders. Visibility is the in-frame flag. If the metadata carries
``camera.keypoint_occluded`` (ray-cast occlusion), it is written as the
optional ``occluded`` list: occluded points keep visibility 1 (their location
is known) but are flagged.

Usage:
    python src/tools/blender_to_cvn.py [--min-visible 4]
        [--raw-dir data/blender/raw] [--out-images DIR] [--out-annotations DIR]
"""

import argparse
import json
import math
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.court_geometry import SYMMETRY_PERMS, canonicalize  # noqa: E402

NUM_KEYPOINTS = 30


def _euler_xyz_matrix(rx, ry, rz):
    """Rotation matrix for Blender's default XYZ Euler mode (R = Rz @ Ry @ Rx)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def keypoints_in_front(cam):
    """Boolean mask of keypoints in front of the camera (positive depth).

    Uses the camera pose (position + XYZ Euler rotation, camera looking down
    its local -Z) and the world keypoints. Points behind the camera project
    to the mirrored side of the image and must not be used to fit the
    orientation. Falls back to all-True if the pose is missing.
    """
    try:
        pos = np.asarray(cam["position"], dtype=np.float64)
        rot = cam["rotation"]
        pts = np.asarray(cam["keypoints_3d"], dtype=np.float64)
        if pts.shape != (NUM_KEYPOINTS, 3):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return np.ones(NUM_KEYPOINTS, dtype=bool)
    R = _euler_xyz_matrix(*rot[:3])
    local = (pts - pos) @ R  # rows: R^T (p - pos)
    return -local[:, 2] > 1e-6


def blender_group(meta):
    """Split group of a render: one per render seed (Blender data is train-only)."""
    if meta.get("seed") is not None:
        return f"blender_{meta['seed']}"
    if meta.get("image_index") is not None:
        return f"blender_{meta['image_index']}"
    return "blender"


def convert_metadata(meta, min_visible=4):
    """Convert a single Blender metadata dict to CVN annotation format.

    Args:
        meta: dict from blender_render.py metadata JSON
        min_visible: minimum in-frame keypoints to accept

    Returns:
        CVN annotation dict, or None if below min_visible threshold (or, in
        degenerate cases, if the court orientation cannot be determined).
    """
    cam = meta["camera"]
    res_x, res_y = meta["resolution"]
    image_size = [int(res_y), int(res_x)]

    # keypoints_2d are continuous pixel coords (pixel i spans [i, i+1]), so
    # dividing by the resolution gives pixel-centre-correct normalized coords.
    raw = np.asarray(cam["keypoints_2d"], dtype=np.float64) / np.array([res_x, res_y], dtype=np.float64)
    # visibility / in_frame = inside the image AND in front of the camera
    in_frame = np.asarray(cam.get("in_frame", cam["visibility"])) > 0
    occluded_raw = cam.get("keypoint_occluded", meta.get("keypoint_occluded"))
    occluded = (np.asarray(occluded_raw, dtype=bool) if occluded_raw is not None
                and len(occluded_raw) == NUM_KEYPOINTS else None)

    if int(in_frame.sum()) < min_visible:
        return None

    # Orientation: from the in-frame points when they determine a homography,
    # so the output is a fixed point of canonicalize() (which only sees the
    # labeled points) and re-canonicalizing downstream is a no-op. Otherwise
    # from every reliable projection (in front of the camera, finite, not
    # absurdly far away), which is always enough for a rendered court.
    _, _, perm_name = canonicalize(raw, in_frame, image_size)
    if perm_name is None:
        usable = (keypoints_in_front(cam) & np.isfinite(raw).all(axis=1)
                  & (np.abs(raw) < 1e3).all(axis=1))
        _, _, perm_name = canonicalize(np.where(usable[:, None], raw, -1.0), usable, image_size)
    if perm_name is None:
        return None
    perm = SYMMETRY_PERMS[perm_name]

    raw, in_frame = raw[perm], in_frame[perm]
    if occluded is not None:
        occluded = occluded[perm]

    keypoints, visibility = [], []
    for i in range(NUM_KEYPOINTS):
        if in_frame[i]:
            keypoints.append([float(raw[i, 0]), float(raw[i, 1])])
            visibility.append(1)
        else:
            keypoints.append([-1.0, -1.0])
            visibility.append(0)

    vis_kps = [i for i in range(NUM_KEYPOINTS) if visibility[i]]
    xs = [keypoints[i][0] for i in vis_kps]
    ys = [keypoints[i][1] for i in vis_kps]
    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

    ann = {
        "image_path": os.path.basename(meta["image_file"]),
        "image_size": image_size,
        "keypoints": keypoints,
        "visibility": visibility,
        "bounding_box": bbox,
        "source": "blender",
        "group": blender_group(meta),
        "canonical_perm": perm_name,
    }
    if occluded is not None:
        # Only meaningful for in-frame points; off-frame points are not labeled.
        ann["occluded"] = [bool(occluded[i]) and bool(visibility[i]) for i in range(NUM_KEYPOINTS)]
    return ann


def convert_all(raw_dir, out_images, out_annotations, min_visible=4):
    """Convert all Blender raw output to CVN format.

    Args:
        raw_dir: path to data/blender/raw/
        out_images: output image directory
        out_annotations: output annotation directory
        min_visible: minimum in-frame keypoints

    Returns:
        number of images converted
    """
    raw_images = os.path.join(raw_dir, "images")
    raw_metadata = os.path.join(raw_dir, "metadata")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_annotations, exist_ok=True)

    count = 0
    for fname in sorted(os.listdir(raw_metadata)):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(raw_metadata, fname), encoding="utf-8") as f:
            meta = json.load(f)

        ann = convert_metadata(meta, min_visible=min_visible)
        if ann is None:
            continue

        img_name = os.path.basename(meta["image_file"])
        src_img = os.path.join(raw_images, img_name)
        if not os.path.isfile(src_img):
            continue

        shutil.copy2(src_img, os.path.join(out_images, img_name))

        stem = os.path.splitext(img_name)[0]
        ann_path = os.path.join(out_annotations, f"{stem}.json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(ann, f, indent=2)

        count += 1

    return count


def main():
    blender_dir = os.path.join(ROOT, "data", "blender")
    parser = argparse.ArgumentParser(description="Convert Blender metadata to CVN annotations")
    parser.add_argument("--min-visible", type=int, default=4)
    parser.add_argument("--raw-dir", default=os.path.join(blender_dir, "raw"))
    parser.add_argument("--out-images", default=os.path.join(blender_dir, "images"))
    parser.add_argument("--out-annotations", default=os.path.join(blender_dir, "annotations"))
    args = parser.parse_args()

    if not os.path.isdir(args.raw_dir):
        print(f"No raw directory found at {args.raw_dir}")
        print("Run blender_render.py first.")
        sys.exit(1)

    print("Converting Blender raw output to CVN format...")
    print(f"  Raw:         {args.raw_dir}")
    print(f"  Images out:  {args.out_images}")
    print(f"  Ann out:     {args.out_annotations}")
    print(f"  Min visible: {args.min_visible}")

    count = convert_all(args.raw_dir, args.out_images, args.out_annotations, args.min_visible)
    print(f"\nDone: {count} images converted")


if __name__ == "__main__":
    main()

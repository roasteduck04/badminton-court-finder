"""Convert Blender render metadata to CVN annotation format.

Reads from data/blender/raw/ (images + metadata JSONs), writes
CVN-format annotations to data/blender/annotations/ and copies
images to data/blender/images/.

Usage:
    python src/tools/blender_to_cvn.py [--min-visible 4]
"""

import json
import os
import shutil
import sys

NUM_KEYPOINTS = 30


def convert_metadata(meta, min_visible=4):
    """Convert a single Blender metadata dict to CVN annotation format.

    Args:
        meta: dict from blender_render.py metadata JSON
        min_visible: minimum visible keypoints to accept

    Returns:
        CVN annotation dict, or None if below min_visible threshold
    """
    cam = meta["camera"]
    resolution = meta["resolution"]
    res_x, res_y = resolution

    raw_kp = cam["keypoints_2d"]
    raw_vis = cam["visibility"]

    keypoints = []
    visibility = []

    for i in range(NUM_KEYPOINTS):
        if raw_vis[i]:
            nx = raw_kp[i][0] / res_x
            ny = raw_kp[i][1] / res_y
            keypoints.append([nx, ny])
            visibility.append(1)
        else:
            keypoints.append([-1.0, -1.0])
            visibility.append(0)

    n_visible = sum(visibility)
    if n_visible < min_visible:
        return None

    vis_kps = [i for i in range(NUM_KEYPOINTS) if visibility[i]]
    xs = [keypoints[i][0] for i in vis_kps]
    ys = [keypoints[i][1] for i in vis_kps]
    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

    return {
        "image_path": meta["image_file"],
        "image_size": [res_y, res_x],
        "keypoints": keypoints,
        "visibility": visibility,
        "bounding_box": bbox,
    }


def convert_all(raw_dir, out_images, out_annotations, min_visible=4):
    """Convert all Blender raw output to CVN format.

    Args:
        raw_dir: path to data/blender/raw/
        out_images: path to data/blender/images/
        out_annotations: path to data/blender/annotations/
        min_visible: minimum visible keypoints

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

        with open(os.path.join(raw_metadata, fname)) as f:
            meta = json.load(f)

        ann = convert_metadata(meta, min_visible=min_visible)
        if ann is None:
            continue

        img_name = meta["image_file"]
        src_img = os.path.join(raw_images, img_name)
        if not os.path.isfile(src_img):
            continue

        shutil.copy2(src_img, os.path.join(out_images, img_name))

        stem = os.path.splitext(img_name)[0]
        ann_path = os.path.join(out_annotations, f"{stem}.json")
        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2)

        count += 1

    return count


def main():
    min_visible = 4
    if "--min-visible" in sys.argv:
        idx = sys.argv.index("--min-visible")
        min_visible = int(sys.argv[idx + 1])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(script_dir))
    blender_dir = os.path.join(root, "data", "blender")

    raw_dir = os.path.join(blender_dir, "raw")
    out_images = os.path.join(blender_dir, "images")
    out_annotations = os.path.join(blender_dir, "annotations")

    if not os.path.isdir(raw_dir):
        print(f"No raw directory found at {raw_dir}")
        print("Run blender_render.py first.")
        sys.exit(1)

    print("Converting Blender raw output to CVN format...")
    print(f"  Raw:         {raw_dir}")
    print(f"  Images out:  {out_images}")
    print(f"  Ann out:     {out_annotations}")
    print(f"  Min visible: {min_visible}")

    count = convert_all(raw_dir, out_images, out_annotations, min_visible)
    print(f"\nDone: {count} images converted")


if __name__ == "__main__":
    main()

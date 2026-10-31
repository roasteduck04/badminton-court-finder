"""Tests for src.training.dataset.CourtDataset.

Also provides synthetic-court helpers (``make_court_annotation``,
``write_sample``) reused by test_augmentation.py and test_training.py.
"""

import json
import os

import cv2
import numpy as np
import pytest
import torch

from src.court_geometry import (
    COURT_KEYPOINTS_TEMPLATE,
    NUM_KEYPOINTS,
    canonicalize,
    get_court_lines,
)
from src.training.dataset import CourtDataset, infer_source, load_manifest_split

# Template corners K0, K25, K29, K4 -> image pixels (continuous coords).
END_ON_VIEW = {"size_wh": (640, 480),
               "corners": [(60, 450), (200, 120), (440, 120), (580, 450)]}
SIDE_VIEW = {"size_wh": (640, 480),
             "corners": [(40, 380), (600, 380), (500, 200), (140, 200)]}


def view_homography(view):
    src = COURT_KEYPOINTS_TEMPLATE[[0, 25, 29, 4]].astype(np.float32)
    dst = np.array(view["corners"], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def make_court_annotation(view, hidden=()):
    """(bgr_image, annotation dict) for a synthetic court in ``view``.

    Labels are canonicalized, white court lines are drawn on a green floor.
    """
    w, h = view["size_wh"]
    H = view_homography(view)
    pts = cv2.perspectiveTransform(
        COURT_KEYPOINTS_TEMPLATE.reshape(-1, 1, 2).astype(np.float64), H
    ).reshape(-1, 2)
    image = np.full((h, w, 3), (40, 140, 40), dtype=np.uint8)
    for a, b in get_court_lines():
        if (a, b) == (12, 17):
            continue
        p0 = tuple(int(round((v - 0.5) * 16)) for v in pts[a])
        p1 = tuple(int(round((v - 0.5) * 16)) for v in pts[b])
        cv2.line(image, p0, p1, (255, 255, 255), 2, cv2.LINE_AA, 4)
    kps = pts / np.array([w, h])
    vis = ((kps >= 0) & (kps < 1)).all(axis=1).astype(float)
    for i in hidden:
        vis[i] = 0
    kps[vis == 0] = -1
    kps, vis, _ = canonicalize(kps, vis, [h, w])
    ann = {
        "image_size": [h, w],
        "keypoints": kps.tolist(),
        "visibility": vis.astype(int).tolist(),
        "bounding_box": [0.5, 0.5, 1.0, 1.0],
    }
    return image, ann


def write_sample(img_dir, ann_dir, name, image, ann, absolute_path=False, **extra):
    img_path = os.path.join(str(img_dir), f"{name}.png")
    cv2.imwrite(img_path, image)
    ann = dict(ann, image_path=img_path if absolute_path else f"{name}.png", **extra)
    with open(os.path.join(str(ann_dir), f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(ann, f)
    return img_path


@pytest.fixture
def court_dirs(tmp_path):
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()
    return img_dir, ann_dir


@pytest.fixture
def mock_dataset(court_dirs):
    img_dir, ann_dir = court_dirs
    for i, view in enumerate([END_ON_VIEW, SIDE_VIEW, END_ON_VIEW]):
        image, ann = make_court_annotation(view)
        write_sample(img_dir, ann_dir, f"frame_{i:03d}", image, ann)
    return str(img_dir), str(ann_dir)


# ----------------------------------------------------------------------
# Sample structure
# ----------------------------------------------------------------------
def test_dataset_length(mock_dataset):
    img_dir, ann_dir = mock_dataset
    assert len(CourtDataset(ann_dir, img_dir, image_size=128)) == 3


def test_dataset_item_keys_shapes_ranges(mock_dataset):
    img_dir, ann_dir = mock_dataset
    ds = CourtDataset(ann_dir, img_dir, image_size=640)
    s = ds[0]
    assert s["image"].shape == (7, 640, 640) and s["image"].dtype == torch.float32
    assert s["heatmaps"].shape == (30, 160, 160)
    assert s["keypoints"].shape == (30, 2)
    assert s["visibility"].shape == (30,)
    assert s["mask"].shape == (1, 640, 640)
    assert s["valid"].shape == (1, 640, 640)
    assert 0.0 <= s["image"].min() and s["image"].max() <= 1.0
    assert 0.0 <= s["heatmaps"].min() and s["heatmaps"].max() <= 1.0
    assert set(torch.unique(s["mask"]).tolist()) <= {0.0, 1.0}
    assert set(torch.unique(s["valid"]).tolist()) <= {0.0, 1.0}
    assert set(torch.unique(s["visibility"]).tolist()) <= {0.0, 1.0}
    assert isinstance(s["source"], str) and isinstance(s["group"], str)
    vis = s["visibility"] > 0
    assert torch.all((s["keypoints"][vis] >= 0) & (s["keypoints"][vis] < 1))
    assert torch.all(s["keypoints"][~vis] == -1)


def test_rgb_only_channels(mock_dataset):
    img_dir, ann_dir = mock_dataset
    ds7 = CourtDataset(ann_dir, img_dir, image_size=128)
    ds3 = CourtDataset(ann_dir, img_dir, image_size=128, in_channels=3)
    s3, s7 = ds3[0], ds7[0]
    assert s3["image"].shape == (3, 128, 128)
    torch.testing.assert_close(s3["image"], s7["image"][:3])


def test_letterbox_padding_and_valid_mask(mock_dataset):
    """640x480 image -> 640 canvas with 80 px bands of padding top/bottom."""
    img_dir, ann_dir = mock_dataset
    s = CourtDataset(ann_dir, img_dir, image_size=640)[0]
    valid = s["valid"][0]
    assert valid[:80].sum() == 0 and valid[560:].sum() == 0
    assert valid[80:560].all()
    assert s["image"][:, :80].abs().sum() == 0  # padding is black in every channel
    assert s["mask"][0][~valid.bool()].sum() == 0


def test_invisible_keypoints(court_dirs):
    img_dir, ann_dir = court_dirs
    image, ann = make_court_annotation(END_ON_VIEW, hidden=range(2, 30))
    write_sample(img_dir, ann_dir, "frame", image, ann)
    s = CourtDataset(str(ann_dir), str(img_dir), image_size=128)[0]
    assert s["visibility"][2] == 0 and s["visibility"][0] == 1
    assert s["heatmaps"][2].sum() == 0
    assert s["heatmaps"][0].sum() > 0
    assert torch.all(s["keypoints"][2] == -1)


# ----------------------------------------------------------------------
# Coordinate conventions
# ----------------------------------------------------------------------
@pytest.mark.parametrize("orig_wh", [(640, 480), (1280, 960), (480, 640)])
def test_dot_lands_at_heatmap_peak(court_dirs, orig_wh):
    """A dot drawn at a keypoint lands on canvas pixel floor(kp*S), and the
    heatmap target peaks at index floor(kp*hm) (pixel-centre convention).
    """
    img_dir, ann_dir = court_dirs
    w, h = orig_wh
    scale = 640 / max(w, h)
    rng = np.random.default_rng(0)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    block = int(round(1 / scale))  # one canvas pixel in original pixels
    kps = np.full((30, 2), -1.0)
    vis = np.zeros(30)
    for k in range(5):
        # Pick an original-pixel block aligned with canvas pixels.
        cx, cy = rng.integers(5, int(w * scale) - 5), rng.integers(5, int(h * scale) - 5)
        x0, y0 = cx * block, cy * block
        image[y0:y0 + block, x0:x0 + block] = 255
        kps[k] = [(x0 + block / 2) / w, (y0 + block / 2) / h]
        vis[k] = 1
    ann = {"image_size": [h, w], "keypoints": kps.tolist(),
           "visibility": vis.astype(int).tolist(), "bounding_box": [0, 0, 0, 0]}
    write_sample(img_dir, ann_dir, "dots", image, ann)
    ds = CourtDataset(str(ann_dir), str(img_dir), image_size=640, heatmap_size=160,
                      canonicalize_labels=False)
    s = ds[0]
    gray = s["image"][3].numpy()
    for k in range(5):
        x, y = s["keypoints"][k].numpy()
        px, py = int(np.floor(x * 640)), int(np.floor(y * 640))
        assert gray[py, px] > 0.9, (k, px, py)
        hm = s["heatmaps"][k].numpy()
        iy, ix = np.unravel_index(np.argmax(hm), hm.shape)
        assert (ix, iy) == (int(np.floor(x * 160)), int(np.floor(y * 160)))


def test_heatmap_gaussian_centre_and_sigma(mock_dataset):
    img_dir, ann_dir = mock_dataset
    ds = CourtDataset(ann_dir, img_dir, image_size=128, heatmap_size=32, sigma=2.0)
    kps = np.full((30, 2), -1.0)
    vis = np.zeros(30)
    kps[0] = [(10 + 0.5) / 32, (20 + 0.5) / 32]  # exactly on heatmap pixel (10, 20)
    vis[0] = 1
    hm = ds._generate_heatmaps(kps, vis)
    assert hm[0, 20, 10] == pytest.approx(1.0)
    assert hm[0, 20, 12] == pytest.approx(np.exp(-4 / 8), rel=1e-5)
    assert hm[1:].sum() == 0


# ----------------------------------------------------------------------
# Segmentation target
# ----------------------------------------------------------------------
def _mask_at(mask, pt_norm, size):
    x, y = int(np.floor(pt_norm[0] * size)), int(np.floor(pt_norm[1] * size))
    return mask[y - 1:y + 2, x - 1:x + 2].max()


def test_seg_mask_projects_all_lines_and_skips_net_segment(court_dirs):
    img_dir, ann_dir = court_dirs
    # Hide the whole far half: lines there must still be drawn via homography.
    image, ann = make_court_annotation(END_ON_VIEW, hidden=range(15, 30))
    write_sample(img_dir, ann_dir, "half", image, ann)
    full_image, full_ann = make_court_annotation(END_ON_VIEW)
    ds = CourtDataset(str(ann_dir), str(img_dir), image_size=640)
    s = ds[0]
    mask = s["mask"][0].numpy()
    # Ground-truth positions of all 30 points in canvas coords.
    from src.preprocessing.letterbox import keypoints_to_letterbox, letterbox
    _, info = letterbox(full_image, 640)
    full = keypoints_to_letterbox(np.array(full_ann["keypoints"]), info)
    k = full
    mid = lambda a, b: (k[a] + k[b]) / 2  # noqa: E731
    assert _mask_at(mask, mid(20, 25), 640) == 1  # far sideline, both ends hidden
    assert _mask_at(mask, mid(26, 27), 640) == 1  # far baseline
    assert _mask_at(mask, mid(2, 7), 640) == 1    # painted centre line
    # Net-crossing centre segment K12-K17 is virtual: not in the mask.
    assert _mask_at(mask, mid(12, 17), 640) == 0


def test_seg_mask_fallback_without_homography(court_dirs):
    img_dir, ann_dir = court_dirs
    image, ann = make_court_annotation(END_ON_VIEW, hidden=[i for i in range(30) if i not in (0, 1, 2)])
    write_sample(img_dir, ann_dir, "row", image, ann)
    s = CourtDataset(str(ann_dir), str(img_dir), image_size=640)[0]
    mask = s["mask"][0].numpy()
    k = s["keypoints"].numpy()
    assert _mask_at(mask, (k[0] + k[1]) / 2, 640) == 1
    # Only the baseline segments are drawn.
    assert mask.sum() < 0.01 * mask.size


def test_seg_mask_matches_drawn_lines(mock_dataset):
    """The projected mask overlaps the white lines painted in the image."""
    img_dir, ann_dir = mock_dataset
    s = CourtDataset(ann_dir, img_dir, image_size=640, line_thickness=5)[1]
    rgb = s["image"][:3].numpy()
    white = (rgb.min(axis=0) > 0.6)
    mask = s["mask"][0].numpy() > 0
    assert white.sum() > 0
    assert (white & mask).sum() / white.sum() > 0.9


# ----------------------------------------------------------------------
# Data sources: directory, absolute paths, manifest
# ----------------------------------------------------------------------
def test_absolute_image_path_and_source_inference(court_dirs, tmp_path):
    img_dir, ann_dir = court_dirs
    image, ann = make_court_annotation(SIDE_VIEW)
    write_sample(img_dir, ann_dir, "bl_blender_0001", image, ann, absolute_path=True)
    write_sample(img_dir, ann_dir, "mine_IMG_1", image, ann)
    ds = CourtDataset(str(ann_dir), None, image_size=64)
    assert sorted(ds.sources) == ["blender", "manual"]
    assert ds[0]["image"].shape == (7, 64, 64)
    assert ds[1]["image"].shape == (7, 64, 64)  # basename found in ../images
    assert infer_source("ds_output_image_3_png.rf.abc.json") == "roboflow"
    assert infer_source("whatever.json") == "unknown"


def test_exclude_sources(court_dirs):
    img_dir, ann_dir = court_dirs
    image, ann = make_court_annotation(SIDE_VIEW)
    write_sample(img_dir, ann_dir, "a", image, ann, source="blender")
    write_sample(img_dir, ann_dir, "b", image, ann, source="roboflow", group="vid1")
    ds = CourtDataset(str(ann_dir), str(img_dir), image_size=64, exclude_sources=["blender"])
    assert len(ds) == 1
    s = ds[0]
    assert s["source"] == "roboflow" and s["group"] == "vid1"


def _write_manifest(root):
    """prepare_colab.py layout: <root>/<split>/{annotations,images} + manifest."""
    splits = {}
    for split, specs in {"train": [("bl_0", "blender", "g0"), ("rf_1", "roboflow", "v1")],
                         "val": [("mine_2", "manual", "s1")]}.items():
        a_dir, i_dir = root / split / "annotations", root / split / "images"
        a_dir.mkdir(parents=True)
        i_dir.mkdir(parents=True)
        splits[split] = []
        for name, source, group in specs:
            image, ann = make_court_annotation(END_ON_VIEW)
            write_sample(i_dir, a_dir, name, image, ann, source=source, group=group)
            splits[split].append({"file": f"{name}.json", "image": f"{name}.png",
                                  "source": source, "group": group})
    path = root / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"format_version": 1, "splits": splits, "dropped": []}, f)
    return str(path)


def test_manifest_split(tmp_path):
    manifest = _write_manifest(tmp_path / "v2")
    items = load_manifest_split(manifest, "train")
    assert [it["source"] for it in items] == ["blender", "roboflow"]
    ds = CourtDataset(manifest=manifest, split="train", image_size=64)
    assert len(ds) == 2 and ds[1]["source"] == "roboflow" and ds[1]["group"] == "v1"
    ds = CourtDataset(manifest=manifest, split="train", image_size=64, exclude_sources=["blender"])
    assert len(ds) == 1
    val = CourtDataset(manifest=manifest, split="val", image_size=64)
    assert len(val) == 1 and val[0]["source"] == "manual"
    with pytest.raises(KeyError):
        load_manifest_split(manifest, "test_real")


def test_dataset_with_train_transform(mock_dataset):
    from src.preprocessing.augmentation import get_train_transforms

    img_dir, ann_dir = mock_dataset
    ds = CourtDataset(ann_dir, img_dir, transform=get_train_transforms(128, seed=0), image_size=128)
    for i in range(len(ds)):
        s = ds[i]
        assert s["image"].shape == (7, 128, 128)
        assert s["heatmaps"].shape == (30, 32, 32)
        vis = s["visibility"] > 0
        assert torch.all(s["keypoints"][~vis] == -1)
        assert torch.all((s["keypoints"][vis] >= 0) & (s["keypoints"][vis] < 1))
        assert NUM_KEYPOINTS == s["keypoints"].shape[0]


def test_dataset_with_val_transform_is_noop(mock_dataset):
    from src.preprocessing.augmentation import get_val_transforms

    img_dir, ann_dir = mock_dataset
    plain = CourtDataset(ann_dir, img_dir, image_size=128)[0]
    noop = CourtDataset(ann_dir, img_dir, transform=get_val_transforms(128), image_size=128)[0]
    torch.testing.assert_close(plain["image"], noop["image"])
    torch.testing.assert_close(plain["keypoints"], noop["keypoints"])

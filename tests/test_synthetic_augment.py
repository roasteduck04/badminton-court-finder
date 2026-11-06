"""Tests for src/tools/synthetic_augment.py (keypoint/image consistency)."""

import json
import os

import cv2
import numpy as np
import pytest

from src.tools.synthetic_augment import (
    DEFAULT_PROBS, apply_chromatic_aberration, apply_defocus, apply_flicker_banding,
    apply_interlace, apply_jpeg, apply_motion_blur, apply_noise, apply_vignette,
    augment_dataset, augment_sample, distort_brown_conrady, make_rng, sample_distortion,
)

H, W = 360, 480


def _dot_image(kps_norm, radius=2):
    """Grey image with white dots at the keypoints (pixel-centre convention)."""
    img = np.full((H, W, 3), 60, np.uint8)
    for x, y in kps_norm:
        cx, cy = x * W - 0.5, y * H - 0.5
        # Anti-aliased dot rendered at sub-pixel position via supersampling
        big = np.zeros((H * 8, W * 8), np.uint8)
        cv2.circle(big, (int(round((cx + 0.5) * 8 - 0.5)), int(round((cy + 0.5) * 8 - 0.5))),
                   radius * 8, 255, -1)
        small = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
        img = np.maximum(img, np.repeat(small[..., None], 3, axis=2))
    return img


def _centroid_near(img, x_norm, y_norm, win=12):
    gray = img[..., 0].astype(np.float64) - 60.0
    gray[gray < 0] = 0
    cx, cy = x_norm * W - 0.5, y_norm * H - 0.5
    x0, x1 = int(max(0, cx - win)), int(min(W, cx + win + 1))
    y0, y1 = int(max(0, cy - win)), int(min(H, cy + win + 1))
    patch = gray[y0:y1, x0:x1]
    assert patch.sum() > 0, "dot vanished"
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return (xx * patch).sum() / patch.sum(), (yy * patch).sum() / patch.sum()


KPS = np.array([[0.5, 0.5], [0.2, 0.25], [0.8, 0.3], [0.15, 0.8], [0.85, 0.85],
                [0.6, 0.1], [0.4, 0.9]])


@pytest.mark.parametrize("params", [
    {"f": 400.0, "k1": -0.30, "k2": 0.02, "p1": 0.001, "p2": -0.001},   # strong barrel
    {"f": 300.0, "k1": 0.15, "k2": 0.0, "p1": 0.0, "p2": 0.0},          # pincushion
    {"f": 350.0, "k1": -0.15, "k2": -0.02, "p1": -0.002, "p2": 0.002},
])
def test_distortion_moves_dots_with_keypoints(params):
    img = _dot_image(KPS)
    out, kps, vis, _ = distort_brown_conrady(img, KPS, np.ones(len(KPS)), params)
    assert out.shape == img.shape
    moved = 0
    for k in range(len(KPS)):
        if not vis[k]:
            continue
        mx, my = _centroid_near(out, *kps[k])
        ex, ey = kps[k][0] * W - 0.5, kps[k][1] * H - 0.5
        assert abs(mx - ex) < 0.6 and abs(my - ey) < 0.6, (k, mx, my, ex, ey)
        moved = max(moved, np.hypot(*(kps[k] - KPS[k]) * [W, H]))
    assert vis.sum() >= 3
    assert moved > 2.0  # the transform actually did something


def test_keypoints_leaving_frame_become_invisible():
    kps = np.array([[0.005, 0.005], [0.5, 0.5], [0.995, 0.5]])
    params = {"f": 300.0, "k1": -0.35, "k2": 0.0, "p1": 0.0, "p2": 0.0}
    _, out_kps, vis, scale = distort_brown_conrady(_dot_image(kps), kps, np.ones(3), params)
    assert scale > 1.0                 # barrel distortion crops to avoid black borders
    assert vis[1] == 1 and np.allclose(out_kps[1], [0.5, 0.5], atol=1e-3)
    assert vis[0] == 0 and list(out_kps[0]) == [-1.0, -1.0]


def test_invisible_input_keypoints_stay_invisible():
    kps = np.array([[-1.0, -1.0], [0.5, 0.5]])
    params = {"f": 400.0, "k1": -0.2, "k2": 0.0, "p1": 0.0, "p2": 0.0}
    _, out_kps, vis, _ = distort_brown_conrady(_dot_image(kps[1:]), kps, np.array([0, 1]), params)
    assert vis[0] == 0 and list(out_kps[0]) == [-1.0, -1.0]


def test_no_black_border_after_distortion():
    img = np.full((H, W, 3), 200, np.uint8)
    params = {"f": 300.0, "k1": -0.3, "k2": 0.0, "p1": 0.0, "p2": 0.0}
    out, _, _, _ = distort_brown_conrady(img, KPS, np.ones(len(KPS)), params)
    assert out.min() >= 199


def test_sampled_distortion_is_valid():
    rng = make_rng(0, "x")
    for _ in range(20):
        d = sample_distortion(rng, (H, W))
        assert -0.35 <= d["k1"] <= 0.15
        img = _dot_image(KPS)
        out, kps, vis, _ = distort_brown_conrady(img, KPS, np.ones(len(KPS)), d)
        for k in np.nonzero(vis)[0]:
            mx, my = _centroid_near(out, *kps[k])
            assert np.hypot(mx - (kps[k][0] * W - 0.5), my - (kps[k][1] * H - 0.5)) < 1.0


def test_photometric_ops_keep_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    for out in (apply_vignette(img, 0.4), apply_defocus(img, 2.5),
                apply_motion_blur(img, 9, 30.0), apply_noise(img, rng, 4.0, 0.2),
                apply_jpeg(img, 40), apply_chromatic_aberration(img, 1.5),
                apply_flicker_banding(img, 4.0, 0.1, 0.3), apply_interlace(img, 2.5)):
        assert out.shape == img.shape and out.dtype == np.uint8
    v = apply_vignette(np.full((H, W, 3), 200, np.uint8), 0.5)
    assert v[H // 2, W // 2, 0] >= 199 and v[0, 0, 0] < 120


def test_chromatic_aberration_keeps_green_and_centre():
    img = _dot_image(KPS)
    out = apply_chromatic_aberration(img, 2.0)
    assert np.array_equal(out[..., 1], img[..., 1])          # keypoints follow green
    corner = np.zeros((H, W, 3), np.uint8)
    cv2.circle(corner, (20, 20), 3, (255, 255, 255), -1)
    shifted = apply_chromatic_aberration(corner, 2.0)
    assert not np.array_equal(shifted[..., 2], corner[..., 2])  # red moves near the corner


def test_flicker_bands_vary_by_row_only():
    flat = np.full((H, W, 3), 128, np.uint8)
    out = apply_flicker_banding(flat, 3.0, 0.1, 0.0).astype(int)
    assert (out.std(axis=1) == 0).all()                       # constant along each row
    assert out[:, 0, 0].max() - out[:, 0, 0].min() >= 20     # +-10% of 128


def test_interlace_changes_only_odd_rows():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    out = apply_interlace(img, 3.0)
    assert np.array_equal(out[0::2], img[0::2]) and not np.array_equal(out[1::2], img[1::2])


def _ann():
    return {"image_path": "img.png", "image_size": [H, W], "keypoints": KPS.tolist() + [[-1, -1]] * 23,
            "visibility": [1] * 7 + [0] * 23, "bounding_box": [0, 0, 1, 1], "source": "blender",
            "occluded": [False] * 30}


def test_augment_sample_deterministic_and_consistent():
    img = _dot_image(KPS)
    probs = {"distort": 1.0, "vignette": 1.0, "blur": 0.0, "noise": 1.0, "jpeg": 1.0}
    a1, n1, p1 = augment_sample(img, _ann(), make_rng(3, "img"), probs)
    a2, n2, p2 = augment_sample(img, _ann(), make_rng(3, "img"), probs)
    a3, _, _ = augment_sample(img, _ann(), make_rng(4, "img"), probs)
    assert np.array_equal(a1, a2) and n1 == n2 and p1 == p2
    assert not np.array_equal(a1, a3)
    assert len(n1["keypoints"]) == 30 and len(n1["visibility"]) == 30
    assert n1["source"] == "blender" and "distortion" in n1["augment"]
    for k in range(30):
        if not n1["visibility"][k]:
            assert n1["keypoints"][k] == [-1.0, -1.0]


def test_identity_when_all_probabilities_zero():
    img = _dot_image(KPS)
    out, ann, params = augment_sample(img, _ann(), make_rng(0, "img"),
                                      {k: 0.0 for k in DEFAULT_PROBS})
    assert params == {} and np.array_equal(out, img)
    assert np.allclose(np.array(ann["keypoints"])[:7], KPS)


def test_cli_dataset_roundtrip(tmp_path):
    img_dir, ann_dir, out = tmp_path / "images", tmp_path / "annotations", tmp_path / "out"
    img_dir.mkdir()
    ann_dir.mkdir()
    cv2.imwrite(str(img_dir / "img.png"), _dot_image(KPS))
    with open(ann_dir / "img.json", "w", encoding="utf-8") as f:
        json.dump(_ann(), f)
    n = augment_dataset(str(img_dir), str(ann_dir), str(out), seed=1, copies=2)
    assert n == 2
    assert sorted(os.listdir(out / "images")) == ["img_aug0.png", "img_aug1.png"]
    with open(out / "annotations" / "img_aug0.json", encoding="utf-8") as f:
        ann = json.load(f)
    assert ann["image_path"] == "img_aug0.png"
    assert ann["image_size"] == [H, W]

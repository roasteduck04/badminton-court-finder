"""Tests for the augmentation pipeline and label handling under augmentation.

Covers: albumentations keypoint coordinate convention (pixel-centre),
canonical relabeling after flips/affine (including side views, where the
fixed FLIP_PAIRS swap is wrong), the FLIP_PAIRS fallback, and per-worker
RNG seeding.
"""

import albumentations as A
import numpy as np
import pytest
import torch

from src.court_geometry import FLIP_PAIRS, canonicalize, has_homography_support, permute_keypoints
from src.preprocessing.augmentation import (
    get_train_transforms,
    get_val_transforms,
    reseed_transform,
    seed_worker,
)
from src.training.dataset import CourtDataset
from tests.test_dataset import END_ON_VIEW, SIDE_VIEW, make_court_annotation, write_sample

SIZE = 128


def _dataset(tmp_path, views, transform, size=SIZE, **kwargs):
    img_dir, ann_dir = tmp_path / "images", tmp_path / "annotations"
    img_dir.mkdir(exist_ok=True)
    ann_dir.mkdir(exist_ok=True)
    for i, view in enumerate(views):
        image, ann = make_court_annotation(view)
        write_sample(img_dir, ann_dir, f"s{i}", image, ann)
    return CourtDataset(str(ann_dir), str(img_dir), transform=transform, image_size=size, **kwargs)


def _forced(*transforms):
    return A.ReplayCompose(list(transforms), keypoint_params=A.KeypointParams(
        format="xy", remove_invisible=False))


# ----------------------------------------------------------------------
# Pipeline structure
# ----------------------------------------------------------------------
def test_get_train_transforms_is_replay_compose():
    assert isinstance(get_train_transforms(640), A.ReplayCompose)


def test_train_transform_on_bgr_canvas_with_mask_and_keypoints():
    t = get_train_transforms(SIZE, seed=0)
    img = np.random.default_rng(0).integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    mask = np.ones((SIZE, SIZE), np.uint8)
    kps = np.array([[10.0, 20.0], [100.0, 50.0]])
    out = t(image=img, mask=mask, keypoints=kps)
    assert out["image"].shape == (SIZE, SIZE, 3) and out["image"].dtype == np.uint8
    assert out["mask"].shape == (SIZE, SIZE)
    assert set(np.unique(out["mask"])) <= {0, 1}
    assert len(out["keypoints"]) == 2  # remove_invisible=False keeps order/count


def test_val_transforms_are_noop():
    t = get_val_transforms(SIZE)
    img = np.random.default_rng(0).integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    np.testing.assert_array_equal(t(image=img, keypoints=np.zeros((0, 2)))["image"], img)


def test_seeded_transforms_are_reproducible():
    img = np.random.default_rng(0).integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    a = get_train_transforms(SIZE, seed=5)(image=img, keypoints=np.zeros((0, 2)))["image"]
    b = get_train_transforms(SIZE, seed=5)(image=img, keypoints=np.zeros((0, 2)))["image"]
    c = get_train_transforms(SIZE, seed=6)(image=img, keypoints=np.zeros((0, 2)))["image"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# ----------------------------------------------------------------------
# Geometry: keypoints follow the pixels
# ----------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_keypoints_follow_pixels_under_geometric_aug(tmp_path, seed):
    """Dots drawn at keypoint locations must end up at the augmented
    keypoint locations (checks the pixel-centre <-> index conversion).
    """
    t = get_train_transforms(SIZE, seed=seed, photometric=False)
    ds = CourtDataset(None, None, transform=t, image_size=SIZE, canonicalize_labels=False)
    rng = np.random.default_rng(seed)
    canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    kps = np.full((30, 2), -1.0)
    vis = np.zeros(30, np.float32)
    for k in range(8):
        px, py = rng.integers(16, SIZE - 16, size=2)
        canvas[py, px] = 255
        kps[k] = [(px + 0.5) / SIZE, (py + 0.5) / SIZE]
        vis[k] = 1
    valid = np.ones((SIZE, SIZE), np.uint8)
    out, valid_out, new_kps, new_vis, _ = ds._apply_transform(canvas, valid, kps, vis)
    gray = out[..., 0].astype(float)
    checked = 0
    for k in np.flatnonzero(new_vis):
        x, y = new_kps[k] * SIZE - 0.5  # index coords
        xi, yi = int(round(x)), int(round(y))
        if not (3 <= xi < SIZE - 3 and 3 <= yi < SIZE - 3):
            continue
        win = gray[yi - 3:yi + 4, xi - 3:xi + 4]
        if win.sum() == 0:
            continue  # dot interpolated away under strong downscale
        yy, xx = np.mgrid[yi - 3:yi + 4, xi - 3:xi + 4]
        cx, cy = (win * xx).sum() / win.sum(), (win * yy).sum() / win.sum()
        assert abs(cx - x) < 0.75 and abs(cy - y) < 0.75, (k, cx - x, cy - y)
        checked += 1
    assert checked >= 3


def test_valid_mask_tracks_padding(tmp_path):
    t = _forced(A.Affine(scale=0.6, p=1.0, border_mode=0, fill=0, fill_mask=0))
    ds = CourtDataset(None, None, transform=t, image_size=SIZE)
    canvas = np.full((SIZE, SIZE, 3), 200, np.uint8)
    valid = np.ones((SIZE, SIZE), np.uint8)
    out, valid_out, _, _, _ = ds._apply_transform(canvas, valid, np.full((30, 2), -1.0),
                                                  np.zeros(30, np.float32))
    assert 0.25 < valid_out.mean() < 0.5  # ~0.6^2 of the canvas is content
    # Away from the (bilinear-blended) boundary, padding pixels are black.
    import cv2
    near = cv2.dilate(valid_out.astype(np.uint8), np.ones((3, 3), np.uint8))
    assert out[near == 0].max() == 0


# ----------------------------------------------------------------------
# Labels stay canonical
# ----------------------------------------------------------------------
@pytest.mark.parametrize("view", [END_ON_VIEW, SIDE_VIEW], ids=["end_on", "side"])
def test_augmented_labels_are_canonical(tmp_path, view):
    """After the full train augmentation, any sample whose visible points
    determine the orientation must already be canonical ('identity')."""
    ds = _dataset(tmp_path, [view], get_train_transforms(SIZE, seed=0))
    checked = 0
    for trial in range(40):
        ds.reseed(trial)
        s = ds[0]
        kps, vis = s["keypoints"].numpy(), s["visibility"].numpy()
        if not has_homography_support(vis):
            continue
        _, _, perm = canonicalize(kps, vis, [SIZE, SIZE])
        assert perm == "identity", (trial, perm)
        checked += 1
    assert checked >= 20


@pytest.mark.parametrize("view", [END_ON_VIEW, SIDE_VIEW], ids=["end_on", "side"])
def test_forced_flip_stays_canonical(tmp_path, view):
    ds = _dataset(tmp_path, [view], _forced(A.HorizontalFlip(p=1.0)))
    s = ds[0]
    _, _, perm = canonicalize(s["keypoints"].numpy(), s["visibility"].numpy(), [SIZE, SIZE])
    assert perm == "identity"


def test_flip_pairs_alone_is_wrong_for_side_views(tmp_path):
    """Documents why canonicalize replaced FLIP_PAIRS: mirroring a side view
    and swapping FLIP_PAIRS gives a non-canonical (rot180) labeling, while
    for an end-on view it happens to be right."""
    for view, expected in [(END_ON_VIEW, "identity"), (SIDE_VIEW, "rot180")]:
        _, ann = make_court_annotation(view)
        kps, vis = np.array(ann["keypoints"]), np.array(ann["visibility"], float)
        mirrored = kps.copy()
        mirrored[vis > 0, 0] = 1.0 - mirrored[vis > 0, 0]
        kp2, vis2 = permute_keypoints(mirrored, vis, "col_flip")
        h, w = ann["image_size"]
        assert canonicalize(kp2, vis2, [h, w])[2] == expected


def test_flip_fallback_when_orientation_undeterminable(tmp_path):
    """With too few points for canonicalize, a flip falls back to FLIP_PAIRS."""
    ds = CourtDataset(None, None, transform=_forced(A.HorizontalFlip(p=1.0)), image_size=SIZE)
    kps = np.full((30, 2), -1.0)
    vis = np.zeros(30, np.float32)
    kps[0], vis[0] = [0.25, 0.5], 1  # K0 only
    canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    _, valid, new_kps, new_vis, flipped = ds._apply_transform(
        canvas, np.ones((SIZE, SIZE), np.uint8), kps, vis)
    assert flipped
    new_kps, new_vis = ds._relabel(new_kps, new_vis, flipped)
    partner = dict(FLIP_PAIRS)[0]
    assert new_vis[partner] == 1 and new_vis[0] == 0
    np.testing.assert_allclose(new_kps[partner], [0.75, 0.5], atol=1e-6)


def test_no_flip_no_swap_without_orientation(tmp_path):
    ds = CourtDataset(None, None, transform=_forced(A.HorizontalFlip(p=0.0)), image_size=SIZE)
    kps = np.full((30, 2), -1.0)
    vis = np.zeros(30, np.float32)
    kps[0], vis[0] = [0.25, 0.5], 1
    _, _, new_kps, new_vis, flipped = ds._apply_transform(
        np.zeros((SIZE, SIZE, 3), np.uint8), np.ones((SIZE, SIZE), np.uint8), kps, vis)
    new_kps, new_vis = ds._relabel(new_kps, new_vis, flipped)
    assert not flipped and new_vis[0] == 1
    np.testing.assert_allclose(new_kps[0], [0.25, 0.5], atol=1e-6)


def test_keypoints_leaving_frame_become_invisible(tmp_path):
    t = _forced(A.Affine(scale=2.5, p=1.0, border_mode=0))
    ds = _dataset(tmp_path, [END_ON_VIEW], t)
    s = ds[0]
    vis = s["visibility"].numpy()
    assert 0 < vis.sum() < 30
    assert np.all(s["keypoints"].numpy()[vis == 0] == -1)


class TestReplayHasHorizontalFlipHelper:
    def test_detects_applied_flip(self):
        replay = {"__class_fullname__": "ReplayCompose", "transforms": [
            {"__class_fullname__": "Affine", "applied": True, "transforms": []},
            {"__class_fullname__": "HorizontalFlip", "applied": True, "transforms": []},
        ]}
        assert CourtDataset._replay_has_horizontal_flip(replay) is True

    def test_ignores_unapplied_flip(self):
        replay = {"__class_fullname__": "ReplayCompose", "transforms": [
            {"__class_fullname__": "HorizontalFlip", "applied": False, "transforms": []},
        ]}
        assert CourtDataset._replay_has_horizontal_flip(replay) is False

    def test_handles_none(self):
        assert CourtDataset._replay_has_horizontal_flip(None) is False


# ----------------------------------------------------------------------
# Seeding
# ----------------------------------------------------------------------
def test_reseed_controls_augmentation(tmp_path):
    ds = _dataset(tmp_path, [END_ON_VIEW], get_train_transforms(SIZE, seed=0))
    ds.reseed(1)
    a = ds[0]["image"]
    ds.reseed(1)
    b = ds[0]["image"]
    ds.reseed(2)
    c = ds[0]["image"]
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, c)


def test_seed_worker_reseeds_albumentations(monkeypatch, tmp_path):
    """seed_worker must reseed the dataset's transform from the worker seed."""
    ds = _dataset(tmp_path, [END_ON_VIEW], get_train_transforms(SIZE, seed=0))

    class Info:
        def __init__(self, seed):
            self.seed, self.dataset, self.id = seed, ds, 0

    outs = []
    for seed in (100, 101, 100):
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda s=seed: Info(s))
        seed_worker(0)
        outs.append(ds[0]["image"])
    torch.testing.assert_close(outs[0], outs[2])
    assert not torch.equal(outs[0], outs[1])
    reseed_transform(None, 0)  # no-op, must not raise


def _loader_images(ds, seed):
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                        worker_init_fn=seed_worker, generator=g)
    return [b["image"][0] for b in loader]


@pytest.mark.slow
def test_seed_worker_with_real_workers(tmp_path):
    """Different workers -> different augmentations of identical samples;
    same generator seed -> identical batches across runs."""
    ds = _dataset(tmp_path, [END_ON_VIEW] * 4, get_train_transforms(SIZE, seed=0))
    run1 = _loader_images(ds, seed=7)
    run2 = _loader_images(ds, seed=7)
    for a, b in zip(run1, run2):
        torch.testing.assert_close(a, b)
    # Batches 0 and 1 come from workers 0 and 1 (first item of each).
    assert not torch.equal(run1[0], run1[1])

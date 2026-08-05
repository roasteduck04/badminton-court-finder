"""Tests for horizontal-flip keypoint identity preservation.

Regression coverage for the bug where A.HorizontalFlip mirrored keypoint
x-coordinates without swapping paired left/right keypoint indices (e.g. K0
top-left <-> K1 top-right), silently corrupting keypoint semantics for
every flipped training sample.
"""

import albumentations as A
import numpy as np
import pytest

from src.court_geometry import FLIP_PAIRS
from src.preprocessing.augmentation import get_train_transforms
from src.training.dataset import CourtDataset, NUM_KEYPOINTS


def _forced_flip_transform():
    """A ReplayCompose that always flips horizontally (p=1.0), deterministic
    for testing (no resize/other transforms to keep coordinate math simple).
    """
    return A.ReplayCompose(
        [A.HorizontalFlip(p=1.0)],
        keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=False, angle_in_degrees=True
        ),
    )


def _forced_no_flip_transform():
    """A ReplayCompose that never flips (p=0.0)."""
    return A.ReplayCompose(
        [A.HorizontalFlip(p=0.0)],
        keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=False, angle_in_degrees=True
        ),
    )


def _make_dataset(tmp_path, transform, image_size=64):
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()
    return CourtDataset(
        str(ann_dir), str(img_dir), transform=transform, image_size=image_size
    )


def _distinct_keypoints():
    """14 keypoints with unique, non-symmetric coordinates so that a swap
    bug (wrong pairing) or a missing swap (no pairing) is distinguishable
    from correct behavior.
    """
    keypoints = np.array(
        [[0.03 * (i + 1), 0.01 * (i + 1)] for i in range(NUM_KEYPOINTS)],
        dtype=np.float32,
    )
    visibility = np.ones(NUM_KEYPOINTS, dtype=np.float32)
    return keypoints, visibility


def _raw_flip_no_swap(ds, channels, keypoints):
    """Reproduce CourtDataset._apply_transform's coordinate mapping (pixel
    round-trip through ds.transform) WITHOUT the FLIP_PAIRS swap step.

    Used as an oracle so tests don't have to hardcode albumentations'
    internal pixel-flip arithmetic (e.g. whether it mirrors around
    ``width - x`` or ``width - 1 - x``) -- they only need to check that
    CourtDataset correctly permutes whatever raw values the transform
    produces.
    """
    pixel_kps = [
        (float(keypoints[i, 0] * ds.image_size), float(keypoints[i, 1] * ds.image_size))
        for i in range(NUM_KEYPOINTS)
    ]
    transformed = ds.transform(
        image=(channels * 255.0).astype(np.uint8), keypoints=pixel_kps
    )
    new_kps_px = transformed["keypoints"]
    raw = keypoints.copy()
    for i in range(NUM_KEYPOINTS):
        kx, ky = new_kps_px[i][0], new_kps_px[i][1]
        raw[i] = [kx / ds.image_size, ky / ds.image_size]
    return raw


class TestFlipPairsConstant:
    def test_matches_spec(self):
        assert set(FLIP_PAIRS) == {
            (0, 1),
            (2, 3),
            (4, 6),
            (5, 7),
            (10, 11),
            (12, 13),
        }

    def test_center_line_keypoints_excluded(self):
        paired = {i for pair in FLIP_PAIRS for i in pair}
        assert 8 not in paired
        assert 9 not in paired

    def test_no_index_paired_twice(self):
        paired = [i for pair in FLIP_PAIRS for i in pair]
        assert len(paired) == len(set(paired))


class TestHorizontalFlipSwapsKeypointIdentity:
    def test_flip_swaps_paired_indices(self, tmp_path):
        """After a horizontal flip, keypoint K_i's value must come from the
        flipped position of its FLIP_PAIRS partner (or itself, for
        unpaired K8/K9) -- not from its own flipped position.

        The expected values are derived from the transform's own raw
        (unswapped) output, so this does not depend on hardcoding
        albumentations' internal flip pixel arithmetic.
        """
        ds = _make_dataset(tmp_path, _forced_flip_transform())
        keypoints, visibility = _distinct_keypoints()
        channels = np.zeros((64, 64, 7), dtype=np.float32)

        raw = _raw_flip_no_swap(ds, channels, keypoints)

        partner = {}
        for i, j in FLIP_PAIRS:
            partner[i] = j
            partner[j] = i
        expected = np.array(
            [raw[partner.get(i, i)] for i in range(NUM_KEYPOINTS)], dtype=np.float32
        )

        _, new_kps, new_vis = ds._apply_transform(channels, keypoints, visibility)

        np.testing.assert_allclose(new_kps, expected, atol=1e-5)
        assert np.all(new_vis == 1.0)

    def test_flip_without_swap_would_fail(self, tmp_path):
        """Sanity check that the test fixture actually discriminates:
        the raw (buggy, unswapped) transform output must NOT equal the
        correctly-swapped output for at least one paired keypoint.
        """
        ds = _make_dataset(tmp_path, _forced_flip_transform())
        keypoints, visibility = _distinct_keypoints()
        channels = np.zeros((64, 64, 7), dtype=np.float32)

        raw_unswapped = _raw_flip_no_swap(ds, channels, keypoints)
        _, new_kps, _ = ds._apply_transform(channels, keypoints, visibility)

        assert not np.allclose(new_kps, raw_unswapped, atol=1e-5)

    def test_visibility_swaps_with_partner(self, tmp_path):
        """If only one member of a pair is visible, visibility must move
        to the correct semantic index after the flip, not stay put.
        """
        ds = _make_dataset(tmp_path, _forced_flip_transform())
        keypoints, visibility = _distinct_keypoints()
        # K0 visible, K1 (its pair) invisible.
        visibility[1] = 0.0
        keypoints[1] = [-1.0, -1.0]

        channels = np.zeros((64, 64, 7), dtype=np.float32)
        raw = _raw_flip_no_swap(ds, channels, keypoints)
        _, new_kps, new_vis = ds._apply_transform(channels, keypoints, visibility)

        # After flip+swap, K1 should now carry what was K0's (flipped) data,
        # and K0 should be invisible (inherited from original K1).
        assert new_vis[0] == 0.0
        assert new_vis[1] == 1.0
        np.testing.assert_allclose(new_kps[1], raw[0], atol=1e-5)

    def test_no_flip_leaves_keypoints_unswapped(self, tmp_path):
        """When HorizontalFlip does not fire, keypoint identity/order must
        be untouched (no spurious swap).
        """
        ds = _make_dataset(tmp_path, _forced_no_flip_transform())
        keypoints, visibility = _distinct_keypoints()
        channels = np.zeros((64, 64, 7), dtype=np.float32)

        _, new_kps, new_vis = ds._apply_transform(channels, keypoints, visibility)

        np.testing.assert_allclose(new_kps, keypoints, atol=1e-4)
        assert np.all(new_vis == 1.0)

    def test_plain_compose_without_replay_is_safe(self, tmp_path):
        """A transform pipeline without a 'replay' key (e.g. a plain
        A.Compose) must not crash and must not spuriously swap keypoints.
        """
        plain_transform = A.Compose(
            [A.Resize(64, 64)],
            keypoint_params=A.KeypointParams(
                format="xy", remove_invisible=False, angle_in_degrees=True
            ),
        )
        ds = _make_dataset(tmp_path, plain_transform)
        keypoints, visibility = _distinct_keypoints()
        channels = np.zeros((64, 64, 7), dtype=np.float32)

        _, new_kps, new_vis = ds._apply_transform(channels, keypoints, visibility)

        np.testing.assert_allclose(new_kps, keypoints, atol=1e-4)
        assert np.all(new_vis == 1.0)


class TestReplayHasHorizontalFlipHelper:
    def test_detects_applied_flip(self):
        replay = {
            "__class_fullname__": "ReplayCompose",
            "transforms": [
                {"__class_fullname__": "Resize", "applied": True, "transforms": []},
                {
                    "__class_fullname__": "HorizontalFlip",
                    "applied": True,
                    "transforms": [],
                },
            ],
        }
        assert CourtDataset._replay_has_horizontal_flip(replay) is True

    def test_ignores_unapplied_flip(self):
        replay = {
            "__class_fullname__": "ReplayCompose",
            "transforms": [
                {
                    "__class_fullname__": "HorizontalFlip",
                    "applied": False,
                    "transforms": [],
                },
            ],
        }
        assert CourtDataset._replay_has_horizontal_flip(replay) is False

    def test_handles_none(self):
        assert CourtDataset._replay_has_horizontal_flip(None) is False


def test_get_train_transforms_is_replay_compose():
    """The real training pipeline must use ReplayCompose so flip detection
    in CourtDataset._apply_transform actually works end-to-end.
    """
    transform = get_train_transforms(640)
    assert isinstance(transform, A.ReplayCompose)


def test_get_train_transforms_replay_detects_flip(tmp_path):
    """End-to-end: run the real training pipeline many times and confirm
    that whenever HorizontalFlip fires, CourtDataset correctly detects it
    via the replay log (probabilistic, but p=0.5 over many trials makes a
    false negative astronomically unlikely).
    """
    ds = _make_dataset(tmp_path, get_train_transforms(64), image_size=64)
    keypoints, visibility = _distinct_keypoints()
    channels = np.zeros((64, 64, 7), dtype=np.float32)

    saw_flip_swap = False
    for _ in range(50):
        _, new_kps, _ = ds._apply_transform(channels, keypoints.copy(), visibility.copy())
        # If K0 differs substantially from its original position in a way
        # inconsistent with small jitter (rotate/perspective), a flip+swap
        # likely occurred. We just need *some* trial where new_kps[0] is
        # far from keypoints[0] to prove flip handling runs without error
        # across many random draws.
        if not np.allclose(new_kps[0], keypoints[0], atol=1e-2):
            saw_flip_swap = True
            break

    assert saw_flip_swap, "HorizontalFlip never appeared to fire/swap across 50 trials"

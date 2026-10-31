import numpy as np
import pytest

from src.court_geometry import (
    COURT_KEYPOINTS_TEMPLATE,
    FLIP_PAIRS,
    SYMMETRY_PERMS,
    canonicalize,
    court_orientation,
    has_homography_support,
    permute_keypoints,
    project_points,
)

IMG_H, IMG_W = 720, 1280


def _view(H):
    """Project the template through H into normalized coords of a 1280x720 image."""
    px = project_points(H, COURT_KEYPOINTS_TEMPLATE)
    return px / np.array([IMG_W, IMG_H]), np.ones(30)


# End-on broadcast view: camera behind the K0 baseline, so row 0 is near
# (bottom of the image) and column 0 is on the left. This is canonical.
END_ON = np.array([[0.0, 120.0, 280.0],
                   [-40.0, 0.0, 680.0],
                   [0.04, 0.0, 1.0]])
# Side view: camera beyond the y=6.1 sideline (near sideline at the bottom),
# row 0 on the image left.
SIDE = np.array([[80.0, 0.0, 100.0],
                 [0.0, 50.0, 300.0],
                 [0.0, -0.04, 1.0]])


@pytest.mark.parametrize("H", [END_ON, SIDE])
def test_canonical_view_is_identity(H):
    kps, vis = _view(H)
    out_kps, out_vis, name = canonicalize(kps, vis, [IMG_H, IMG_W])
    assert name == "identity"
    np.testing.assert_allclose(out_kps, kps)


@pytest.mark.parametrize("H", [END_ON, SIDE])
@pytest.mark.parametrize("perm", ["col_flip", "row_flip", "rot180"])
def test_any_symmetric_relabeling_is_undone(H, perm):
    kps, vis = _view(H)
    bad_kps, bad_vis = permute_keypoints(kps, vis, perm)
    out_kps, out_vis, name = canonicalize(bad_kps, bad_vis, [IMG_H, IMG_W])
    # All four permutations are involutions, so the fix is the same symmetry.
    assert name == perm
    np.testing.assert_allclose(out_kps, kps, atol=1e-9)
    np.testing.assert_array_equal(out_vis, vis)


def test_canonical_orientation_is_positive():
    kps, vis = _view(END_ON)
    handedness, row_alignment = court_orientation(kps, vis, [IMG_H, IMG_W])
    assert handedness == 1
    assert row_alignment > 0


def test_canonicalize_is_idempotent_with_partial_visibility():
    kps, vis = _view(SIDE)
    vis = vis.copy()
    vis[[0, 1, 2, 3, 4, 5, 9, 27]] = 0
    kps = kps.copy()
    kps[vis == 0] = -1
    kps, vis = permute_keypoints(kps, vis, "row_flip")
    once, vis1, _ = canonicalize(kps, vis, [IMG_H, IMG_W])
    twice, vis2, name = canonicalize(once, vis1, [IMG_H, IMG_W])
    assert name == "identity"
    np.testing.assert_allclose(twice, once)


def test_collinear_points_are_left_unchanged():
    kps, vis = _view(END_ON)
    vis = np.zeros(30)
    vis[[10, 11, 12, 13, 17]] = 1  # one full row plus a single point
    assert not has_homography_support(vis)
    out_kps, out_vis, name = canonicalize(kps, vis, [IMG_H, IMG_W])
    assert name is None
    np.testing.assert_allclose(out_kps, kps)


def test_homography_support():
    vis = np.zeros(30)
    vis[[0, 4, 25, 29]] = 1
    assert has_homography_support(vis)
    vis = np.zeros(30)
    vis[[0, 1, 2]] = 1
    assert not has_homography_support(vis)


def test_permutations_are_involutions_and_col_flip_matches_flip_pairs():
    for perm in SYMMETRY_PERMS.values():
        assert sorted(perm) == list(range(30))
        assert [perm[perm[k]] for k in range(30)] == list(range(30))
    col = SYMMETRY_PERMS["col_flip"]
    for i, j in FLIP_PAIRS:
        assert col[i] == j and col[j] == i


def test_permutations_preserve_template_geometry():
    """Each symmetry maps the template onto itself (up to a rigid motion)."""
    tpl = COURT_KEYPOINTS_TEMPLATE
    d_ref = np.linalg.norm(tpl[:, None] - tpl[None], axis=-1)
    for perm in SYMMETRY_PERMS.values():
        p = tpl[perm]
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        np.testing.assert_allclose(d, d_ref, atol=1e-9)

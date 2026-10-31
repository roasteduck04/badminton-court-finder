"""Tests for blender_to_cvn converter."""

import json
import math
import os
import shutil
import tempfile

import cv2
import numpy as np
import pytest

from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, SYMMETRY_PERMS, court_orientation
from src.tools.blender_to_cvn import convert_all, convert_metadata, keypoints_in_front

RES = 640


def _canonical_projection():
    """Pixel positions of all 30 keypoints for an end-on view, canonical labels."""
    src = COURT_KEYPOINTS_TEMPLATE[[0, 4, 25, 29]].astype(np.float32)
    dst = np.float32([[100, 600], [540, 600], [220, 150], [420, 150]])
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.perspectiveTransform(COURT_KEYPOINTS_TEMPLATE[None], H)[0]


@pytest.fixture
def tmp_dirs():
    """Create temporary directory structure mimicking data/blender/raw/."""
    base = tempfile.mkdtemp()
    raw_img = os.path.join(base, "raw", "images")
    raw_meta = os.path.join(base, "raw", "metadata")
    out_img = os.path.join(base, "images")
    out_ann = os.path.join(base, "annotations")
    os.makedirs(raw_img)
    os.makedirs(raw_meta)
    yield {
        "base": base,
        "raw_img": raw_img,
        "raw_meta": raw_meta,
        "out_img": out_img,
        "out_ann": out_ann,
    }
    shutil.rmtree(base)


def _make_sample_metadata(visible_count=20, perm="identity", occluded=None, visible=None):
    """Metadata whose keypoints_2d is a real court projection, labeled with ``perm``.

    ``visible`` (list of K indices, in canonical labels) overrides visible_count.
    """
    kp = _canonical_projection()
    vis = np.zeros(30, dtype=int)
    if visible is None:
        vis[:visible_count] = 1
    else:
        vis[visible] = 1
    p = SYMMETRY_PERMS[perm]
    meta = {
        "image_file": "blender_0001.png",
        "resolution": [RES, RES],
        "camera": {
            "strategy": "broadcast",
            "keypoints_2d": kp[p].tolist(),
            "keypoints_3d": [[0, 0, 0]] * 30,
            "visibility": vis[p].tolist(),
        },
    }
    if occluded is not None:
        occ = np.zeros(30, dtype=bool)
        occ[occluded] = True
        meta["camera"]["keypoint_occluded"] = occ[p].tolist()
    return meta


def test_convert_metadata_normalizes_coordinates():
    meta = _make_sample_metadata(visible_count=10)
    result = convert_metadata(meta)
    kp = _canonical_projection()

    for i in range(10):
        assert abs(result["keypoints"][i][0] - kp[i, 0] / RES) < 1e-6
        assert abs(result["keypoints"][i][1] - kp[i, 1] / RES) < 1e-6

    for i in range(10, 30):
        assert result["keypoints"][i] == [-1.0, -1.0]
        assert result["visibility"][i] == 0


def test_convert_metadata_computes_bounding_box():
    meta = _make_sample_metadata(visible_count=10)
    result = convert_metadata(meta)
    bb = result["bounding_box"]
    assert len(bb) == 4
    assert bb[0] >= 0
    assert bb[1] >= 0
    assert bb[2] > 0
    assert bb[3] > 0


def test_convert_metadata_returns_none_below_min_visible():
    meta = _make_sample_metadata(visible_count=2)
    result = convert_metadata(meta, min_visible=4)
    assert result is None


def test_convert_metadata_passes_at_min_visible():
    meta = _make_sample_metadata(visible_count=4)
    result = convert_metadata(meta, min_visible=4)
    assert result is not None
    assert result["image_path"] == "blender_0001.png"


@pytest.mark.parametrize("perm", ["identity", "col_flip", "row_flip", "rot180"])
def test_convert_metadata_canonicalizes_any_labeling(perm):
    meta = _make_sample_metadata(visible_count=30, perm=perm)
    result = convert_metadata(meta)
    assert result["canonical_perm"] == perm  # perms are involutions
    np.testing.assert_allclose(np.array(result["keypoints"]) * RES, _canonical_projection(), atol=1e-6)
    hand, align = court_orientation(result["keypoints"], result["visibility"], result["image_size"])
    assert hand > 0 and align >= 0
    assert result["source"] == "blender"
    assert result["group"] == "blender"


def test_convert_metadata_uses_offframe_points_when_inframe_points_are_collinear():
    # Only the near baseline (K0-K4, all on one line) is in frame: the in-frame
    # points cannot determine a homography, the off-frame projections can.
    meta = _make_sample_metadata(perm="rot180", visible=[0, 1, 2, 3, 4])
    result = convert_metadata(meta, min_visible=4)
    assert result is not None
    assert result["canonical_perm"] == "rot180"
    assert result["visibility"][:5] == [1] * 5
    assert sum(result["visibility"]) == 5


def test_convert_metadata_occlusion_flags_are_permuted_and_kept_visible():
    meta = _make_sample_metadata(visible_count=30, perm="col_flip", occluded=[3, 12])
    result = convert_metadata(meta)
    assert result["occluded"][3] and result["occluded"][12]
    assert sum(result["occluded"]) == 2
    assert result["visibility"][3] == 1 and result["visibility"][12] == 1


def test_convert_metadata_without_occlusion_has_no_field():
    result = convert_metadata(_make_sample_metadata(visible_count=30))
    assert "occluded" not in result


def test_keypoints_in_front():
    cam = {"position": [0.0, 0.0, 10.0], "rotation": [0.0, 0.0, 0.0],
           "keypoints_3d": [[float(i % 5), float(i // 5), 0.0] for i in range(30)]}
    assert keypoints_in_front(cam).all()  # looking straight down -Z at the floor
    cam["rotation"] = [math.pi, 0.0, 0.0]  # looking straight up
    assert not keypoints_in_front(cam).any()
    assert keypoints_in_front({}).all()  # missing pose -> no filtering


def test_convert_all_end_to_end(tmp_dirs):
    # Write a fake image
    img_path = os.path.join(tmp_dirs["raw_img"], "blender_0001.png")
    with open(img_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    # Write metadata
    meta = _make_sample_metadata(visible_count=20)
    meta_path = os.path.join(tmp_dirs["raw_meta"], "blender_0001.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    count = convert_all(
        raw_dir=os.path.join(tmp_dirs["base"], "raw"),
        out_images=tmp_dirs["out_img"],
        out_annotations=tmp_dirs["out_ann"],
        min_visible=4,
    )

    assert count == 1
    assert os.path.isfile(os.path.join(tmp_dirs["out_img"], "blender_0001.png"))

    ann_path = os.path.join(tmp_dirs["out_ann"], "blender_0001.json")
    assert os.path.isfile(ann_path)
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)
    assert ann["image_path"] == "blender_0001.png"
    assert ann["image_size"] == [640, 640]
    assert len(ann["keypoints"]) == 30
    assert len(ann["visibility"]) == 30
    assert len(ann["bounding_box"]) == 4
    assert ann["canonical_perm"] == "identity"


def test_convert_all_skips_low_visibility(tmp_dirs):
    img_path = os.path.join(tmp_dirs["raw_img"], "blender_0002.png")
    with open(img_path, "wb") as f:
        f.write(b"\x89PNG" + b"\x00" * 100)

    meta = _make_sample_metadata(visible_count=2)
    meta["image_file"] = "blender_0002.png"
    meta_path = os.path.join(tmp_dirs["raw_meta"], "blender_0002.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    count = convert_all(
        raw_dir=os.path.join(tmp_dirs["base"], "raw"),
        out_images=tmp_dirs["out_img"],
        out_annotations=tmp_dirs["out_ann"],
        min_visible=4,
    )

    assert count == 0
    assert not os.path.isfile(os.path.join(tmp_dirs["out_ann"], "blender_0002.json"))


def test_convert_metadata_new_fields_group_occlusion_in_frame():
    meta = _make_sample_metadata(visible_count=30, perm="col_flip")
    occ = [False] * 30
    occ[SYMMETRY_PERMS["col_flip"][7]] = True
    meta["keypoint_occluded"] = occ  # top-level copy only
    meta["camera"]["in_frame"] = [1] * 29 + [0]  # takes precedence over visibility
    meta["seed"] = 1234
    result = convert_metadata(meta)
    assert result["group"] == "blender_1234"
    assert result["occluded"][7] and sum(result["occluded"]) == 1
    assert sum(result["visibility"]) == 29
    meta.pop("seed")
    meta["image_index"] = 5
    assert convert_metadata(meta)["group"] == "blender_5"

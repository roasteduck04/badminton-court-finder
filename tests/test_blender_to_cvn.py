"""Tests for blender_to_cvn converter."""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from src.tools.blender_to_cvn import convert_metadata, convert_all


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


def _make_sample_metadata(visible_count=20):
    """Create a sample metadata dict."""
    kp_2d = [[i * 20.0, i * 15.0] for i in range(30)]
    vis = [1] * visible_count + [0] * (30 - visible_count)
    return {
        "image_file": "blender_0001.png",
        "resolution": [640, 640],
        "camera": {
            "strategy": "broadcast",
            "keypoints_2d": kp_2d,
            "keypoints_3d": [[0, 0, 0]] * 30,
            "visibility": vis,
        },
    }


def test_convert_metadata_normalizes_coordinates():
    meta = _make_sample_metadata(visible_count=10)
    result = convert_metadata(meta)

    for i in range(10):
        expected_x = (i * 20.0) / 640.0
        expected_y = (i * 15.0) / 640.0
        assert abs(result["keypoints"][i][0] - expected_x) < 1e-6
        assert abs(result["keypoints"][i][1] - expected_y) < 1e-6

    for i in range(10, 30):
        assert result["keypoints"][i] == [-1.0, -1.0]
        assert result["visibility"][i] == 0


def test_convert_metadata_computes_bounding_box():
    meta = _make_sample_metadata(visible_count=5)
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


def test_convert_all_end_to_end(tmp_dirs):
    # Write a fake image
    img_path = os.path.join(tmp_dirs["raw_img"], "blender_0001.png")
    with open(img_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    # Write metadata
    meta = _make_sample_metadata(visible_count=20)
    meta_path = os.path.join(tmp_dirs["raw_meta"], "blender_0001.json")
    with open(meta_path, "w") as f:
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
    with open(ann_path) as f:
        ann = json.load(f)
    assert ann["image_path"] == "blender_0001.png"
    assert ann["image_size"] == [640, 640]
    assert len(ann["keypoints"]) == 30
    assert len(ann["visibility"]) == 30
    assert len(ann["bounding_box"]) == 4


def test_convert_all_skips_low_visibility(tmp_dirs):
    img_path = os.path.join(tmp_dirs["raw_img"], "blender_0002.png")
    with open(img_path, "wb") as f:
        f.write(b"\x89PNG" + b"\x00" * 100)

    meta = _make_sample_metadata(visible_count=2)
    meta["image_file"] = "blender_0002.png"
    meta_path = os.path.join(tmp_dirs["raw_meta"], "blender_0002.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    count = convert_all(
        raw_dir=os.path.join(tmp_dirs["base"], "raw"),
        out_images=tmp_dirs["out_img"],
        out_annotations=tmp_dirs["out_ann"],
        min_visible=4,
    )

    assert count == 0
    assert not os.path.isfile(os.path.join(tmp_dirs["out_ann"], "blender_0002.json"))

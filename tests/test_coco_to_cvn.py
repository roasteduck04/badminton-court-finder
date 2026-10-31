"""Tests for the Roboflow COCO converters (coco_to_cvn.py, convert_dataset.py)."""

import json

import cv2
import numpy as np

from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, SYMMETRY_PERMS, court_orientation
from src.tools.coco_to_cvn import (
    NP_TO_K,
    coco_image_to_cvn,
    coco_keypoints_to_cvn,
    convert_split,
    pick_court_annotation,
)
from src.tools.convert_dataset import convert_coco

W, H = 640, 480
K_TO_NP = {k: n for n, k in NP_TO_K.items()}


def _projection():
    src = COURT_KEYPOINTS_TEMPLATE[[0, 4, 25, 29]].astype(np.float32)
    dst = np.float32([[100, 450], [540, 450], [220, 100], [420, 100]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    return cv2.perspectiveTransform(COURT_KEYPOINTS_TEMPLATE[None], Hm)[0]


def _coco_flat(perm="identity", v_override=None, visible=range(30)):
    """Flat Roboflow-order keypoints for a court labeled with symmetry ``perm``."""
    kp = _projection()[SYMMETRY_PERMS[perm]]
    flat = [0.0] * 90
    for k in visible:
        n = K_TO_NP[k]
        v = (v_override or {}).get(k, 2)
        flat[n * 3:n * 3 + 3] = [float(kp[k, 0]), float(kp[k, 1]), v]
    return flat


def test_v1_points_are_kept_as_occluded():
    flat = _coco_flat(v_override={7: 1})
    kps, vis, occ = coco_keypoints_to_cvn(flat, W, H)
    assert vis[7] == 1 and occ[7] is True
    assert sum(occ) == 1
    assert sum(vis) == 30


def test_pick_court_annotation_takes_most_keypoints(capsys):
    a = {"keypoints": _coco_flat(visible=range(6))}
    b = {"keypoints": _coco_flat(visible=range(20))}
    junk = {"keypoints": []}
    assert pick_court_annotation([a, junk, b], "x.jpg") is b
    assert "WARNING" in capsys.readouterr().out
    assert pick_court_annotation([junk]) is None


def test_coco_image_to_cvn_canonicalizes_and_groups():
    img = {"file_name": "output_122222_image_0042_png.rf.abc.jpg", "width": W, "height": H}
    ann = coco_image_to_cvn(img, [{"keypoints": _coco_flat(perm="row_flip", v_override={3: 1})}])
    assert ann["canonical_perm"] == "row_flip"
    assert ann["group"] == "output_122222"
    assert ann["source"] == "roboflow"
    hand, align = court_orientation(ann["keypoints"], ann["visibility"], ann["image_size"])
    assert hand > 0 and align >= 0
    np.testing.assert_allclose(np.array(ann["keypoints"]) * [W, H], _projection(), atol=1e-4)
    # the v=1 point moved with its label: K3 under row_flip is K28
    assert ann["occluded"][SYMMETRY_PERMS["row_flip"][3]]


def test_convert_split_end_to_end(tmp_path):
    split = tmp_path / "coco" / "train"
    split.mkdir(parents=True)
    (split / "output5_image_0001_png.rf.x.jpg").write_bytes(b"jpg")
    coco = {
        "images": [{"id": 1, "file_name": "output5_image_0001_png.rf.x.jpg", "width": W, "height": H}],
        "annotations": [{"image_id": 1, "keypoints": _coco_flat(perm="rot180")},
                        {"image_id": 1, "keypoints": _coco_flat(visible=range(5))}],
        "categories": [],
    }
    (split / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    assert convert_split(tmp_path / "coco", tmp_path / "out", "train") == 1
    out = json.loads((tmp_path / "out" / "train" / "annotations" /
                      "output5_image_0001_png.rf.x.json").read_text(encoding="utf-8"))
    assert out["canonical_perm"] == "rot180"
    assert sum(out["visibility"]) == 30
    assert out["group"] == "output5"


def test_convert_dataset_canonicalizes_coco(tmp_path):
    coco = {
        "images": [{"id": 1, "file_name": "output_image_0001_png.rf.y.jpg", "width": W, "height": H}],
        "annotations": [{"image_id": 1, "keypoints": _coco_flat(perm="col_flip")},
                        {"image_id": 1, "keypoints": _coco_flat(visible=range(4))}],
        "categories": [{"id": 1, "name": "court", "keypoints": ["p"] * 30}],
    }
    src = tmp_path / "in.json"
    src.write_text(json.dumps(coco), encoding="utf-8")
    convert_coco(str(src), str(tmp_path / "o" / "out.json"))
    out = json.loads((tmp_path / "o" / "out.json").read_text(encoding="utf-8"))
    assert len(out["annotations"]) == 1
    a = out["annotations"][0]
    assert a["canonical_perm"] == "col_flip"
    assert a["group"] == "output"
    kp = np.array(a["keypoints"]).reshape(30, 3)
    np.testing.assert_allclose(kp[:, :2], _projection(), atol=1e-4)

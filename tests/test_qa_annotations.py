"""Tests for label QA (scripts/qa_annotations.py, src/data_manifest.py) and
scripts/canonicalize_annotations.py."""

import json
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import canonicalize_annotations  # noqa: E402
import qa_annotations  # noqa: E402
from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, SYMMETRY_PERMS  # noqa: E402
from src.data_manifest import (  # noqa: E402
    canonicalize_annotation,
    homography_residual,
    image_shape,
    is_canonical,
    size_status,
)

W, H = 800, 600


def _ann(perm="identity", name="a.jpg", extra=None):
    src = COURT_KEYPOINTS_TEMPLATE[[0, 4, 25, 29]].astype(np.float32)
    dst = np.float32([[100, 580], [700, 580], [300, 120], [500, 120]])
    kp = cv2.perspectiveTransform(COURT_KEYPOINTS_TEMPLATE[None], cv2.getPerspectiveTransform(src, dst))[0]
    kp = (kp / [W, H])[SYMMETRY_PERMS[perm]]
    ann = {"image_path": name, "image_size": [H, W], "keypoints": kp.tolist(),
           "visibility": [1] * 30, "bounding_box": [0, 0, 1, 1]}
    ann.update(extra or {})
    return ann


def test_homography_residual_detects_bad_point():
    ann = _ann()
    r = homography_residual(ann["keypoints"], ann["visibility"], ann["image_size"])
    assert r["max"] < 1e-3 and r["n"] == 30 and r["dof"] == 52
    ann["keypoints"][17][0] += 40 / W
    r = homography_residual(ann["keypoints"], ann["visibility"], ann["image_size"])
    assert r["max"] > 15 and r["worst_kp"] == 17
    # collinear-only labels cannot be checked
    vis = [1 if i < 5 else 0 for i in range(30)]
    assert homography_residual(ann["keypoints"], vis, ann["image_size"]) is None


def test_canonicalize_annotation_permutes_per_point_fields():
    occ = [False] * 30
    occ[2] = True
    origin = ["clicked"] * 30
    origin[5] = "suggested"
    ann = _ann("rot180", extra={"occluded": [occ[i] for i in SYMMETRY_PERMS["rot180"]],
                                "keypoint_origin": [origin[i] for i in SYMMETRY_PERMS["rot180"]]})
    assert is_canonical(ann) is False
    new, applied = canonicalize_annotation(ann)
    assert applied == "rot180" and new["canonical_perm"] == "rot180"
    assert new["occluded"] == occ and new["keypoint_origin"] == origin
    assert is_canonical(new) is True
    again, applied2 = canonicalize_annotation(new)
    assert applied2 == "identity" and again["canonical_perm"] == "rot180"  # cumulative


def test_size_status_and_image_shape(tmp_path):
    assert size_status([600, 800], (600, 800)) == "ok"
    assert size_status([800, 600], (600, 800)) == "transposed"
    assert size_status([300, 400], (600, 800)) == "rescaled"
    assert size_status([600, 600], (600, 800)) == "mismatch"
    assert size_status([600, 800], None) == "unreadable"
    p = str(tmp_path / "x.png")
    cv2.imwrite(p, np.zeros((30, 50, 3), np.uint8))
    assert image_shape(p) == (30, 50)


def test_qa_directory_flags(tmp_path):
    ann_dir, img_dir = tmp_path / "set" / "annotations", tmp_path / "set" / "images"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    for n in ("good", "rot", "bad", "size"):
        cv2.imwrite(str(img_dir / f"{n}.jpg"), np.zeros((H, W, 3), np.uint8))
    cv2.imwrite(str(img_dir / "size.jpg"), np.zeros((W, H, 3), np.uint8))  # transposed
    anns = {"good": _ann(name="good.jpg"), "rot": _ann("rot180", "rot.jpg"),
            "bad": _ann(name="bad.jpg"), "size": _ann(name="size.jpg"), "missing": _ann(name="nope.jpg")}
    anns["bad"]["keypoints"][3][1] += 0.1
    for n, a in anns.items():
        (ann_dir / f"{n}.json").write_text(json.dumps(a), encoding="utf-8")
    rows = {os.path.basename(r["file"])[:-5]: r for r in qa_annotations.qa_directory(str(ann_dir))}
    assert rows["good"]["flags"] == []
    assert rows["rot"]["flags"] == ["not_canonical(rot180)"]
    assert any(f.startswith("residual>") for f in rows["bad"]["flags"])
    assert "image_transposed" in rows["size"]["flags"]
    assert "image_missing_image" in rows["missing"]["flags"]
    assert rows["good"]["source"] == "manual"
    text = qa_annotations.summarize(list(rows.values()), 15.0)
    assert "manual" in text


def test_qa_cli_writes_csv(tmp_path, capsys):
    d = tmp_path / "annotations"
    d.mkdir()
    (d / "a.json").write_text(json.dumps(_ann("col_flip")), encoding="utf-8")
    out = tmp_path / "qa.csv"
    qa_annotations.main([str(d), "--no-image-check", "--csv", str(out)])
    assert "not_canonical(col_flip)" in capsys.readouterr().out
    assert "col_flip" in out.read_text(encoding="utf-8")


@pytest.fixture
def ann_dir(tmp_path):
    d = tmp_path / "anns"
    d.mkdir()
    (d / "canon.json").write_text(json.dumps(_ann()), encoding="utf-8")
    (d / "rot.json").write_text(json.dumps(_ann("rot180")), encoding="utf-8")
    sparse = _ann()
    sparse["visibility"] = [1] * 5 + [0] * 25
    (d / "undet.json").write_text(json.dumps(sparse), encoding="utf-8")
    return d


def _snapshot(d):
    return {f: (d / f).read_text(encoding="utf-8") for f in sorted(os.listdir(d))}


def test_canonicalize_dry_run_writes_nothing(ann_dir):
    before = _snapshot(ann_dir)
    rep = canonicalize_annotations.main([str(ann_dir), "--dry-run"])
    assert rep["counts"] == {"identity": 1, "rot180": 1, None: 1}
    assert _snapshot(ann_dir) == before


def test_canonicalize_to_out_dir(ann_dir, tmp_path):
    before = _snapshot(ann_dir)
    out = tmp_path / "out"
    canonicalize_annotations.main([str(ann_dir), "--out", str(out)])
    assert _snapshot(ann_dir) == before
    assert sorted(os.listdir(out)) == ["canon.json", "rot.json", "undet.json"]
    rot = json.loads((out / "rot.json").read_text(encoding="utf-8"))
    assert rot["canonical_perm"] == "rot180" and is_canonical(rot)
    assert json.loads((out / "undet.json").read_text(encoding="utf-8"))["canonical_perm"] is None
    with pytest.raises(ValueError):
        canonicalize_annotations.canonicalize_directory(str(ann_dir), str(ann_dir))


def test_canonicalize_in_place_keeps_backup(ann_dir):
    orig_rot = (ann_dir / "rot.json").read_text(encoding="utf-8")
    canonicalize_annotations.main([str(ann_dir), "--in-place"])
    assert sorted(os.listdir(ann_dir)) == ["canon.json", "rot.json", "rot.json.bak", "undet.json"]
    assert (ann_dir / "rot.json.bak").read_text(encoding="utf-8") == orig_rot
    assert is_canonical(json.loads((ann_dir / "rot.json").read_text(encoding="utf-8")))

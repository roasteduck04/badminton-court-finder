"""Tests for the leak-free dataset builder (scripts/prepare_colab.py) and split helpers."""

import json
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import prepare_colab  # noqa: E402
from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, SYMMETRY_PERMS, court_orientation  # noqa: E402
from src.data_manifest import (  # noqa: E402
    assign_groups_to_splits,
    compose_perm_names,
    cross_split_near_duplicates,
    hand_label_groups,
    roboflow_group,
    thumb_vector,
)

IMG = 96


def _projection(w=IMG, h=IMG):
    src = COURT_KEYPOINTS_TEMPLATE[[0, 4, 25, 29]].astype(np.float32)
    dst = np.float32([[0.15 * w, 0.95 * h], [0.85 * w, 0.95 * h], [0.35 * w, 0.2 * h], [0.65 * w, 0.2 * h]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    return cv2.perspectiveTransform(COURT_KEYPOINTS_TEMPLATE[None], Hm)[0] / [w, h]


def _ann(img_name, perm="identity", n_vis=30, extra=None):
    kp = _projection()[SYMMETRY_PERMS[perm]]
    vis = [1] * n_vis + [0] * (30 - n_vis)
    vis = [vis[i] for i in SYMMETRY_PERMS[perm]]
    kps = [list(map(float, p)) if v else [-1.0, -1.0] for p, v in zip(kp, vis)]
    ann = {"image_path": img_name, "image_size": [IMG, IMG], "keypoints": kps,
           "visibility": vis, "bounding_box": [0, 0, 1, 1]}
    ann.update(extra or {})
    return ann


_rng = np.random.default_rng(0)


def _write_image(path, img=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img is None:
        img = _rng.integers(0, 255, (IMG, IMG, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return img


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


@pytest.fixture
def fake_data(tmp_path):
    d = tmp_path / "data"
    # Hand labels: 4 sessions by timestamp + 2 web singles
    hand = []
    for t in ["120000", "120005", "120010"]:
        hand.append(f"IMG_20260101_{t}.jpg")
    for t in ["150000", "150003"]:
        hand.append(f"IMG_20260101_{t}.jpg")
    hand += ["IMG_20260202_090000.jpg", "IMG_20260303_090000.jpg", "images (1).jpeg", "images (2).jpeg"]
    for i, name in enumerate(hand):
        _write_image(str(d / "images" / name))
        perm = ["identity", "rot180", "col_flip", "row_flip"][i % 4]
        _write_json(str(d / "annotations" / (os.path.splitext(name)[0] + ".json")), _ann(name, perm))
    # Roboflow: 5 videos; vidE shares an identical frame with vidD
    shared = None
    for v, n in [("vidA", 6), ("vidB", 4), ("vidC", 3), ("vidD", 2), ("vidE", 2)]:
        for k in range(n):
            name = f"{v}_image_{k:04d}_png.rf.h{k}.jpg"
            split = ["train", "valid", "test"][k % 3]
            img = shared if (v == "vidE" and k == 0) else None
            out = _write_image(str(d / "cvn_dataset" / split / "images" / name), img)
            if v == "vidD" and k == 0:
                shared = out
            _write_json(str(d / "cvn_dataset" / split / "annotations" / (name[:-4] + ".json")), _ann(name))
    # One Roboflow label with a gross error (one point 30% of the image off)
    bad = _ann("vidA_image_0099_png.rf.z.jpg")
    bad["keypoints"][12] = [bad["keypoints"][12][0] + 0.3, bad["keypoints"][12][1]]
    _write_image(str(d / "cvn_dataset" / "train" / "images" / "vidA_image_0099_png.rf.z.jpg"))
    _write_json(str(d / "cvn_dataset" / "train" / "annotations" / "vidA_image_0099_png.rf.z.json"), bad)
    # Blender raw: 3 renders, one with too few in-frame points
    for i, nvis in enumerate([30, 12, 2]):
        name = f"blender_{i:04d}.png"
        _write_image(str(d / "blender" / "raw" / "images" / name))
        kp = _projection() * IMG
        vis = [1] * nvis + [0] * (30 - nvis)
        p = SYMMETRY_PERMS["col_flip"]
        _write_json(str(d / "blender" / "raw" / "metadata" / f"blender_{i:04d}.json"), {
            "image_file": name, "resolution": [IMG, IMG],
            "camera": {"keypoints_2d": kp[p].tolist(), "visibility": [vis[j] for j in p],
                       "keypoints_3d": [[0, 0, 0]] * 30}})
    return d


def _args(d, out, *extra):
    return ["--annotations", str(d / "annotations"), "--images", str(d / "images"),
            "--dataset", str(d / "cvn_dataset"), "--blender", str(d / "blender"),
            "--out", str(out), "--max-residual", "3", *extra]


def _load_split_files(out):
    res = {}
    for sp in prepare_colab.SPLITS:
        ad = os.path.join(out, sp, "annotations")
        res[sp] = sorted(os.listdir(ad)) if os.path.isdir(ad) else []
    return res


def test_build_is_group_disjoint_canonical_and_reproducible(fake_data, tmp_path):
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    m1 = prepare_colab.main(_args(fake_data, out1))
    m2 = prepare_colab.main(_args(fake_data, out2))
    assert m1["splits"] == m2["splits"]  # deterministic

    split_of_group = {}
    for sp, entries in m1["splits"].items():
        for e in entries:
            key = (e["source"], e["group"])
            assert split_of_group.setdefault(key, sp) == sp, f"group {key} spans splits"
            if e["source"] == "blender":
                assert sp == "train"
            if e["source"] == "manual":
                assert sp in ("train", "val", "test_real")
            if e["source"] == "roboflow":
                assert sp in ("train", "val", "test_roboflow")
            ann = json.load(open(os.path.join(out1, sp, "annotations", e["file"]), encoding="utf-8"))
            assert os.path.isfile(os.path.join(out1, sp, "images", ann["image_path"]))
            hand, align = court_orientation(ann["keypoints"], ann["visibility"], ann["image_size"])
            assert hand > 0 and align >= 0
            assert ann["source"] == e["source"] and ann["group"] == e["group"]

    # vidD / vidE share a frame -> merged into one group
    groups = {e["group"] for es in m1["splits"].values() for e in es if e["source"] == "roboflow"}
    assert "vidD+vidE" in groups
    assert m1["near_duplicates"] == []
    # residual outlier and low-visibility Blender render are dropped and reported
    reasons = {(d["source"], d["reason"]) for d in m1["dropped"]}
    assert ("roboflow", "residual_outlier") in reasons
    assert ("blender", "below_min_visible_train") in reasons
    assert len(m1["splits"]["test_real"]) > 0 and len(m1["splits"]["test_roboflow"]) > 0
    assert os.path.isfile(os.path.join(out1, "summary.txt"))


def test_rebuild_from_manifest_is_identical(fake_data, tmp_path):
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    prepare_colab.main(_args(fake_data, out1, "--seed", "7"))
    m2 = prepare_colab.main(_args(fake_data, out2, "--manifest", str(out1 / "manifest.json")))
    m1 = json.load(open(out1 / "manifest.json", encoding="utf-8"))
    assert m1["splits"] == m2["splits"]
    assert _load_split_files(out1) == _load_split_files(out2)


def test_rebuild_detects_changed_source(fake_data, tmp_path):
    out1 = tmp_path / "o1"
    m = prepare_colab.main(_args(fake_data, out1))
    entry = m["splits"]["train"][0]
    with open(os.path.join(prepare_colab._abs(entry["src_annotation"])), "a", encoding="utf-8") as f:
        f.write(" ")
    with pytest.raises(SystemExit):
        prepare_colab.main(_args(fake_data, tmp_path / "o2", "--manifest", str(out1 / "manifest.json")))


def test_refuses_existing_or_legacy_output(fake_data, tmp_path):
    out = tmp_path / "o"
    prepare_colab.main(_args(fake_data, out))
    with pytest.raises(SystemExit):
        prepare_colab.main(_args(fake_data, out))  # exists, no --overwrite
    prepare_colab.main(_args(fake_data, out, "--overwrite"))
    with pytest.raises(SystemExit):
        prepare_colab.prepare_out_dir(prepare_colab.LEGACY_OUTPUT_DIR, overwrite=True)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        prepare_colab.prepare_out_dir(str(foreign), overwrite=True)
    assert (foreign / "keep.txt").exists()


def test_min_visible_applies_to_train_only(fake_data, tmp_path):
    # Make every hand label sparse (5 points incl. a non-collinear set)
    for f in os.listdir(fake_data / "annotations"):
        p = fake_data / "annotations" / f
        ann = json.load(open(p, encoding="utf-8"))
        keep = {0, 4, 25, 29, 12}
        ann["visibility"] = [1 if i in keep else 0 for i in range(30)]
        ann["keypoints"] = [k if i in keep else [-1.0, -1.0] for i, k in enumerate(ann["keypoints"])]
        _write_json(str(p), ann)
    m = prepare_colab.main(_args(fake_data, tmp_path / "o", "--min-visible", "6"))
    assert not any(e["source"] == "manual" for e in m["splits"]["train"])
    assert any(e["source"] == "manual" for e in m["splits"]["test_real"] + m["splits"]["val"])


# --- helpers ---------------------------------------------------------------

def test_roboflow_group():
    assert roboflow_group("output_122222_image_0171_png.rf.26b.json") == "output_122222"
    assert roboflow_group("output6_image_0708_png.rf.x.jpg") == "output6"
    assert roboflow_group("weird.jpg") == "weird"


def test_hand_label_groups(tmp_path):
    names = ["IMG_20260815_201556.jpg", "IMG_20260815_201650.jpg", "IMG_20260815_203500.jpg",
             "Screenshot 2026-08-13 130730.png", "Screenshot 2026-08-13 131001.png",
             "IMG_3850_01.jpg", "IMG_3855_02.jpg", "IMG_3736_frame_00000.jpg", "IMG_3776_frame_00013.jpg",
             "images (4).jpeg", "photo_2026-09-10_17-37-33.jpg"]
    g = hand_label_groups([(n, str(tmp_path / n)) for n in names])
    assert g[names[0]] == g[names[1]] != g[names[2]]  # 20 min gap splits the session
    assert g[names[3]] == g[names[4]]
    assert g[names[5]] == g[names[6]]  # camera counters 3850..3855
    assert g[names[7]] != g[names[8]]  # 3736 vs 3776: counter gap > 20
    assert g[names[9]].startswith("single_")
    assert g[names[10]].startswith("session_20260910")


def test_assign_groups_to_splits_is_deterministic_and_complete():
    sizes = {f"g{i}": (i % 7) + 1 for i in range(30)}
    ratios = {"train": 0.7, "val": 0.1, "test": 0.2}
    a = assign_groups_to_splits(sizes, ratios, seed=3)
    assert a == assign_groups_to_splits(dict(reversed(list(sizes.items()))), ratios, seed=3)
    assert set(a) == set(sizes)
    tot = {s: sum(sizes[g] for g in a if a[g] == s) for s in ratios}
    n = sum(sizes.values())
    for s, r in ratios.items():
        assert abs(tot[s] - r * n) <= max(sizes.values())
    assert set(assign_groups_to_splits(sizes, {"train": 1, "val": 0}, 0).values()) == {"train"}


def test_cross_split_near_duplicates():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    b = rng.integers(0, 255, (64, 64), dtype=np.uint8)
    V = np.array([thumb_vector(a), thumb_vector(a.copy()), thumb_vector(b)])
    assert cross_split_near_duplicates(V, ["train", "test", "test"]) == [(0, 1, pytest.approx(1.0))]
    assert cross_split_near_duplicates(V, ["train", "train", "test"]) == []


def test_compose_perm_names():
    for a in SYMMETRY_PERMS:
        assert compose_perm_names(a, a) == "identity"
        assert compose_perm_names("identity", a) == a
        # composition matches applying the permutations
        for b in SYMMETRY_PERMS:
            pa, pb = SYMMETRY_PERMS[a], SYMMETRY_PERMS[b]
            composed = [pa[pb[k]] for k in range(30)]
            assert composed == SYMMETRY_PERMS[compose_perm_names(a, b)]
    assert compose_perm_names(None, "rot180") == "rot180"


def test_extra_synthetic_is_train_only(fake_data, tmp_path):
    syn = tmp_path / "aug"
    for i in range(3):
        name = f"blender_{i:04d}.png"
        _write_image(str(syn / "images" / name))
        _write_json(str(syn / "annotations" / f"blender_{i:04d}.json"),
                    _ann(name, "col_flip", extra={"augment": {"k1": 0.0}, "group": f"blender_{i}"}))
    m = prepare_colab.main(_args(fake_data, tmp_path / "o", "--extra-synthetic", str(syn)))
    extra = [e for sp, es in m["splits"].items() for e in es if e["file"].startswith("blx0_")]
    assert len(extra) == 3
    assert all(e["source"] == "blender" for e in extra)
    assert all(e in m["splits"]["train"] for e in extra)
    # rebuild from the manifest reproduces the prefixed files
    m2 = prepare_colab.main(_args(fake_data, tmp_path / "o2", "--manifest", str(tmp_path / "o" / "manifest.json")))
    assert m2["splits"] == m["splits"]

"""Tests for src.evaluation.evaluate and scripts/nn_baseline.py.

The model / CourtPredictor are replaced by fakes following the
CourtDetection contract, so these tests do not depend on predict.py.
"""

import hashlib
import importlib.util
import json
import os
import sys
import types
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, NUM_KEYPOINTS, SYMMETRY_PERMS, project_points
from src.evaluation.evaluate import (
    DEFAULT_SEG_THICKNESS,
    evaluate,
    infer_group,
    infer_source,
    letterbox_info,
    load_items,
    main,
    markdown_report,
    print_summary,
    seg_iou_letterbox,
    write_outputs,
)
from src.evaluation.metrics import render_line_mask
from src.preprocessing.letterbox import keypoints_to_letterbox, letterbox

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _court_kps(h, w, rng):
    """Random plausible court labels for an (h, w) image."""
    s = rng.uniform(0.5, 0.8) * min(w / 13.4, h / 6.1)
    H = np.array([[s, 0, rng.uniform(0, w - 13.4 * s)], [0, s, rng.uniform(0, h - 6.1 * s)],
                  [rng.uniform(-1e-3, 1e-3), 0, 1.0]])
    return project_points(H, COURT_KEYPOINTS_TEMPLATE) / np.array([w, h])


def _key(img):
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def _make_dataset(root, names, sizes, seed=0, vis_fn=None):
    """Write PNG images + CVN annotations; returns {image md5: (kps, vis)}."""
    rng = np.random.default_rng(seed)
    img_dir, ann_dir = root / "images", root / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    truth = {}
    for i, (name, (h, w)) in enumerate(zip(names, sizes)):
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{name}.png"), img)
        kps = _court_kps(h, w, rng)
        vis = np.ones(NUM_KEYPOINTS) if vis_fn is None else vis_fn(i)
        ann = {"image_path": f"{name}.png", "image_size": [h, w],
               "keypoints": kps.tolist(), "visibility": vis.astype(int).tolist(),
               "bounding_box": [0, 0, 1, 1]}
        with open(ann_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(ann, f)
        truth[_key(cv2.imread(str(img_dir / f"{name}.png")))] = (kps, vis)
    return str(ann_dir), str(img_dir), truth


class FakePredictor:
    """CourtDetection-contract fake: refined keypoints = GT (optionally
    transformed), raw keypoints = GT + offset."""

    def __init__(self, truth, raw_offset_px=40.0, perm=None, seg=True):
        self.truth = truth
        self.raw_offset_px = raw_offset_px
        self.perm = perm
        self.seg = seg

    def predict(self, img):
        kps, vis = self.truth[_key(img)]
        h, w = img.shape[:2]
        refined = kps[self.perm] if self.perm is not None else kps.copy()
        seg_mask = None
        if self.seg:
            info = letterbox_info(h, w, 640)
            seg_mask = render_line_mask(keypoints_to_letterbox(kps, info), vis, 640, DEFAULT_SEG_THICKNESS).astype(np.float32)
        return SimpleNamespace(
            keypoints=refined,
            raw_keypoints=kps + np.array([self.raw_offset_px / w, 0.0]),
            visibility=np.where(vis > 0, 0.9, 0.1),
            homography=np.eye(3),
            inlier_mask=vis > 0,
            image_size=[h, w],
            seg_mask=seg_mask,
        )

    def predict_tta(self, img):
        return self.predict(img)


@pytest.fixture
def dataset(tmp_path):
    names = ["mine_a", "ds_output2_image_0001_png.rf.x", "ds_output2_image_0002_png.rf.y", "bl_00001"]
    sizes = [(120, 160), (96, 96), (96, 96), (90, 160)]
    return _make_dataset(tmp_path / "ds", names, sizes)


def test_evaluate_perfect_fake_predictor(dataset):
    ann_dir, img_dir, truth = dataset
    summary = evaluate(None, ann_dir, img_dir, predictor=FakePredictor(truth), n_boot=20)
    assert summary["n_images"] == 4
    st = summary["strict"]
    assert st["pck@5px"] == 1.0 and st["pck_norm@0.01"] == 1.0
    assert summary["homography"]["success_rate"] == 1.0
    assert summary["court_iou"]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert summary["visibility"]["f1"] == 1.0
    assert summary["segmentation"]["seg_iou_mean"] == pytest.approx(1.0)
    assert summary["segmentation"]["n"] == 4
    assert set(summary["by_source"]) == {"manual", "roboflow", "blender"}
    assert summary["by_group"]["output2"]["n_images"] == 2
    assert summary["ci"]["pck@10px"] == [1.0, 1.0]


def test_no_refine_scores_raw_keypoints(dataset):
    ann_dir, img_dir, truth = dataset
    s = evaluate(None, ann_dir, img_dir, predictor=FakePredictor(truth, raw_offset_px=40), refine=False,
                 n_boot=0)
    # 40 original px on <=160 px images is >= 160 reference px
    assert s["strict"]["pck@20px"] == 0.0
    assert s["meta"]["refine"] is False


def test_tta_uses_predict_tta(dataset):
    ann_dir, img_dir, truth = dataset
    s = evaluate(None, ann_dir, img_dir, predictor=FakePredictor(truth), use_tta=True, n_boot=0)
    assert s["strict"]["pck@5px"] == 1.0
    with pytest.raises(RuntimeError):
        evaluate(None, ann_dir, img_dir, predictor=SimpleNamespace(predict=None), use_tta=True, n_boot=0)


def test_symmetric_prediction_reported_separately(dataset):
    ann_dir, img_dir, truth = dataset
    s = evaluate(None, ann_dir, img_dir, predictor=FakePredictor(truth, perm=SYMMETRY_PERMS["rot180"]),
                 n_boot=0)
    assert s["strict"]["pck@10px"] < 0.2
    assert s["sym"]["pck@10px"] == 1.0


def test_zero_visible_image_excluded(tmp_path):
    vis_fn = lambda i: np.zeros(NUM_KEYPOINTS) if i == 0 else np.ones(NUM_KEYPOINTS)  # noqa: E731
    ann_dir, img_dir, truth = _make_dataset(tmp_path, ["a", "b"], [(80, 80), (80, 80)], vis_fn=vis_fn)
    s = evaluate(None, ann_dir, img_dir, predictor=FakePredictor(truth), n_boot=0)
    assert s["strict"]["n_images_no_visible"] == 1
    assert s["strict"]["img_pck@10px"] == 1.0


def test_write_outputs_strict_json(dataset, tmp_path):
    ann_dir, img_dir, truth = dataset
    out = tmp_path / "res" / "run.json"
    s = evaluate(None, ann_dir, img_dir, predictor=FakePredictor(truth), n_boot=5, out=str(out))

    def no_nan(x):
        raise ValueError(f"non-standard JSON constant {x}")

    with open(out, encoding="utf-8") as f:
        loaded = json.load(f, parse_constant=no_nan)
    assert loaded["strict"]["pck@10px"] == 1.0
    with open(tmp_path / "res" / "run_rows.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line, parse_constant=no_nan) for line in f]
    assert len(rows) == 4
    for key in ("name", "err_px", "sym_perm", "h_inliers", "gt_handedness", "pred_handedness",
                "source", "group", "model_inliers"):
        assert key in rows[0]
    md = (tmp_path / "res" / "run.md").read_text(encoding="utf-8")
    assert "pck_norm@0.02" in md and "| Source |" in md
    assert s["meta"]["seg_thickness"] == DEFAULT_SEG_THICKNESS


def test_rows_roundtrip_through_json(dataset, tmp_path):
    """Aggregating rows read back from JSONL gives the same summary."""
    from src.evaluation.metrics import aggregate, evaluate_image

    kps = _court_kps(100, 100, np.random.default_rng(1))
    vis = np.ones(NUM_KEYPOINTS)
    pred = kps.copy()
    pred[3] = np.nan  # non-finite prediction: must stay a miss after JSON
    row = evaluate_image(pred, vis, kps, vis, (100, 100))
    stem = str(tmp_path / "r.json")
    write_outputs(stem, [row], aggregate([row], n_boot=0))
    with open(tmp_path / "r_rows.jsonl", encoding="utf-8") as f:
        back = [json.loads(line) for line in f]
    a = aggregate([row], n_boot=0)["strict"]
    b = aggregate(back, n_boot=0)["strict"]
    assert a["pck@5px"] == b["pck@5px"] == pytest.approx(29 / 30)
    assert b["n_points"] == 30


def test_print_summary_smoke(dataset, capsys):
    ann_dir, img_dir, truth = dataset
    s = evaluate(None, ann_dir, img_dir, predictor=FakePredictor(truth), n_boot=0)
    print_summary(s)
    out = capsys.readouterr().out
    assert "CourtVisionNet evaluation" in out and "pck@10px" in out


def test_markdown_handles_empty_summary():
    assert "Metric" in markdown_report({})


def test_manifest_layouts(dataset, tmp_path):
    ann_dir, img_dir, _ = dataset
    anns = sorted(os.listdir(ann_dir))
    m1 = tmp_path / "m1.json"
    m1.write_text(json.dumps({"splits": {"test_real": [os.path.join(ann_dir, anns[0])],
                                         "train": [os.path.join(ann_dir, a) for a in anns[1:]]}}),
                  encoding="utf-8")
    items = load_items(manifest=str(m1), split="test_real", images_dir=img_dir)
    assert len(items) == 1 and os.path.exists(items[0]["image_path"])

    m2 = tmp_path / "m2.jsonl"
    m2.write_text("\n".join(json.dumps({"annotation": os.path.join(ann_dir, a),
                                        "image": os.path.join(img_dir, a.replace(".json", ".png")),
                                        "split": "test_real" if i < 2 else "train",
                                        "group": "sess1"})
                            for i, a in enumerate(anns)), encoding="utf-8")
    items = load_items(manifest=str(m2), split="test_real")
    assert len(items) == 2 and all(it["group"] == "sess1" for it in items)
    with pytest.raises(KeyError):
        load_items(manifest=str(m1), split="nope")


def test_prepare_colab_manifest_layout(tmp_path):
    """Entries {"file", "image", ...} resolve to <manifest dir>/<split>/..."""
    root = tmp_path / "colab_v2"
    ann_dir, img_dir, truth = _make_dataset(root / "test_real", ["mine_a", "mine_b"], [(80, 100)] * 2)
    entries = [{"file": f"mine_{c}.json", "image": f"mine_{c}.png", "source": "manual",
                "group": f"session{c}", "src_annotation": "does/not/exist.json"} for c in "ab"]
    (root / "manifest.json").write_text(json.dumps({"splits": {"test_real": entries, "train": []}}),
                                        encoding="utf-8")
    items = load_items(manifest=str(root / "manifest.json"), split="test_real")
    assert [it["group"] for it in items] == ["sessiona", "sessionb"]
    assert all(os.path.exists(it["image_path"]) for it in items)
    s = evaluate(None, manifest=str(root / "manifest.json"), split="test_real",
                 predictor=FakePredictor(truth), n_boot=0)
    assert s["n_images"] == 2 and s["strict"]["pck@5px"] == 1.0


def test_source_and_group_inference():
    assert infer_source({}, "mine_IMG_1.jpg") == "manual"
    assert infer_source({}, "ds_output2_image_0010_png.rf.abc.jpg") == "roboflow"
    assert infer_source({}, "bl_00012.png") == "blender"
    assert infer_source({"source": "blender"}, "x.jpg") == "blender"
    assert infer_source({}, "x.jpg") == "unknown"
    assert infer_group({}, "ds_output2_image_0010_png.rf.abc.jpg", "roboflow") == "output2"
    assert infer_group({"group": "vid7"}, "ds_output2_image_0010.jpg", "roboflow") == "vid7"
    assert infer_group({}, "mine_IMG_1.jpg", "manual") == "manual"


def test_seg_iou_letterbox_original_shaped_mask():
    h, w = 90, 160
    kps = _court_kps(h, w, np.random.default_rng(2))
    vis = np.ones(NUM_KEYPOINTS)
    info = letterbox_info(h, w, 640)
    canvas_mask = render_line_mask(
        keypoints_to_letterbox(kps, info), vis, 640, DEFAULT_SEG_THICKNESS
    )
    assert seg_iou_letterbox(canvas_mask, kps, vis, (h, w)) == pytest.approx(1.0)
    assert seg_iou_letterbox(canvas_mask * 255, kps, vis, (h, w)) == pytest.approx(1.0)
    assert seg_iou_letterbox(None, kps, vis, (h, w)) is None
    # letterbox_info must agree with the real letterbox()
    _, real = letterbox(np.zeros((h, w), np.uint8), 640)
    assert real == info


def test_cli_with_fake_courtpredictor(dataset, tmp_path, monkeypatch, capsys):
    ann_dir, img_dir, truth = dataset
    fake_mod = types.ModuleType("src.inference.predict")

    class CourtPredictor(FakePredictor):
        def __init__(self, checkpoint_path, device=None, **kw):
            assert checkpoint_path == "ckpt.pt"
            super().__init__(truth)

    fake_mod.CourtPredictor = CourtPredictor
    monkeypatch.setitem(sys.modules, "src.inference.predict", fake_mod)
    out = tmp_path / "cli.json"
    main(["--checkpoint", "ckpt.pt", "--annotations", ann_dir, "--images", img_dir,
          "--out", str(out), "--n-boot", "10"])
    assert out.exists() and (tmp_path / "cli_rows.jsonl").exists() and (tmp_path / "cli.md").exists()
    assert "pck@10px" in capsys.readouterr().out


def _load_nn_baseline():
    spec = importlib.util.spec_from_file_location("nn_baseline", os.path.join(REPO, "scripts", "nn_baseline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_nn_baseline_self_match_is_perfect(dataset, tmp_path, capsys):
    ann_dir, img_dir, _ = dataset
    nnb = _load_nn_baseline()
    out = tmp_path / "nn.json"
    nnb.main(["--train-annotations", ann_dir, "--train-images", img_dir,
              "--eval-annotations", ann_dir, "--eval-images", img_dir,
              "--out", str(out), "--n-boot", "0"])
    with open(out, encoding="utf-8") as f:
        s = json.load(f)
    assert s["strict"]["pck@5px"] == 1.0
    assert s["meta"]["train_eval_filename_overlap"] == 4
    with open(tmp_path / "nn_rows.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert all(r["neighbour"] == r["name"] for r in rows)


def test_nn_baseline_copies_labels_across_aspect_ratios(tmp_path):
    nnb = _load_nn_baseline()
    rng = np.random.default_rng(5)
    img = rng.integers(0, 255, (60, 120, 3), dtype=np.uint8)
    kps = _court_kps(60, 120, rng)
    vis = np.ones(NUM_KEYPOINTS)
    vis[0] = 0
    item = {"name": "t", "image_path": None, "ann": {"keypoints": kps.tolist(), "visibility": vis.tolist()}}
    canvas, cvis = nnb.canvas_labels(item["ann"], 60, 120)
    assert cvis[0] == 0
    # invisible K0 filled from the GT homography, not left at the centre
    info = letterbox_info(60, 120, 640)
    assert np.allclose(canvas[0], keypoints_to_letterbox(kps[:1], info)[0], atol=1e-6)

    predictor = nnb.NearestNeighbourPredictor.__new__(nnb.NearestNeighbourPredictor)
    predictor.embed = nnb.ncc_descriptor
    predictor.feats = np.stack([nnb.ncc_descriptor(img)])
    predictor.labels = [(canvas, cvis)]
    predictor.names = ["t"]
    det = predictor.predict(cv2.resize(img, (240, 120)))  # same aspect, 2x size
    assert np.allclose(det["keypoints"][1:], kps[1:], atol=1e-6)
    assert det["visibility"][0] == 0

import json
import numpy as np
import pytest
from src.tools.annotator import AnnotationState, save_annotation, load_annotation


def test_annotation_state_init():
    state = AnnotationState()
    assert len(state.keypoints) == 30
    assert all(v == 0 for v in state.visibility)


def test_set_keypoint():
    state = AnnotationState()
    state.set_keypoint(0, 0.5, 0.3)
    assert state.keypoints[0] == [0.5, 0.3]
    assert state.visibility[0] == 1


def test_clear_keypoint():
    state = AnnotationState()
    state.set_keypoint(0, 0.5, 0.3)
    state.clear_keypoint(0)
    assert state.keypoints[0] == [-1, -1]
    assert state.visibility[0] == 0


def test_undo_redo():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.2)
    state.set_keypoint(1, 0.9, 0.2)
    assert state.visibility[1] == 1

    state.undo()
    assert state.visibility[1] == 0
    assert state.visibility[0] == 1

    state.redo()
    assert state.visibility[1] == 1


def test_visible_count():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(1, 0.9, 0.1)
    state.set_keypoint(2, 0.9, 0.9)
    assert state.visible_count() == 3


def test_save_and_load(tmp_path):
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(1, 0.9, 0.1)
    state.set_keypoint(2, 0.9, 0.9)
    state.set_keypoint(3, 0.1, 0.9)
    out_path = str(tmp_path / "annotation.json")
    result = save_annotation(state, "test_frame.jpg", out_path)

    assert result["image_path"] == "test_frame.jpg"
    assert len(result["keypoints"]) == 30
    assert len(result["visibility"]) == 30
    assert result["visibility"][0] == 1
    assert result["visibility"][4] == 0

    loaded = load_annotation(out_path)
    assert loaded.keypoints[0] == [0.1, 0.1]
    assert loaded.visibility[0] == 1


def test_get_bounding_box():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(1, 0.9, 0.1)
    state.set_keypoint(2, 0.9, 0.9)
    state.set_keypoint(3, 0.1, 0.9)
    x_min, y_min, w, h = state.get_bounding_box()
    assert abs(x_min - 0.1) < 0.01
    assert abs(y_min - 0.1) < 0.01
    assert abs(w - 0.8) < 0.01
    assert abs(h - 0.8) < 0.01


def test_get_homography_none_with_few_keypoints():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(1, 0.9, 0.1)
    assert state.get_homography() is None


def test_get_homography_and_project_full_template():
    state = AnnotationState()
    # Place the 4 outer corners (K0, K4, K25, K29).
    state.set_keypoint(0, 0.0, 0.0)
    state.set_keypoint(4, 0.0, 1.0)
    state.set_keypoint(25, 1.0, 0.0)
    state.set_keypoint(29, 1.0, 1.0)

    H = state.get_homography()
    assert H is not None
    assert H.shape == (3, 3)

    projected = state.project_full_template()
    assert projected is not None
    assert projected.shape == (30, 2)
    # K0 (real-world origin) should project back near normalized (0, 0).
    assert abs(projected[0, 0] - 0.0) < 1e-6
    assert abs(projected[0, 1] - 0.0) < 1e-6


def test_is_corner_quad_valid_true_for_rectangle():
    state = AnnotationState()
    # CORNER_INDICES = [0, 25, 29, 4] — TL, TR, BR, BL cyclic order
    state.set_keypoint(0, 0.1, 0.1)   # top-left
    state.set_keypoint(25, 0.9, 0.1)  # top-right
    state.set_keypoint(29, 0.9, 0.9)  # bottom-right
    state.set_keypoint(4, 0.1, 0.9)   # bottom-left
    assert state.is_corner_quad_valid() is True


def test_is_corner_quad_valid_false_when_missing_corner():
    state = AnnotationState()
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(4, 0.1, 0.9)
    state.set_keypoint(25, 0.9, 0.1)
    # K29 not placed
    assert state.is_corner_quad_valid() is False


def test_is_corner_quad_valid_false_for_self_intersecting():
    state = AnnotationState()
    # Bowtie ordering: swap positions to create self-intersection.
    state.set_keypoint(0, 0.1, 0.1)
    state.set_keypoint(4, 0.9, 0.9)
    state.set_keypoint(25, 0.9, 0.1)
    state.set_keypoint(29, 0.1, 0.9)
    assert state.is_corner_quad_valid() is False


# ---------------------------------------------------------------------------
# Canonical labels, origins, basename image_path (Tkinter logic)
# ---------------------------------------------------------------------------

import os
import re
import shutil
import subprocess

import cv2

from src.court_geometry import (
    COURT_KEYPOINTS_TEMPLATE,
    SYMMETRY_PERMS,
    canonicalize,
    court_orientation,
)

HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "src", "tools", "annotator.html")


def _court_pixels(w=800, h=600):
    src = COURT_KEYPOINTS_TEMPLATE[[0, 4, 25, 29]].astype(np.float32)
    dst = np.float32([[0.12 * w, 0.95 * h], [0.88 * w, 0.95 * h], [0.38 * w, 0.2 * h], [0.62 * w, 0.2 * h]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    return cv2.perspectiveTransform(COURT_KEYPOINTS_TEMPLATE[None], Hm)[0]


def _state_with(perm, idx, w=800, h=600):
    kp = _court_pixels(w, h)[SYMMETRY_PERMS[perm]] / [w, h]
    state = AnnotationState()
    for i in idx:
        state.set_keypoint(i, *kp[i])
    return state


def test_auto_suggest_inner_works_without_k0_to_k3():
    state = _state_with("identity", [5, 9, 20, 24])
    n = state.auto_suggest_inner()
    assert n > 0
    full = _court_pixels() / [800, 600]
    for i in range(30):
        if state.visibility[i] and i not in (5, 9, 20, 24):
            assert state.origin[i] == "suggested"
            np.testing.assert_allclose(state.keypoints[i], full[i], atol=1e-6)
    assert state.origin[5] == "clicked"
    state.undo()
    assert state.visible_count() == 4


def test_auto_suggest_inner_needs_homography_support():
    state = _state_with("identity", [0, 1, 2, 3, 4])  # one court line only
    assert state.auto_suggest_inner() == 0


def test_save_canonicalizes_and_uses_basename(tmp_path):
    state = _state_with("rot180", list(range(30)))
    out = str(tmp_path / "a.json")
    data = save_annotation(state, os.path.join("some", "dir", "img.jpg"), out, image_size=(600, 800))
    assert data["image_path"] == "img.jpg"
    assert data["canonical_perm"] == "rot180"
    assert data["source"] == "manual"
    assert data["keypoint_origin"] == ["clicked"] * 30
    hand, align = court_orientation(data["keypoints"], data["visibility"], data["image_size"])
    assert hand > 0 and align >= 0
    with open(out, encoding="utf-8") as f:
        assert json.load(f)["canonical_perm"] == "rot180"
    # windows-style path is reduced to the basename too
    data = save_annotation(state, "C:\\x\\y\\img2.jpg", out, image_size=(600, 800))
    assert data["image_path"] == "img2.jpg"


def test_save_undeterminable_orientation_records_null(tmp_path):
    state = _state_with("identity", [0, 1, 2])
    data = save_annotation(state, "a.jpg", str(tmp_path / "a.json"), image_size=(600, 800))
    assert data["canonical_perm"] is None


def test_legacy_annotation_has_unknown_origin(tmp_path):
    legacy = {"image_path": "x.jpg", "image_size": [600, 800],
              "keypoints": [[0.5, 0.5]] + [[-1, -1]] * 29, "visibility": [1] + [0] * 29,
              "bounding_box": [0, 0, 0, 0]}
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy), encoding="utf-8")
    state = load_annotation(str(p))
    assert state.origin[0] is None
    assert "keypoint_origin" not in state.to_dict("x.jpg", (600, 800))
    state.set_keypoint(0, 0.4, 0.4)  # re-clicked -> known again
    assert state.to_dict("x.jpg", (600, 800))["keypoint_origin"][0] == "clicked"


def test_canonicalize_permutes_origin_and_is_undoable():
    state = _state_with("col_flip", list(range(30)))
    state.origin[3] = "suggested"
    assert state.canonicalize((600, 800)) == "col_flip"
    assert state.origin[SYMMETRY_PERMS["col_flip"][3]] == "suggested"
    assert state.canonical_perm == "col_flip"
    state.undo()
    assert state.canonical_perm == "identity" and state.origin[3] == "suggested"


# ---------------------------------------------------------------------------
# annotator.html must agree with court_geometry
# ---------------------------------------------------------------------------

def _html():
    with open(HTML_PATH, encoding="utf-8") as f:
        return f.read()


def test_html_symmetry_perms_match_court_geometry():
    m = re.search(r"const SYMMETRY_PERMS=\{(.*?)\};", _html(), re.S)
    assert m, "SYMMETRY_PERMS literal not found in annotator.html"
    perms = dict(re.findall(r"(\w+):\[([\d,\s]+)\]", m.group(1)))
    assert {k: [int(x) for x in v.split(",")] for k, v in perms.items()} == SYMMETRY_PERMS


def test_html_template_matches_court_geometry():
    html = _html()
    ns = {}
    for decl in re.findall(r"^const ((?:_?[A-Z]+=[^;]+,?\s*)+);", html, re.M):
        for name, expr in re.findall(r"(_?[A-Z]+)=([^,;]+)", decl):
            if re.fullmatch(r"[\w.+\-*/ ()]+", expr):
                ns[name] = eval(expr, {}, ns)
    body = re.search(r"const TPL=\[(.*?)\];", html, re.S).group(1)
    body = re.sub(r"//[^\n]*", "", body)
    tpl = np.array(eval("[" + body + "]", {}, ns), dtype=np.float64)
    np.testing.assert_allclose(tpl, COURT_KEYPOINTS_TEMPLATE, atol=1e-12)


def _js_block(html, start, end):
    i = html.index(start)
    return html[i:html.index(end, i)]


def _node_canonical(cases):
    """Run the annotator's JS canonicalization on cases [(kps, vis, w, h)]."""
    html = _html()
    js = "\n".join([
        _js_block(html, "/* ---- Constants", "/* ---- State ---- */"),
        _js_block(html, "/* ---- Homography (4-point DLT) ---- */", "/* ---- Canonical labeling"),
        _js_block(html, "/* BEGIN CANONICAL */", "/* END CANONICAL */"),
        re.search(r"let csc=1,cox=0,coy=0;\s*function c2s\(cx,cy\)\{[^\n]*\}", html).group(0),
        "const cases=JSON.parse(require('fs').readFileSync(0,'utf8'));",
        "const out=cases.map(([k,v,w,h])=>canonicalPermName(k,v,w,h));",
        # the court diagram drawn by c2s, treated as an image of size 61 x 134
        "csc=10;const scr=TPL.map(([x,y])=>c2s(x,y));",
        "out.push(canonicalPermName(scr.map(([x,y])=>[x/61,y/134]),Array(NK).fill(1),61,134));",
        "process.stdout.write(JSON.stringify(out));",
    ])
    res = subprocess.run(["node", "-e", js], input=json.dumps(cases), capture_output=True,
                         text=True, encoding="utf-8", timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_html_canonicalization_matches_python():
    rng = np.random.default_rng(0)
    cases, expected = [], []
    for trial in range(40):
        w, h = [(800, 600), (600, 800), (1000, 400)][trial % 3]
        jitter = rng.normal(0, 0.08, (4, 2)) * [w, h]
        src = COURT_KEYPOINTS_TEMPLATE[[0, 4, 25, 29]].astype(np.float32)
        base = np.array([[0.12, 0.95], [0.88, 0.95], [0.38, 0.2], [0.62, 0.2]]) * [w, h]
        if trial % 2:  # side view
            base = np.array([[0.05, 0.3], [0.05, 0.9], [0.95, 0.3], [0.95, 0.9]]) * [w, h]
        Hm = cv2.getPerspectiveTransform(src, (base + jitter).astype(np.float32))
        kp = cv2.perspectiveTransform(COURT_KEYPOINTS_TEMPLATE[None], Hm)[0] / [w, h]
        perm = list(SYMMETRY_PERMS)[trial % 4]
        vis = (rng.random(30) < 0.6).astype(int)
        kp, vis = kp[SYMMETRY_PERMS[perm]], vis[SYMMETRY_PERMS[perm]]
        kps = [list(p) if v else [-1.0, -1.0] for p, v in zip(kp.tolist(), vis)]
        cases.append([kps, vis.tolist(), w, h])
        expected.append(canonicalize(kps, vis, [h, w])[2])
    got = _node_canonical(cases)
    assert got[:-1] == expected
    # the court diagram is drawn in the canonical orientation (K0 near-left)
    assert got[-1] == "identity"

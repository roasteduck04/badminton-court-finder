"""Blender smoke + regression tests for the synthetic data pipeline.

Skipped under the project Python (no bpy). Run inside Blender:
    blender --background --factory-startup --python tests/test_blender_smoke.py

Exits with code 0 on success, 1 on failure.
"""

import os
import random
import sys

try:
    import pytest
    pytest.importorskip("bpy")
except ImportError:  # Blender's Python usually has no pytest; run via __main__
    pytest = None

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.tools.blender_common import (  # noqa: E402
    KEYPOINTS_TEMPLATE, NET_POS, NET_HEIGHT_EDGE, NET_HEIGHT_CENTER, NET_DEPTH, reset_scene,
)


def _fresh():
    reset_scene()
    random.seed(1234)


def _homography(src, dst):
    """DLT template->image homography (numpy only, no cv2 in Blender)."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(A))
    return vt[-1].reshape(3, 3)


def _handedness(kp2d, vis):
    src = np.array(KEYPOINTS_TEMPLATE)[np.asarray(vis) > 0]
    dst = np.asarray(kp2d, dtype=float)[np.asarray(vis) > 0]
    H = _homography(src, dst)
    c = src.mean(axis=0)
    w = H[2, 0] * c[0] + H[2, 1] * c[1] + H[2, 2]
    p = (H @ np.array([c[0], c[1], 1.0]))[:2] / w
    J = (H[:2, :2] - np.outer(p, H[2, :2])) / w
    return 1 if np.linalg.det(J) > 0 else -1


def test_reset_scene_leaves_nothing():
    from src.tools.blender_court import build_court
    from src.tools.blender_environment import build_environment
    _fresh()
    build_court({"include_net": True})
    build_environment()
    reset_scene()
    assert len(bpy.data.objects) == 0
    assert len(bpy.data.meshes) == 0
    assert len(bpy.data.materials) == 0
    assert "Cube" not in bpy.data.objects


def test_full_pipeline_and_upright_geometry():
    from src.tools.blender_court import build_court
    from src.tools.blender_environment import build_environment
    from src.tools.blender_lighting import setup_lighting
    from src.tools.blender_venue_details import build_venue_details
    from src.tools.blender_occluders import add_occluders
    _fresh()
    court = build_court({"include_net": True})
    assert len(court["keypoints"]) == 30 and court["net"] is not None and len(court["posts"]) == 2
    env = build_environment({"court_style": court["style"], "divider_chance": 1.0})
    light = setup_lighting("fluorescent", {"court_offsets": env["court_offsets"],
                                           "ceiling_height": env["venue_bounds"]["h"]})
    assert light["preset"] == "fluorescent" and light["lights"]
    details = build_venue_details({"venue_bounds": env["venue_bounds"],
                                   "footprint": env["footprint"]})
    for key in ("num_wall_banners", "seating_type", "spectator_count", "bleacher_placements"):
        assert key in details
    add_occluders({"no_occluder_chance": 0.0})
    bpy.context.view_layer.update()

    # Nothing upright may be collapsed to zero height.
    upright = ("WallBanner", "ExitSign", "Window", "HangBanner_", "PopupBanner", "Towel",
               "Divider", "Scoreboard")
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name.startswith(upright):
            zs = [(obj.matrix_world @ Vector(c)).z for c in obj.bound_box]
            assert max(zs) - min(zs) > 0.1, obj.name

    # Bleachers sit outside the footprint of all courts.
    fp = env["footprint"]
    for obj in bpy.data.objects:
        if obj.name.startswith("Bleacher_"):
            pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            xs, ys = [p.x for p in pts], [p.y for p in pts]
            inside_x = max(xs) > fp["x0"] and min(xs) < fp["x1"]
            inside_y = max(ys) > fp["y0"] and min(ys) < fp["y1"]
            assert not (inside_x and inside_y), obj.name

    # Real net: 0.76 m tall, top 1.55 m at the posts and ~1.524 m at the centre.
    net = bpy.data.objects["NetMesh"]
    zs = [v.co.z for v in net.data.vertices]
    assert abs(max(zs) - NET_HEIGHT_EDGE) < 1e-3
    assert abs(max(zs) - min(zs) - (NET_HEIGHT_EDGE - NET_HEIGHT_CENTER + NET_DEPTH)) < 1e-2
    centre = min(net.data.vertices, key=lambda v: abs(v.co.y - 3.05) - v.co.z * 1e-6)
    assert abs(max(v.co.z for v in net.data.vertices if abs(v.co.y - centre.co.y) < 1e-6)
               - NET_HEIGHT_CENTER) < 2e-3


def test_camera_handedness_and_projection_regression():
    """World-fixed labels: every strategy yields the same (negative) handedness."""
    from src.tools.blender_court import build_court
    from src.tools.blender_environment import build_environment
    from src.tools.blender_venue_details import build_venue_details
    from src.tools.blender_camera import place_camera, STRATEGIES
    for i, strategy in enumerate(STRATEGIES):
        _fresh()
        random.seed(100 + i)
        court = build_court({"include_net": True})
        env = build_environment({"court_style": court["style"], "divider_chance": 0.0})
        det = build_venue_details({"venue_bounds": env["venue_bounds"], "footprint": env["footprint"]})
        _, meta = place_camera(court["keypoints"], strategy, {"resolution": (640, 360)},
                               {"bounds": env["venue_bounds"], "court_offsets": env["court_offsets"],
                                "bleachers": det["bleacher_placements"]})
        assert len(meta["keypoints_2d"]) == 30 and len(meta["keypoint_occluded"]) == 30
        assert abs(meta["roll_deg"]) <= 12.0 + 1e-9
        assert 13.0 - 1e-6 <= meta["focal_length"] <= 200.0 + 1e-6
        assert meta["court_coverage"] >= 0.12
        K = np.array(meta["intrinsics"]["K"])
        R = np.array(meta["extrinsics_cv"]["R"])
        t = np.array(meta["extrinsics_cv"]["t"])
        X = np.array(meta["keypoints_3d"])
        xc = (R @ X.T).T + t
        front = xc[:, 2] > 0
        proj = (K @ xc.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        assert np.abs(proj[front] - np.array(meta["keypoints_2d"])[front]).max() < 0.05, strategy
        assert _handedness(meta["keypoints_2d"], front) == -1, strategy


def test_ray_cast_occlusion_ignores_net_and_floor():
    from src.tools.blender_court import build_court
    from src.tools.blender_camera import keypoint_occlusion
    from src.tools.blender_common import add_box, new_collection
    _fresh()
    court = build_court({"include_net": True})
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    origin = (NET_POS + 3.0, 3.05, 1.5)       # ray to K2 crosses the net at ~1.04 m
    k2 = court["keypoints"][2]
    dg = bpy.context.evaluated_depsgraph_get()
    assert keypoint_occlusion(scene, dg, origin, [k2]) == [False]
    col = new_collection("Blocker")
    add_box("Blocker", (3.0, 3.05, 0.6), (0.5, 0.5, 0.5), col)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    assert keypoint_occlusion(scene, dg, origin, [k2]) == [True]


def test_mannequin_limbs_stay_attached_and_clear_of_net():
    from src.tools.blender_common import new_collection
    from src.tools.blender_mannequin import build_mannequin, UPPER_ARM, UPPER_LEG
    _fresh()
    col = new_collection("Occluders")
    for i in range(10):
        meta = build_mannequin(col, i, {"position": (NET_POS - 1.25, 3.0), "facing": 0.0})
        bpy.context.view_layer.update()
        objs = {o.name: o for o in col.objects if o.name.startswith(f"Player_{i}_")}
        for s in "LR":
            ua, fa = objs[f"Player_{i}_UpperArm_{s}"], objs[f"Player_{i}_Forearm_{s}"]
            elbow = ua.matrix_world @ Vector((0, 0, -UPPER_ARM))
            assert (fa.matrix_world.translation - elbow).length < 1e-4
            ul, ll = objs[f"Player_{i}_UpperLeg_{s}"], objs[f"Player_{i}_LowerLeg_{s}"]
            knee = ul.matrix_world @ Vector((0, 0, -UPPER_LEG))
            assert (ll.matrix_world.translation - knee).length < 1e-4
        pts = [o.matrix_world @ Vector(c) for o in objs.values() if o.type == 'MESH' for c in o.bound_box]
        assert max(p.x for p in pts) <= NET_POS - 0.3 + 1e-6, meta
        assert abs(min(p.z for p in pts)) < 1e-4  # standing on the floor


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"  FAIL {t.__name__}: {e!r}")
    print(f"\n=== {len(tests) - failures}/{len(tests)} passed ===")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

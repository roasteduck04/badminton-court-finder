"""Blender smoke test — validates the full synthetic data pipeline.

Run inside Blender:
    blender --background --python tests/test_blender_smoke.py

Exits with code 0 on success, 1 on failure.
"""

import sys
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bpy
import random

random.seed(42)

errors = []


def check(condition, msg):
    if not condition:
        errors.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  OK: {msg}")


def test_full_pipeline():
    """Build a complete scene and validate all components."""
    from src.tools.blender_court import build_court
    from src.tools.blender_environment import build_environment
    from src.tools.blender_camera import place_camera
    from src.tools.blender_lighting import setup_lighting
    from src.tools.blender_venue_details import build_venue_details
    from src.tools.blender_occluders import add_occluders

    print("\n=== Blender Smoke Test ===\n")

    # 1. Build court
    print("1. Building court...")
    court = build_court({"surface_color": "random", "line_color": "random", "include_net": True})
    check(court["surface"] is not None, "Court surface created")
    check(len(court["keypoints"]) == 30, f"30 keypoints created (got {len(court['keypoints'])})")
    check(court["net"] is not None, "Net created")
    check(len(court["posts"]) == 2, "2 net posts created")
    check("Court" in bpy.data.collections, "Court collection exists")
    check("NetAssembly" in bpy.data.collections, "NetAssembly collection exists")

    # Check net post improvements
    net_col = bpy.data.collections["NetAssembly"]
    net_obj_names = [o.name for o in net_col.objects]
    check(any("BasePlate" in n for n in net_obj_names), "Post base plates created")
    check(any("TensionCable" in n for n in net_obj_names), "Tension cable created")

    # 2. Build environment
    print("\n2. Building environment...")
    env_meta = build_environment()
    check("adjacent_courts" in env_meta, "adjacent_courts in metadata")
    check("venue_size" in env_meta, "venue_size in metadata")
    check("floor_type" in env_meta, "floor_type in metadata")
    check(env_meta["floor_type"] in ["concrete", "rubber", "wood"],
          f"Valid floor type: {env_meta['floor_type']}")
    check("venue_bounds" in env_meta, "venue_bounds in metadata")
    bounds = env_meta["venue_bounds"]
    check(all(k in bounds for k in ["cx", "cy", "d", "w", "h"]),
          "venue_bounds has all required keys")

    # 3. Camera
    print("\n3. Placing camera...")
    cam, cam_meta = place_camera(
        court["keypoints"], strategy="broadcast",
        config={"resolution": (640, 640)},
    )
    check(cam is not None, "Camera created")
    check("visibility" in cam_meta, "visibility in camera metadata")
    check(len(cam_meta["visibility"]) == 30, "30 visibility values")

    # 4. Lighting
    print("\n4. Setting up lighting...")
    light_meta = setup_lighting("fluorescent")
    check("preset" in light_meta, "preset in lighting metadata")
    check("lights" in light_meta, "lights in lighting metadata")
    check("Lighting" in bpy.data.collections, "Lighting collection exists")

    # 5. Venue details
    print("\n5. Building venue details...")
    details_meta = build_venue_details({
        "venue_bounds": bounds,
        "lighting_preset": "fluorescent",
    })
    check("num_wall_banners" in details_meta, "num_wall_banners in metadata")
    check("has_light_housings" in details_meta, "has_light_housings in metadata")
    check("seating_type" in details_meta, "seating_type in metadata")
    check("spectator_count" in details_meta, "spectator_count in metadata")
    check("clutter_items" in details_meta, "clutter_items in metadata")
    check("VenueDetails" in bpy.data.collections, "VenueDetails collection exists")

    # 6. Occluders
    print("\n6. Adding occluders...")
    occ_meta = add_occluders({"no_occluder_chance": 0})
    check("occluders" in occ_meta, "occluders in metadata")
    players = [o for o in occ_meta["occluders"] if o["type"] == "player"]
    if players:
        p = players[0]
        check("pose" in p, "Player has pose field")
        check("has_racket" in p, "Player has has_racket field")
        check("facing_angle" in p, "Player has facing_angle field")
        check(p["pose"] in ["standing", "ready", "lunging", "serving", "walking"],
              f"Valid pose: {p['pose']}")

    # Summary
    print(f"\n=== Results: {len(errors)} errors ===")
    if errors:
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    test_full_pipeline()

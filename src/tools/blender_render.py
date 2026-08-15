"""Batch renderer: wires all Blender modules together for synthetic data generation.

Usage:
    blender --background court_template.blend --python src/tools/blender_render.py -- --count 500
"""

import bpy
import json
import os
import random
import sys
import time

# Parse args after "--"
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

# Simple arg parsing
def _get_arg(flag, default):
    if flag in argv:
        return argv[argv.index(flag) + 1]
    return default

COUNT = int(_get_arg("--count", "10"))
START = int(_get_arg("--start", "1"))
SEED = int(_get_arg("--seed", "-1"))
ENGINE = _get_arg("--engine", "CYCLES")
SAMPLES = int(_get_arg("--samples", "64"))
RES_MIN = int(_get_arg("--res-min", "640"))
RES_MAX = int(_get_arg("--res-max", "1280"))

# Find project root (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Add project root to path so we can import modules
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.join(ROOT, "data", "blender", "raw")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")

STRATEGY_WEIGHTS = {
    "broadcast": 0.30,
    "sideline": 0.20,
    "corner": 0.20,
    "overhead": 0.15,
    "low": 0.10,
    "random": 0.05,
}

LIGHTING_PRESETS = ["fluorescent", "mixed", "dim", "harsh", "competition"]


def _weighted_choice(weights_dict):
    """Pick a key from a dict of {key: weight}."""
    keys = list(weights_dict.keys())
    weights = [weights_dict[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def render_batch():
    """Generate COUNT synthetic court images with metadata."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    if SEED >= 0:
        random.seed(SEED)

    # Configure render engine
    bpy.context.scene.render.engine = ENGINE
    if ENGINE == "CYCLES":
        bpy.context.scene.cycles.samples = SAMPLES
        bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Set world ambient so no scene is ever fully black
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.12, 0.12, 0.12, 1.0)
        bg.inputs["Strength"].default_value = 1.0

    from src.tools.blender_court import build_court
    from src.tools.blender_camera import place_camera
    from src.tools.blender_lighting import setup_lighting
    from src.tools.blender_occluders import add_occluders
    from src.tools.blender_environment import build_environment
    from src.tools.blender_venue_details import build_venue_details

    print(f"\n=== Rendering {COUNT} images ===")
    print(f"  Engine: {ENGINE}, Samples: {SAMPLES}")
    print(f"  Output: {OUTPUT_DIR}\n")

    for i in range(COUNT):
        t0 = time.time()
        idx = f"{START + i:04d}"

        # 0. Randomize resolution
        res = random.randint(RES_MIN, RES_MAX)
        # Round to nearest multiple of 32 for GPU efficiency
        res = (res // 32) * 32
        bpy.context.scene.render.resolution_x = res
        bpy.context.scene.render.resolution_y = res

        # 1. Build court with random colors
        court = build_court({
            "surface_color": "random",
            "line_color": "random",
            "include_net": random.random() > 0.1,
        })

        # 2. Build environment
        env_meta = build_environment()

        # 3. Place camera
        strategy = _weighted_choice(STRATEGY_WEIGHTS)
        cam, cam_meta = place_camera(
            court["keypoints"], strategy=strategy,
            config={"resolution": (res, res)},
        )

        # 4. Setup lighting
        preset = random.choice(LIGHTING_PRESETS)
        light_meta = setup_lighting(preset)

        # 5. Build venue details (after lighting, needs light positions)
        details_meta = build_venue_details({
            "venue_bounds": env_meta.get("venue_bounds", {}),
            "lighting_preset": preset,
        })

        # 6. Add occluders
        occ_meta = add_occluders()

        # 7. Render
        img_name = f"blender_{idx}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        bpy.context.scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)

        # 8. Export metadata
        env_export = {
            "adjacent_courts": env_meta["adjacent_courts"],
            "venue_size": env_meta["venue_size"],
            "has_dividers": env_meta["has_dividers"],
            "has_scoreboard": env_meta["has_scoreboard"],
            "floor_type": env_meta.get("floor_type", "unknown"),
            "has_floor_markings": env_meta.get("has_floor_markings", False),
        }
        env_export.update(details_meta)

        metadata = {
            "image_file": img_name,
            "resolution": [res, res],
            "camera": cam_meta,
            "lighting": light_meta,
            "occluders": occ_meta["occluders"],
            "court": {
                "surface_color": court["surface_color"],
                "line_color": court["line_color"],
            },
            "environment": env_export,
        }

        meta_path = os.path.join(METADATA_DIR, f"blender_{idx}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - t0
        vis = sum(cam_meta["visibility"])
        details_count = sum(1 for k in details_meta if details_meta[k])
        print(f"  [{i + 1}/{COUNT}] {img_name} | {strategy} | {preset} | "
              f"{vis}/30 kp | {details_count} details | {elapsed:.1f}s")

    print(f"\nDone: {COUNT} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    render_batch()

"""Batch renderer: wires all Blender modules together for synthetic data generation.

Usage:
    blender --background --factory-startup --python src/tools/blender_render.py -- \
        --count 500 --seed 42 --engine BLENDER_EEVEE --samples 32

Every image gets its own seed = hash(global seed, image index), so batches
rendered with different --start values are distinct and any single image
can be reproduced. The scene is rebuilt from scratch for every image (all
data-blocks removed and orphans purged), so nothing leaks between frames.

Metadata (one JSON per image, read by src/tools/blender_to_cvn.py):
    image_file, resolution [w, h], image_size [h, w], seed, global_seed,
    image_index, include_net, camera {intrinsics, extrinsics_cv, pose,
    keypoints_2d (all 30, template order), visibility (in frame),
    keypoint_occluded (ray cast), ...}, keypoint_occluded (copy), lighting,
    color_management, occluders, court, environment.
"""

import argparse
import json
import os
import random
import sys
import time

import bpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4, "1:1": 1.0}
# Real phone photos in the dataset are mostly portrait; broadcast is 16:9.
DEFAULT_ASPECT_WEIGHTS = {"16:9": 0.30, "9:16": 0.25, "4:3": 0.15, "3:4": 0.20, "1:1": 0.10}
LIGHTING_PRESETS = ["fluorescent", "mixed", "dim", "harsh", "competition"]
VIEW_TRANSFORMS = {"AgX": 0.5, "Filmic": 0.25, "Standard": 0.25}
LAST_RESET_COUNTS = {}


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="blender_render.py")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--start", type=int, default=1, help="index of the first image")
    p.add_argument("--seed", type=int, default=42, help="global seed; -1 = random")
    p.add_argument("--engine", default="BLENDER_EEVEE", help="BLENDER_EEVEE or CYCLES")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--res-min", type=int, default=480, help="min SHORT side (px, >= 480)")
    p.add_argument("--res-max", type=int, default=720, help="max SHORT side (px)")
    p.add_argument("--aspects", default=",".join(DEFAULT_ASPECT_WEIGHTS),
                   help="comma list from 16:9,9:16,4:3,3:4,1:1 (default weights when unchanged)")
    p.add_argument("--strategies", default="",
                   help="comma list restricting camera strategies (default: all, weighted)")
    p.add_argument("--min-usable", type=int, default=6,
                   help="min in-frame & unoccluded keypoints per image")
    p.add_argument("--output", default=os.path.join(ROOT, "data", "blender", "raw"))
    args = p.parse_args(argv)
    args.res_min = max(480, args.res_min)
    args.res_max = max(args.res_min, args.res_max)
    return args


def _sample_resolution(args):
    names = [a.strip() for a in args.aspects.split(",") if a.strip()]
    for n in names:
        if n not in ASPECTS:
            raise ValueError(f"Unknown aspect ratio {n!r}; choose from {list(ASPECTS)}")
    if set(names) == set(DEFAULT_ASPECT_WEIGHTS):
        weights = [DEFAULT_ASPECT_WEIGHTS[n] for n in names]
    else:
        weights = [1.0] * len(names)
    name = random.choices(names, weights=weights)[0]
    ratio = ASPECTS[name]
    short = random.randint(args.res_min, args.res_max) // 8 * 8
    long_ = int(round(short * max(ratio, 1 / ratio) / 8)) * 8
    w, h = (long_, short) if ratio >= 1 else (short, long_)
    return name, w, h


def _configure_engine(scene, engine, samples):
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    else:
        scene.eevee.taa_render_samples = samples
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False


def _color_management(scene):
    """Random view transform, exposure and white balance."""
    vs = scene.view_settings
    names = list(VIEW_TRANSFORMS)
    chosen = random.choices(names, weights=[VIEW_TRANSFORMS[n] for n in names])[0]
    for name in [chosen] + names:
        try:
            vs.view_transform = name
            break
        except TypeError:
            continue
    try:
        vs.look = 'None'
    except TypeError:
        pass
    vs.exposure = max(-1.0, min(1.0, random.gauss(0.0, 0.4)))
    vs.gamma = random.uniform(0.9, 1.1)
    meta = {"view_transform": vs.view_transform, "exposure": round(vs.exposure, 3),
            "gamma": round(vs.gamma, 3)}
    if hasattr(vs, "use_white_balance"):
        vs.use_white_balance = True
        vs.white_balance_temperature = random.uniform(4300, 8000)
        vs.white_balance_tint = random.uniform(-12, 12)
        meta["white_balance_temperature"] = round(vs.white_balance_temperature, 1)
        meta["white_balance_tint"] = round(vs.white_balance_tint, 2)
    return meta


def datablock_counts():
    return {"objects": len(bpy.data.objects), "meshes": len(bpy.data.meshes),
            "materials": len(bpy.data.materials), "lights": len(bpy.data.lights),
            "cameras": len(bpy.data.cameras), "collections": len(bpy.data.collections)}


def build_and_render(image_index, image_seed_value, args, images_dir, metadata_dir, global_seed):
    """Build one scene from scratch, render it and write its metadata."""
    from src.tools.blender_common import reset_scene
    from src.tools.blender_court import build_court, sample_court_style, style_metadata
    from src.tools.blender_camera import place_camera, STRATEGY_WEIGHTS
    from src.tools.blender_lighting import setup_lighting
    from src.tools.blender_occluders import add_occluders
    from src.tools.blender_environment import build_environment
    from src.tools.blender_venue_details import build_venue_details

    random.seed(image_seed_value)
    reset_scene()
    global LAST_RESET_COUNTS
    LAST_RESET_COUNTS = datablock_counts()
    scene = bpy.context.scene
    _configure_engine(scene, args.engine, args.samples)

    aspect, res_x, res_y = _sample_resolution(args)
    include_net = random.random() > 0.1

    style = sample_court_style()
    court = build_court({"include_net": include_net, "style": style})
    env = build_environment({"court_style": style})
    preset = random.choice(LIGHTING_PRESETS)
    light_meta = setup_lighting(preset, {"court_offsets": env["court_offsets"],
                                         "ceiling_height": env["venue_bounds"]["h"]})
    details = build_venue_details({"venue_bounds": env["venue_bounds"],
                                   "footprint": env["footprint"], "lighting_preset": preset})
    occ = add_occluders()

    allowed = [s.strip() for s in args.strategies.split(",") if s.strip()] or list(STRATEGY_WEIGHTS)
    strategy = random.choices(allowed, weights=[STRATEGY_WEIGHTS[s] for s in allowed])[0]
    _cam, cam_meta = place_camera(
        court["keypoints"], strategy=strategy,
        config={"resolution": (res_x, res_y), "min_usable": args.min_usable},
        scene_info={"bounds": env["venue_bounds"], "court_offsets": env["court_offsets"],
                    "bleachers": details["bleacher_placements"]},
    )
    cm_meta = _color_management(scene)

    stem = f"blender_{image_index:04d}"
    img_name = f"{stem}.png"
    scene.render.filepath = os.path.join(images_dir, img_name)
    bpy.ops.render.render(write_still=True)

    env_export = {k: env[k] for k in ("adjacent_courts", "venue_size", "has_dividers",
                                      "has_scoreboard", "floor_type", "has_floor_markings")}
    env_export["court_offsets"] = env["court_offsets"]
    env_export.update(details)

    metadata = {
        "image_file": img_name,
        "resolution": [res_x, res_y],       # [width, height]
        "image_size": [res_y, res_x],       # [height, width] (CVN convention)
        "aspect": aspect,
        "seed": image_seed_value,
        "global_seed": global_seed,
        "image_index": image_index,
        "include_net": include_net,
        "render": {"engine": args.engine, "samples": args.samples},
        "camera": cam_meta,
        "keypoint_occluded": cam_meta["keypoint_occluded"],
        "lighting": light_meta,
        "color_management": cm_meta,
        "occluders": occ["occluders"],
        "court": style_metadata(court["style"]),
        "environment": env_export,
    }
    with open(os.path.join(metadata_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def render_batch(args=None):
    """Generate args.count synthetic court images with metadata."""
    args = args or parse_args()
    from src.tools.blender_common import image_seed

    # Start from an empty file: no default cube/camera/light in any scene.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    images_dir = os.path.join(args.output, "images")
    metadata_dir = os.path.join(args.output, "metadata")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    global_seed = args.seed if args.seed >= 0 else random.SystemRandom().randrange(2 ** 31)
    print(f"\n=== Rendering {args.count} images (global seed {global_seed}) ===")
    print(f"  Engine: {args.engine}, Samples: {args.samples}, short side {args.res_min}-{args.res_max}")
    print(f"  Output: {args.output}\n")

    for i in range(args.count):
        t0 = time.time()
        idx = args.start + i
        meta = build_and_render(idx, image_seed(global_seed, idx), args,
                                images_dir, metadata_dir, global_seed)
        cam = meta["camera"]
        n_in = sum(cam["in_frame"])
        n_occ = sum(1 for v, o in zip(cam["in_frame"], cam["keypoint_occluded"]) if v and o)
        print(f"  [{i + 1}/{args.count}] {meta['image_file']} {meta['resolution'][0]}x"
              f"{meta['resolution'][1]} | {cam['strategy_used']}/{cam['framing']} "
              f"f={cam['focal_length']:.0f}mm cov={cam['court_coverage']:.2f} roll={cam['roll_deg']:+.0f} | "
              f"{meta['lighting']['preset']} | {n_in}/30 in frame, {n_occ} occluded | "
              f"{time.time() - t0:.1f}s | blocks after reset {LAST_RESET_COUNTS}, "
              f"built {datablock_counts()}")
        sys.stdout.flush()

    print(f"\nDone: {args.count} images in {args.output}")


if __name__ == "__main__":
    render_batch()

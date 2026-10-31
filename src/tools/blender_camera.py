"""Blender camera placement with named strategies, plus keypoint projection
and ray-cast occlusion.

Cameras come from both ends and both sides, from 1 m (phone at the baseline)
to 15 m (overhead), from the stands and from adjacent courts, with roll,
off-centre look-at targets, partial-court framing and 13-85 mm focal lengths
(36 mm sensor on the long image side, i.e. full-frame equivalent).

The camera is placed AFTER the scene is complete so occlusion can be ray
cast against everything; a pose is rejected and resampled until enough
keypoints are in frame and unoccluded to determine a homography.
"""

import math
import random

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector

from src.tools.blender_common import (
    COURT_LENGTH, COURT_WIDTH, KEYPOINTS_TEMPLATE, CORNER_INDICES,
    OCCLUDER_PROP, has_homography_support, log_uniform, clamp,
)

STRATEGY_WEIGHTS = {
    "broadcast": 0.18,
    "end_low": 0.10,
    "sideline": 0.15,
    "corner": 0.15,
    "overhead": 0.08,
    "spectator": 0.17,
    "adjacent_court": 0.17,
}
STRATEGIES = list(STRATEGY_WEIGHTS)

FOCAL_RANGE = (13.0, 85.0)     # mm, 36 mm sensor on the long side
SENSOR_WIDTH = 36.0
ROLL_MAX_DEG = 12.0
PARTIAL_FRAMING_PROB = 0.3
OCCLUSION_EPS = 0.005           # hit must be this much closer than the keypoint
MIN_UNOCCLUDED_FRACTION = 0.5   # reject views where dividers etc. hide most of the court


def _sample_focal():
    """Mixture of phone ultra-wide, phone main camera, and zoom/tele (mm, FF-equiv)."""
    r = random.random()
    if r < 0.2:
        return random.uniform(13.0, 18.0)
    if r < 0.65:
        return random.uniform(22.0, 32.0)
    return log_uniform(random, 32.0, FOCAL_RANGE[1])

DEFAULT_CONFIG = {
    "resolution": (640, 480),   # (width, height)
    "min_usable": 6,            # in-frame & unoccluded keypoints required
    "max_tries": 60,
}


# --------------------------------------------------------------------------
# Position sampling
# --------------------------------------------------------------------------

def _sample_position(strategy, ctx):
    L, W = COURT_LENGTH, COURT_WIDTH
    end = random.choice([-1, 1])
    if strategy == "broadcast":
        d = random.uniform(2.0, 12.0)
        return Vector((-d if end < 0 else L + d, W / 2 + random.uniform(-2.5, 2.5),
                       random.uniform(3.5, 10.0)))
    if strategy == "end_low":
        d = random.uniform(0.3, 6.0)
        return Vector((-d if end < 0 else L + d, random.uniform(-1.0, W + 1.0),
                       random.uniform(1.0, 2.0)))
    if strategy == "sideline":
        d = random.uniform(1.0, 9.0)
        return Vector((random.uniform(-2.0, L + 2.0), -d if end < 0 else W + d,
                       random.uniform(1.2, 6.0)))
    if strategy == "corner":
        dx, dy = random.uniform(0.5, 7.0), random.uniform(0.5, 7.0)
        sx, sy = random.choice([-1, 1]), random.choice([-1, 1])
        return Vector((-dx if sx < 0 else L + dx, -dy if sy < 0 else W + dy,
                       random.uniform(1.3, 8.0)))
    if strategy == "overhead":
        return Vector((random.uniform(-2.0, L + 2.0), random.uniform(-3.0, W + 3.0),
                       random.uniform(9.0, 15.0)))
    if strategy == "spectator":
        placements = ctx.get("bleachers") or []
        if placements:
            p = random.choice(placements)
            r = random.randrange(p["num_rows"])
            depth = p["start"] + p["direction"] * (r + 0.5) * p["row_depth"]
            along = random.uniform(*p["span"])
            z = (r + 1) * 0.4 + random.uniform(0.9, 1.7)
            if p["side"] in ("back", "front"):
                return Vector((depth, along, z))
            return Vector((along, depth, z))
        # No stands: elevated gallery/balcony position near a wall.
        b = ctx["bounds"]
        wall = random.choice(["back", "front", "left", "right"])
        inset = random.uniform(0.5, 3.0)
        z = random.uniform(3.0, 10.0)
        if wall == "back":
            return Vector((b["x0"] + inset, random.uniform(b["y0"], b["y1"]), z))
        if wall == "front":
            return Vector((b["x1"] - inset, random.uniform(b["y0"], b["y1"]), z))
        if wall == "left":
            return Vector((random.uniform(b["x0"], b["x1"]), b["y0"] + inset, z))
        return Vector((random.uniform(b["x0"], b["x1"]), b["y1"] - inset, z))
    if strategy == "adjacent_court":
        offs = ctx.get("court_offsets") or []
        if offs:
            xo, yo = random.choice(offs)
            return Vector((xo + random.uniform(-1.0, L + 1.0), yo + random.uniform(-1.0, W + 1.0),
                           random.uniform(1.0, 2.2)))
        # No adjacent courts: stand further out to the side at phone height.
        d = random.uniform(3.0, 7.0)
        return Vector((random.uniform(-3.0, L + 3.0), -d if end < 0 else W + d,
                       random.uniform(1.0, 2.2)))
    raise ValueError(f"Unknown camera strategy: {strategy}")


def _clamp_to_venue(pos, bounds):
    return Vector((clamp(pos.x, bounds["x0"] + 0.4, bounds["x1"] - 0.4),
                   clamp(pos.y, bounds["y0"] + 0.4, bounds["y1"] - 0.4),
                   clamp(pos.z, 0.8, bounds["h"] - 0.6)))


def _sample_roll_deg():
    if random.random() < 0.3:
        return random.uniform(-ROLL_MAX_DEG, ROLL_MAX_DEG)
    return clamp(random.gauss(0.0, 4.0), -ROLL_MAX_DEG, ROLL_MAX_DEG)


def _look_rotation(pos, target, roll_deg):
    direction = target - pos
    quat = direction.to_track_quat('-Z', 'Y')
    return quat @ Quaternion((0.0, 0.0, 1.0), math.radians(roll_deg))


def _fit_focal(pos, quat, points, res):
    """Largest focal length (mm) that keeps every point in frame, or None."""
    R_inv = quat.to_matrix().transposed()
    w, h = res
    long_side = max(w, h)
    half_u = SENSOR_WIDTH / 2 * w / long_side   # tan(half fov) * f
    half_v = SENSOR_WIDTH / 2 * h / long_side
    f_max = float("inf")
    for p in points:
        pc = R_inv @ (Vector(p) - pos)
        depth = -pc.z
        if depth < 0.1:
            return None
        u, v = abs(pc.x / depth), abs(pc.y / depth)
        if u > 1e-9:
            f_max = min(f_max, half_u / u)
        if v > 1e-9:
            f_max = min(f_max, half_v / v)
    return f_max * 0.95


# --------------------------------------------------------------------------
# Projection, intrinsics, occlusion
# --------------------------------------------------------------------------

def camera_intrinsics(scene, cam_obj):
    """Pinhole intrinsics in pixels (continuous coords, origin = image corner)."""
    cam = cam_obj.data
    res_x, res_y = scene.render.resolution_x, scene.render.resolution_y
    fit = cam.sensor_fit
    if fit == 'AUTO':
        fit = 'HORIZONTAL' if res_x >= res_y else 'VERTICAL'
    if fit == 'HORIZONTAL':
        f_px = cam.lens / cam.sensor_width * res_x
        sensor_long = cam.sensor_width
    else:
        sensor = cam.sensor_width if cam.sensor_fit == 'AUTO' else cam.sensor_height
        f_px = cam.lens / sensor * res_y
        sensor_long = sensor
    max_res = max(res_x, res_y)
    cx = res_x / 2 + cam.shift_x * max_res
    cy = res_y / 2 - cam.shift_y * max_res
    return {
        "focal_mm": cam.lens,
        "sensor_width_mm": cam.sensor_width,
        "sensor_height_mm": cam.sensor_height,
        "sensor_fit": cam.sensor_fit,
        "sensor_on_long_side_mm": sensor_long,
        "resolution": [res_x, res_y],
        "fx": f_px, "fy": f_px, "cx": cx, "cy": cy,
        "K": [[f_px, 0.0, cx], [0.0, f_px, cy], [0.0, 0.0, 1.0]],
        "hfov_deg": math.degrees(2 * math.atan(res_x / 2 / f_px)),
        "vfov_deg": math.degrees(2 * math.atan(res_y / 2 / f_px)),
    }


def camera_extrinsics_cv(cam_obj):
    """OpenCV-convention world->camera R (3x3) and t: x_cam = R @ X + t."""
    mw = cam_obj.matrix_world
    R_bcam = mw.to_3x3().normalized().transposed()       # world -> Blender camera
    R_cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1))) @ R_bcam
    C = mw.translation
    t = -(R_cv @ C)
    return {"R": [list(r) for r in R_cv], "t": list(t), "C": list(C)}


def project_points(scene, cam_obj, points):
    """Project world points. Returns (pixels [[x, y]], in_frame [0/1], depth)."""
    res_x, res_y = scene.render.resolution_x, scene.render.resolution_y
    px, vis, depth = [], [], []
    for p in points:
        co = world_to_camera_view(scene, cam_obj, Vector(p))
        x, y = co.x * res_x, (1.0 - co.y) * res_y  # image y points down
        inside = co.z > 0 and 0.0 <= x <= res_x and 0.0 <= y <= res_y
        px.append([x, y])
        vis.append(1 if inside else 0)
        depth.append(co.z)
    return px, vis, depth


def keypoint_occlusion(scene, depsgraph, origin, points):
    """Ray-cast camera -> keypoint. True = an occluding object is in between.

    Objects tagged with OCCLUDER_PROP = False (floor, painted lines, decals,
    nets) are passed through.
    """
    occluded = []
    origin = Vector(origin)
    for p in points:
        target = Vector(p)
        vec = target - origin
        dist = vec.length
        direction = vec.normalized()
        start = origin.copy()
        remaining = dist - OCCLUSION_EPS
        hit_occluder = False
        for _ in range(32):
            if remaining <= 0:
                break
            hit, loc, _n, _i, obj, _m = scene.ray_cast(depsgraph, start, direction, distance=remaining)
            if not hit:
                break
            src = obj.original if obj is not None and hasattr(obj, "original") else obj
            if src is not None and src.get(OCCLUDER_PROP, True):
                hit_occluder = True
                break
            step = (loc - start).length + 1e-4
            start = loc + direction * 1e-4
            remaining -= step
        occluded.append(hit_occluder)
    return occluded


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------

def _create_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_width = SENSOR_WIDTH
    cam_data.sensor_height = 24.0
    cam_data.sensor_fit = 'AUTO'
    cam_data.clip_start = 0.05
    cam_data.clip_end = 300.0
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def place_camera(keypoints, strategy="broadcast", config=None, scene_info=None):
    """Place the camera, resampling until enough keypoints are usable.

    Args:
        keypoints: 30 world (x, y, z) positions in template order
        strategy: one of STRATEGIES
        config: dict with "resolution" (width, height), "min_usable", "max_tries"
        scene_info: dict with "bounds" (venue_bounds), "court_offsets",
            "bleachers" (bleacher placements)

    Returns:
        (camera_obj, metadata_dict)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    info = scene_info or {}
    bounds = info.get("bounds") or {"x0": -30, "y0": -30, "x1": 45, "y1": 40, "h": 20}
    info = dict(info, bounds=bounds)
    if strategy not in STRATEGY_WEIGHTS:
        raise ValueError(f"Unknown camera strategy: {strategy}")

    scene = bpy.context.scene
    res = tuple(cfg["resolution"])
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    cam_obj = _create_camera()
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    corners = [keypoints[k] for k in CORNER_INDICES]
    used_strategy = strategy
    best = None
    for attempt in range(cfg["max_tries"]):
        if attempt == cfg["max_tries"] // 2 and best is None:
            used_strategy = "broadcast"  # fallback: reliable full view
        pos = _clamp_to_venue(_sample_position(used_strategy, info), bounds)
        framing = "partial" if random.random() < PARTIAL_FRAMING_PROB else "full"
        if framing == "full":
            target = Vector((COURT_LENGTH / 2 + random.uniform(-2.0, 2.0),
                             COURT_WIDTH / 2 + random.uniform(-1.5, 1.5), random.uniform(0.0, 0.8)))
        else:
            kx, ky = random.choice(KEYPOINTS_TEMPLATE)
            target = Vector((kx + random.uniform(-1.5, 1.5), ky + random.uniform(-1.5, 1.5), 0.0))
        if (target - pos).length < 0.5:
            continue
        roll = _sample_roll_deg()
        quat = _look_rotation(pos, target, roll)
        f_fit = _fit_focal(pos, quat, corners, res)
        focal = _sample_focal()
        if framing == "full" and f_fit and focal > f_fit and random.random() < 0.6:
            # Photographer zooms out until the whole court fits.
            focal = f_fit * random.uniform(0.75, 1.0)
        elif framing == "partial" and f_fit and focal < f_fit * 1.2:
            focal = log_uniform(random, min(f_fit * 1.2, FOCAL_RANGE[1]), FOCAL_RANGE[1])
        focal = clamp(focal, *FOCAL_RANGE)

        cam_obj.data.lens = focal
        cam_obj.location = pos
        cam_obj.rotation_mode = 'QUATERNION'
        cam_obj.rotation_quaternion = quat
        bpy.context.view_layer.update()

        kp_2d, in_frame, depth = project_points(scene, cam_obj, keypoints)
        if sum(in_frame) < cfg["min_usable"]:
            continue
        occluded = keypoint_occlusion(scene, depsgraph, pos, keypoints)
        usable = [1 if v and not o else 0 for v, o in zip(in_frame, occluded)]
        n_usable = sum(usable)
        if (n_usable < cfg["min_usable"] or n_usable < MIN_UNOCCLUDED_FRACTION * sum(in_frame)
                or not has_homography_support(usable)):
            continue
        best = (n_usable, attempt, framing, roll, target, focal, kp_2d, in_frame, occluded, f_fit)
        break

    if best is None:
        raise RuntimeError(f"Could not place a '{strategy}' camera with "
                           f">= {cfg['min_usable']} usable keypoints")

    n_usable, attempt, framing, roll, target, focal, kp_2d, in_frame, occluded, f_fit = best
    euler = cam_obj.rotation_quaternion.to_euler()
    metadata = {
        "strategy": strategy,
        "strategy_used": used_strategy,
        "attempts": attempt + 1,
        "framing": framing,
        "position": list(cam_obj.location),
        "rotation": list(euler),
        "rotation_quaternion": list(cam_obj.rotation_quaternion),
        "matrix_world": [list(r) for r in cam_obj.matrix_world],
        "roll_deg": roll,
        "look_at": list(target),
        "focal_length": focal,
        "fit_focal_length": f_fit,
        "intrinsics": camera_intrinsics(scene, cam_obj),
        "extrinsics_cv": camera_extrinsics_cv(cam_obj),
        "keypoints_3d": [list(p) for p in keypoints],
        "keypoints_2d": kp_2d,
        "visibility": in_frame,          # 1 = projects inside the image (in front of camera)
        "in_frame": in_frame,
        "keypoint_occluded": occluded,   # True = non-floor, non-net object in the way
        "num_usable": n_usable,
        "pixel_convention": "continuous pixels, origin at top-left image corner; "
                            "pixel i spans [i, i+1], centre i+0.5",
    }
    return cam_obj, metadata

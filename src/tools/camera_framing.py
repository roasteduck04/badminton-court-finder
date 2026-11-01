"""Per-image automatic field of view ("auto-FOV") for the Blender camera.

Pure numpy, no ``bpy``: importable from the project Python for tests and from
Blender's bundled Python (which ships numpy).

Pipeline for one camera (see ``blender_camera.place_camera``):
    1. choose what the photographer frames: the whole court, or a region of it
       (``sample_framing_target``);
    2. aim the camera at the angular centre of that region and compute the
       focal length that fits it with a margin (``frame_points``);
    3. turn that *required* focal length into the one a real device would
       use (``choose_focal``): phones have fixed native lenses plus limited
       digital zoom, broadcast cameras have a continuous zoom range;
    4. measure how much of the image the court actually covers
       (``court_coverage``) and reject unrealistic views.

Conventions:
    world: x along the court length, y across the width, z up (metres).
    camera (Blender): looks along -Z, +Y is image up, +X is image right.
    Focal lengths are 35 mm-equivalent: the sensor diagonal is always that of
    a 36x24 mm frame (43.27 mm) whatever the aspect ratio, which is how phone
    and camera makers quote "equivalent" focal lengths.

The defaults were tuned so the rendered views match the court-framing
statistics of the real images in data/colab_data_v2 (court covers roughly
15-85% of the image; real phone photos are often partial views).
"""

import math

import numpy as np

EQUIV_DIAGONAL_MM = math.hypot(36.0, 24.0)

# Native lenses of a typical phone (35 mm-equivalent, mm).
PHONE_ULTRAWIDE = (13.0, 16.0)
PHONE_MAIN = (23.0, 28.0)
PHONE_TELE = (48.0, 77.0)
PHONE_TELE_PROB = 0.6           # fraction of phones with a tele module
PHONE_NO_ZOOM_PROB = 0.25       # photographer does not digitally zoom at all
PHONE_DIGITAL_ZOOM_MAX = (1.3, 3.0)

# Broadcast / fixed installation zoom lens (35 mm-equivalent, mm); a 2/3" 24x
# ENG zoom such as the Fujinon UA24x7.8 spans roughly 25-600 mm-equivalent,
# but master shots of a badminton court sit at about 25-200 mm.
ZOOM_RANGE = (22.0, 200.0)
NET_POST_HEIGHT = 1.55

LENS_MODELS = ("phone", "zoom")


def equivalent_sensor_long_side(width, height):
    """Sensor long side (mm) giving a 43.27 mm diagonal at this aspect ratio."""
    long_px, short_px = max(width, height), min(width, height)
    return EQUIV_DIAGONAL_MM * long_px / math.hypot(long_px, short_px)


def half_extents(width, height, sensor_long_mm):
    """Half sensor size (mm) along image x and y."""
    long_px = max(width, height)
    return sensor_long_mm / 2 * width / long_px, sensor_long_mm / 2 * height / long_px


def look_rotation(direction, roll_deg=0.0):
    """Camera-to-world rotation (3x3, columns = camera x, y, z axes in world).

    The camera looks along ``direction`` (its -Z axis), keeps +Y as close to
    world up as possible, then rolls by ``roll_deg`` about the view axis.
    """
    f = np.asarray(direction, float)
    f = f / np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-6:            # looking straight up/down
        x = np.cross(f, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    z = -f
    y = np.cross(z, x)
    a = math.radians(roll_deg)
    xr = math.cos(a) * x + math.sin(a) * y
    yr = -math.sin(a) * x + math.cos(a) * y
    return np.stack([xr, yr, z], axis=1)


def to_camera_plane(points, cam_pos, R, min_depth=0.1):
    """Project world points to the normalized image plane (tan units, +v up).

    Returns (uv (N, 2), depth (N,)); uv is NaN for points closer than
    ``min_depth`` in front of the camera.
    """
    P = np.asarray(points, float)
    if P.shape[1] == 2:
        P = np.concatenate([P, np.zeros((len(P), 1))], axis=1)
    pc = (P - np.asarray(cam_pos, float)) @ R      # world -> camera: R^T p
    depth = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pc[:, :2] / depth[:, None]
    uv[depth < min_depth] = np.nan
    return uv, depth


def frame_points(cam_pos, points, roll_deg, width, height, sensor_long_mm,
                 margin=0.05, iters=30, tol=1e-9):
    """Aim at the angular centre of ``points`` and fit them in frame.

    Args:
        margin: fraction of the half frame left empty on the tightest side
            (negative = the region over-fills the frame, i.e. a closer crop).

    Returns:
        (R, focal_mm) or None if some point is behind / too close to the camera.
    """
    P = np.asarray(points, float)
    if P.shape[1] == 2:
        P = np.concatenate([P, np.zeros((len(P), 1))], axis=1)
    direction = P.mean(axis=0) - np.asarray(cam_pos, float)
    if np.linalg.norm(direction) < 1e-6:
        return None
    R = look_rotation(direction, roll_deg)
    for _ in range(iters):
        uv, _ = to_camera_plane(P, cam_pos, R)
        if np.isnan(uv).any():
            return None
        cu = (uv[:, 0].max() + uv[:, 0].min()) / 2
        cv = (uv[:, 1].max() + uv[:, 1].min()) / 2
        if abs(cu) < tol and abs(cv) < tol:
            break
        R = look_rotation(R @ np.array([cu, cv, -1.0]), roll_deg)
    uv, _ = to_camera_plane(P, cam_pos, R)
    if np.isnan(uv).any():
        return None
    hx, hy = half_extents(width, height, sensor_long_mm)
    ext_u = max(np.abs(uv[:, 0]).max(), 1e-9)
    ext_v = max(np.abs(uv[:, 1]).max(), 1e-9)
    focal = min(hx / ext_u, hy / ext_v) * (1.0 - margin)
    return R, float(focal)


def jitter_aim(R, roll_deg, focal_mm, points, cam_pos, width, height, sensor_long_mm,
               rng, amount=0.8):
    """Re-aim off-centre within the slack left by the margin (composition).

    Points that fit in frame before still fit after (checked; the original
    rotation is returned otherwise).
    """
    uv, _ = to_camera_plane(points, cam_pos, R)
    if np.isnan(uv).any():
        return R
    hx, hy = half_extents(width, height, sensor_long_mm)
    tu, tv = hx / focal_mm, hy / focal_mm
    su = max(0.0, tu - np.abs(uv[:, 0]).max())
    sv = max(0.0, tv - np.abs(uv[:, 1]).max())
    du, dv = rng.uniform(-su, su) * amount, rng.uniform(-sv, sv) * amount
    R2 = look_rotation(R @ np.array([-du, -dv, -1.0]), roll_deg)
    uv2, _ = to_camera_plane(points, cam_pos, R2)
    if np.isnan(uv2).any() or (np.abs(uv2[:, 0]) > tu).any() or (np.abs(uv2[:, 1]) > tv).any():
        return R
    return R2


def choose_focal(required_mm, lens_model, rng):
    """Focal length a real device would use when it needs ``required_mm``.

    phone: the longest native lens that still fits the target, then optional
        digital zoom (capped) towards ``required_mm``. If even the ultra-wide
        cannot fit the target, the ultra-wide is used and the view is cropped.
    zoom: continuous zoom, slightly looser than required, within ZOOM_RANGE.

    Returns (focal_mm, info dict).
    """
    if lens_model == "zoom":
        f = float(np.clip(required_mm * rng.uniform(0.8, 1.0), *ZOOM_RANGE))
        return f, {"lens_model": "zoom", "required_mm": float(required_mm)}
    if lens_model != "phone":
        raise ValueError(f"Unknown lens model {lens_model!r}; choose from {LENS_MODELS}")
    natives = [("ultrawide", rng.uniform(*PHONE_ULTRAWIDE)), ("main", rng.uniform(*PHONE_MAIN))]
    if rng.random() < PHONE_TELE_PROB:
        natives.append(("tele", rng.uniform(*PHONE_TELE)))
    fitting = [n for n in natives if n[1] <= required_mm]
    name, native = fitting[-1] if fitting else natives[0]
    zoom = 1.0
    if fitting and rng.random() >= PHONE_NO_ZOOM_PROB:
        zoom = min(required_mm / native, rng.uniform(*PHONE_DIGITAL_ZOOM_MAX))
    f = float(native * zoom)
    return f, {"lens_model": "phone", "lens": name, "native_mm": float(native),
               "digital_zoom": float(zoom), "required_mm": float(required_mm)}


def sample_framing_target(rng, court_length, court_width, full_prob=0.5):
    """What the photographer frames: the whole court or a floor rectangle.

    Returns (mode, points (N, 3) to keep in frame, margin).
    """
    if rng.random() < full_prob:
        pts = [(0, 0, 0), (court_length, 0, 0), (court_length, court_width, 0),
               (0, court_width, 0)]
        # Keep the net posts in frame too, as a photographer would.
        pts += [(court_length / 2, 0, NET_POST_HEIGHT),
                (court_length / 2, court_width, NET_POST_HEIGHT)]
        margin = rng.uniform(0.0, 0.25)
        return "full", np.array(pts, float), margin
    lx = rng.uniform(0.25, 0.7) * court_length
    ly = rng.uniform(0.4, 1.0) * court_width
    x0 = rng.uniform(-0.1 * court_length, court_length * 1.1 - lx)
    y0 = rng.uniform(-0.1 * court_width, court_width * 1.1 - ly)
    pts = [(x0, y0), (x0 + lx, y0), (x0 + lx, y0 + ly), (x0, y0 + ly)]
    margin = rng.uniform(-0.15, 0.15)
    return "partial", np.array([(x, y, 0.0) for x, y in pts]), margin


def court_coverage(cam_pos, R, focal_mm, width, height, sensor_long_mm,
                   court_length, court_width, grid=(48, 36)):
    """(fraction of the image showing the court surface,
        fraction of the court surface inside the image)."""
    hx, hy = half_extents(width, height, sensor_long_mm)
    gu = (np.arange(grid[0]) + 0.5) / grid[0] * 2 - 1
    gv = (np.arange(grid[1]) + 0.5) / grid[1] * 2 - 1
    U, V = np.meshgrid(gu * hx / focal_mm, gv * hy / focal_mm)
    rays = np.stack([U.ravel(), V.ravel(), -np.ones(U.size)], axis=1) @ R.T
    C = np.asarray(cam_pos, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -C[2] / rays[:, 2]
    X = C[None, :] + rays * s[:, None]
    hit = (s > 0) & (X[:, 0] >= 0) & (X[:, 0] <= court_length) & (X[:, 1] >= 0) & (X[:, 1] <= court_width)
    xs, ys = np.meshgrid(np.linspace(0, court_length, 41), np.linspace(0, court_width, 19))
    floor = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    uv, _ = to_camera_plane(floor, C, R, min_depth=1e-3)
    inside = (~np.isnan(uv[:, 0]) & (np.abs(uv[:, 0]) <= hx / focal_mm)
              & (np.abs(uv[:, 1]) <= hy / focal_mm))
    return float(hit.mean()), float(inside.mean())


def elevation_deg(cam_pos, court_length, court_width):
    """Angle (degrees) between the floor and the ray to the court centre."""
    C = np.asarray(cam_pos, float)
    horiz = math.hypot(C[0] - court_length / 2, C[1] - court_width / 2)
    return math.degrees(math.atan2(C[2], horiz))

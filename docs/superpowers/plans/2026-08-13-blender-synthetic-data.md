# Blender Synthetic Data Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular Blender Python pipeline that generates synthetic badminton court images with ground-truth keypoint annotations in CVN format.

**Architecture:** Six Blender Python modules (court, camera, lighting, occluders, environment, renderer) that run inside Blender's Python environment (`bpy`), plus one standalone converter script. The modules build scene elements procedurally, randomize parameters, render images, and export metadata. The converter transforms raw metadata into the CVN annotation format used by `CourtDataset`.

**Tech Stack:** Blender Python API (`bpy`), Python stdlib, numpy (converter only)

## Global Constraints

- All Blender modules import `bpy` and run inside Blender's bundled Python — they cannot be executed with the project's regular Python interpreter
- The converter (`blender_to_cvn.py`) runs with the project's Python and uses only stdlib + numpy
- Court dimensions must match `src/court_geometry.py` constants exactly (13.4m × 6.1m, 30 keypoints in 6×5 grid)
- Output CVN annotation format must match what `coco_to_cvn.py` produces: `{"image_path", "image_size", "keypoints", "visibility", "bounding_box"}`
- Render resolution: 640×640 pixels
- All randomization uses Python's `random` module with optional seed for reproducibility
- The existing `data/blender/court_template.blend` file contains the base court geometry (surface, lines, keypoint empties, net, camera, lights) organized into collections: Court, Keypoints, NetAssembly, Lighting

---

### Task 1: Court Geometry Module

**Files:**
- Create: `src/tools/blender_court.py`

**Interfaces:**
- Consumes: Court dimension constants from `src/court_geometry.py` (imported as raw values, not as a module — Blender's Python doesn't have the project on `sys.path` by default, so the constants are duplicated or the path is added at the top of the script)
- Produces: `build_court(config: dict) -> dict` returning `{"surface": bpy.types.Object, "lines": list[bpy.types.Object], "net": bpy.types.Object, "posts": list[bpy.types.Object], "keypoints": list[bpy.types.Object]}`. Config keys: `surface_color` (str), `line_color` (str), `include_net` (bool).

- [ ] **Step 1: Create the module with court constants and config**

```python
"""Blender court geometry builder.

Builds a 3D badminton court with randomizable surface/line colors,
optional net, and 30 keypoint empties at line intersections.
"""

import bpy
import random

# BWF court dimensions (meters) — duplicated from src/court_geometry.py
# to avoid sys.path manipulation inside Blender's Python
COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
SINGLES_OFFSET = 0.46
NET_POS = 6.7
SHORT_SERVICE = 1.98
LONG_SERVICE_DBL = 0.76
LINE_WIDTH = 0.04
LINE_Z = 0.001
NUM_KEYPOINTS = 30

ROWS_X = [0.0, LONG_SERVICE_DBL, NET_POS - SHORT_SERVICE,
           NET_POS + SHORT_SERVICE, COURT_LENGTH - LONG_SERVICE_DBL, COURT_LENGTH]
COLS_Y = [0.0, SINGLES_OFFSET, COURT_WIDTH / 2,
           COURT_WIDTH - SINGLES_OFFSET, COURT_WIDTH]

SURFACE_COLORS = {
    "green": (0.15, 0.45, 0.18, 1.0),
    "blue": (0.12, 0.25, 0.55, 1.0),
    "red": (0.55, 0.15, 0.12, 1.0),
    "wood": (0.55, 0.35, 0.18, 1.0),
    "grey": (0.35, 0.38, 0.40, 1.0),
}

LINE_COLORS = {
    "white": (0.95, 0.95, 0.95, 1.0),
    "yellow": (0.95, 0.90, 0.50, 1.0),
    "light_grey": (0.80, 0.80, 0.80, 1.0),
}

DEFAULT_CONFIG = {
    "surface_color": "green",
    "line_color": "white",
    "include_net": True,
}
```

- [ ] **Step 2: Implement `_clear_collection` and `_make_material` helpers**

```python
def _clear_collection(name):
    """Remove all objects in a collection, then the collection itself."""
    if name in bpy.data.collections:
        col = bpy.data.collections[name]
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)


def _make_material(name, color, roughness=0.7, metallic=0.0):
    """Create or replace a Principled BSDF material."""
    if name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[name])
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat
```

- [ ] **Step 3: Implement `_build_surface`**

```python
def _build_surface(collection, config):
    """Create the court floor plane with a randomizable material."""
    color_name = config.get("surface_color", "green")
    if color_name == "random":
        color_name = random.choice(list(SURFACE_COLORS.keys()))
    color = SURFACE_COLORS[color_name]

    bpy.ops.mesh.primitive_plane_add(size=1, location=(COURT_LENGTH / 2, COURT_WIDTH / 2, 0))
    surface = bpy.context.active_object
    surface.name = "CourtSurface"
    surface.scale = (COURT_LENGTH, COURT_WIDTH, 1)
    bpy.ops.object.transform_apply(scale=True)

    mat = _make_material("CourtSurfaceMat", color, roughness=0.7)
    surface.data.materials.append(mat)

    # Move to collection
    bpy.context.scene.collection.objects.unlink(surface)
    collection.objects.link(surface)

    return surface, color_name
```

- [ ] **Step 4: Implement `_build_lines`**

```python
def _build_lines(collection, config):
    """Create court line meshes (thin planes slightly above the surface)."""
    color_name = config.get("line_color", "white")
    if color_name == "random":
        color_name = random.choice(list(LINE_COLORS.keys()))
    color = LINE_COLORS[color_name]
    mat = _make_material("CourtLineMat", color, roughness=0.5)

    # Line definitions: (name, x1, y1, x2, y2, orientation)
    # orientation: "h" = along Y (cross-court), "v" = along X (sideline)
    ss = NET_POS - SHORT_SERVICE
    rs = NET_POS + SHORT_SERVICE
    rls = COURT_LENGTH - LONG_SERVICE_DBL
    cy = COURT_WIDTH / 2
    sb = COURT_WIDTH - SINGLES_OFFSET

    line_defs = [
        # Cross-court lines (horizontal, along Y)
        ("Line_BaselineL", 0, 0, 0, COURT_WIDTH, "h"),
        ("Line_BaselineR", COURT_LENGTH, 0, COURT_LENGTH, COURT_WIDTH, "h"),
        ("Line_LongSvcL", LONG_SERVICE_DBL, 0, LONG_SERVICE_DBL, COURT_WIDTH, "h"),
        ("Line_LongSvcR", rls, 0, rls, COURT_WIDTH, "h"),
        ("Line_ShortSvcL", ss, 0, ss, COURT_WIDTH, "h"),
        ("Line_ShortSvcR", rs, 0, rs, COURT_WIDTH, "h"),
        # Sidelines (vertical, along X)
        ("Line_DblTop", 0, 0, COURT_LENGTH, 0, "v"),
        ("Line_DblBot", 0, COURT_WIDTH, COURT_LENGTH, COURT_WIDTH, "v"),
        ("Line_SglTop", 0, SINGLES_OFFSET, COURT_LENGTH, SINGLES_OFFSET, "v"),
        ("Line_SglBot", 0, sb, COURT_LENGTH, sb, "v"),
        # Center lines
        ("Line_CenterL", 0, cy, ss, cy, "v"),
        ("Line_CenterR", rs, cy, COURT_LENGTH, cy, "v"),
    ]

    lines = []
    for name, x1, y1, x2, y2, orient in line_defs:
        dx = x2 - x1
        dy = y2 - y1
        length = (dx ** 2 + dy ** 2) ** 0.5

        bpy.ops.mesh.primitive_plane_add(
            size=1,
            location=((x1 + x2) / 2, (y1 + y2) / 2, LINE_Z),
        )
        obj = bpy.context.active_object
        if orient == "h":
            obj.scale = (LINE_WIDTH, length, 1)
        else:
            obj.scale = (length, LINE_WIDTH, 1)
        bpy.ops.object.transform_apply(scale=True)
        obj.name = name
        obj.data.materials.append(mat)

        bpy.context.scene.collection.objects.unlink(obj)
        collection.objects.link(obj)
        lines.append(obj)

    return lines, color_name
```

- [ ] **Step 5: Implement `_build_keypoints`**

```python
def _build_keypoints(collection):
    """Place 30 empties at court line intersections."""
    row_names = ["BaselineL", "LongSvcL", "ShortSvcL",
                 "ShortSvcR", "LongSvcR", "BaselineR"]
    col_names = ["DblTop", "SglTop", "Center", "SglBot", "DblBot"]

    keypoints = []
    for row_i, rx in enumerate(ROWS_X):
        for col_i, cy in enumerate(COLS_Y):
            k_idx = row_i * 5 + col_i
            name = f"K{k_idx:02d}_{row_names[row_i]}_{col_names[col_i]}"

            bpy.ops.object.empty_add(
                type='PLAIN_AXES', location=(rx, cy, LINE_Z), radius=0.15,
            )
            empty = bpy.context.active_object
            empty.name = name

            bpy.context.scene.collection.objects.unlink(empty)
            collection.objects.link(empty)
            keypoints.append(empty)

    return keypoints
```

- [ ] **Step 6: Implement `_build_net`**

```python
NET_HEIGHT_EDGE = 1.55
NET_HEIGHT_CENTER = 1.524
POST_RADIUS = 0.04
POST_EXTENSION = 0.3


def _build_net(collection):
    """Create net mesh and two posts."""
    mat_net = _make_material("NetMat", (0.1, 0.1, 0.1, 1.0), roughness=0.9)
    mat_net.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.5

    mat_post = _make_material("PostMat", (0.6, 0.6, 0.65, 1.0),
                               roughness=0.3, metallic=0.8)

    posts = []
    for i, y_pos in enumerate([-POST_EXTENSION, COURT_WIDTH + POST_EXTENSION]):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=POST_RADIUS,
            depth=NET_HEIGHT_EDGE,
            location=(NET_POS, y_pos, NET_HEIGHT_EDGE / 2),
        )
        post = bpy.context.active_object
        post.name = f"NetPost_{i}"
        post.data.materials.append(mat_post)
        bpy.context.scene.collection.objects.unlink(post)
        collection.objects.link(post)
        posts.append(post)

    bpy.ops.mesh.primitive_plane_add(
        size=1,
        location=(NET_POS, COURT_WIDTH / 2, NET_HEIGHT_CENTER / 2 + 0.2),
    )
    net = bpy.context.active_object
    net.name = "Net"
    net.scale = (0.02, COURT_WIDTH + 2 * POST_EXTENSION, NET_HEIGHT_CENTER * 0.65)
    bpy.ops.object.transform_apply(scale=True)
    net.data.materials.append(mat_net)
    bpy.context.scene.collection.objects.unlink(net)
    collection.objects.link(net)

    return net, posts
```

- [ ] **Step 7: Implement the main `build_court` function**

```python
def build_court(config=None):
    """Build a complete badminton court scene.

    Args:
        config: dict with optional keys: surface_color, line_color, include_net.
            Colors can be specific names or "random".

    Returns:
        dict with keys: surface, lines, net, posts, keypoints,
        and metadata: surface_color, line_color.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Clear existing court collections
    for col_name in ["Court", "Keypoints", "NetAssembly"]:
        _clear_collection(col_name)

    court_col = bpy.data.collections.new("Court")
    bpy.context.scene.collection.children.link(court_col)

    kp_col = bpy.data.collections.new("Keypoints")
    bpy.context.scene.collection.children.link(kp_col)

    surface, surface_color = _build_surface(court_col, cfg)
    lines, line_color = _build_lines(court_col, cfg)
    keypoints = _build_keypoints(kp_col)

    net = None
    posts = []
    if cfg["include_net"]:
        net_col = bpy.data.collections.new("NetAssembly")
        bpy.context.scene.collection.children.link(net_col)
        net, posts = _build_net(net_col)

    return {
        "surface": surface,
        "lines": lines,
        "net": net,
        "posts": posts,
        "keypoints": keypoints,
        "surface_color": surface_color,
        "line_color": line_color,
    }
```

- [ ] **Step 8: Test by running in Blender**

Open the template file and run via Blender's script editor or console:
```python
import sys
sys.path.insert(0, r"C:\1NGWZ\1NGWZ\1-NTU\Projects\badminton-court-finder")
from src.tools.blender_court import build_court
result = build_court({"surface_color": "blue", "line_color": "yellow", "include_net": True})
print(f"Surface: {result['surface'].name}, Keypoints: {len(result['keypoints'])}")
```
Expected: Court rebuilds with blue surface, yellow lines, 30 keypoints, net visible.

- [ ] **Step 9: Commit**

```bash
git add src/tools/blender_court.py
git commit -m "feat(blender): add court geometry module with randomizable surface/line colors"
```

---

### Task 2: Camera Module

**Files:**
- Create: `src/tools/blender_camera.py`

**Interfaces:**
- Consumes: `build_court()` return dict — specifically `keypoints` (list of 30 empties)
- Produces: `place_camera(keypoints: list, strategy: str, config: dict) -> tuple[bpy.types.Object, dict]`. The metadata dict contains: `strategy` (str), `position` (list[float]), `rotation` (list[float]), `focal_length` (float), `keypoints_2d` (list[list[float]]), `keypoints_3d` (list[list[float]]), `visibility` (list[int]).

- [ ] **Step 1: Create module with strategy definitions**

```python
"""Blender camera placement with named strategies.

Each strategy defines a region of valid camera positions for
different viewing angles of the badminton court.
"""

import bpy
import random
import math
from mathutils import Vector

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
COURT_CENTER = Vector((COURT_LENGTH / 2, COURT_WIDTH / 2, 0))

STRATEGIES = {
    "broadcast": {
        "x_range": (-6.0, -3.0),
        "y_range": (COURT_WIDTH / 2 - 1.5, COURT_WIDTH / 2 + 1.5),
        "z_range": (6.0, 10.0),
        "elevation_deg": (30, 45),
    },
    "sideline": {
        "x_range": (2.0, 11.0),
        "y_range": (-5.0, -2.0),
        "z_range": (2.0, 5.0),
        "elevation_deg": (15, 30),
    },
    "corner": {
        "x_range": (-4.0, -1.0),
        "y_range": (-4.0, -1.0),
        "z_range": (3.0, 7.0),
        "elevation_deg": (20, 40),
    },
    "overhead": {
        "x_range": (4.0, 9.0),
        "y_range": (1.0, 5.0),
        "z_range": (10.0, 16.0),
        "elevation_deg": (60, 85),
    },
    "low": {
        "x_range": (-3.0, -1.0),
        "y_range": (0.0, COURT_WIDTH),
        "z_range": (0.5, 1.5),
        "elevation_deg": (0, 10),
    },
}

FOCAL_LENGTH_RANGE = (28.0, 85.0)
LOOK_AT_JITTER = 1.5

DEFAULT_CONFIG = {
    "strategy": "broadcast",
    "resolution": (640, 640),
}
```

- [ ] **Step 2: Implement `_project_keypoints`**

```python
def _project_keypoints(scene, camera, keypoints, resolution):
    """Project 3D keypoint empties to 2D pixel coordinates via the camera.

    Returns:
        keypoints_3d: list of [x, y, z] world positions
        keypoints_2d: list of [px, py] pixel positions
        visibility: list of 0/1 (1 if within frame bounds)
    """
    from bpy_extras.object_utils import world_to_camera_view

    res_x, res_y = resolution
    keypoints_3d = []
    keypoints_2d = []
    visibility = []

    for empty in keypoints:
        world_pos = empty.matrix_world.translation
        keypoints_3d.append([world_pos.x, world_pos.y, world_pos.z])

        co = world_to_camera_view(scene, camera, world_pos)
        px = co.x * res_x
        py = (1.0 - co.y) * res_y  # Blender's Y is bottom-up, image Y is top-down

        in_frame = (0 <= co.x <= 1) and (0 <= co.y <= 1) and (co.z > 0)
        keypoints_2d.append([px, py])
        visibility.append(1 if in_frame else 0)

    return keypoints_3d, keypoints_2d, visibility
```

- [ ] **Step 3: Implement `place_camera`**

```python
def place_camera(keypoints, strategy="broadcast", config=None):
    """Place a camera using the named strategy with randomization.

    Args:
        keypoints: list of 30 bpy empty objects from build_court()
        strategy: one of "broadcast", "sideline", "corner", "overhead",
                  "low", "random"
        config: dict with optional "resolution" (tuple of int)

    Returns:
        (camera_obj, metadata_dict)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    resolution = cfg["resolution"]

    if strategy == "random":
        strategy = random.choice(list(STRATEGIES.keys()))

    strat = STRATEGIES[strategy]

    # Sample camera position within strategy bounds
    x = random.uniform(*strat["x_range"])
    y = random.uniform(*strat["y_range"])
    z = random.uniform(*strat["z_range"])

    # Also allow mirroring: sideline can be on either side
    if strategy == "sideline" and random.random() < 0.5:
        y = COURT_WIDTH - y  # flip to other sideline
    if strategy == "corner":
        # Randomly pick one of the 4 corners
        corner_choice = random.randint(0, 3)
        if corner_choice == 1:
            x = COURT_LENGTH - x + COURT_LENGTH  # far baseline
            x = random.uniform(COURT_LENGTH + 1.0, COURT_LENGTH + 4.0)
        elif corner_choice == 2:
            y = COURT_WIDTH - y + COURT_WIDTH
            y = random.uniform(COURT_WIDTH + 1.0, COURT_WIDTH + 4.0)
        elif corner_choice == 3:
            x = random.uniform(COURT_LENGTH + 1.0, COURT_LENGTH + 4.0)
            y = random.uniform(COURT_WIDTH + 1.0, COURT_WIDTH + 4.0)

    # Random focal length
    focal_length = random.uniform(*FOCAL_LENGTH_RANGE)

    # Look-at target: court center with jitter
    look_at = COURT_CENTER.copy()
    look_at.x += random.uniform(-LOOK_AT_JITTER, LOOK_AT_JITTER)
    look_at.y += random.uniform(-LOOK_AT_JITTER, LOOK_AT_JITTER)

    # Clear existing camera
    for obj in list(bpy.data.objects):
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)
    for cam in list(bpy.data.cameras):
        if cam.users == 0:
            bpy.data.cameras.remove(cam)

    # Create camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = focal_length
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_obj.location = (x, y, z)
    direction = look_at - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    # Set render resolution
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]

    # Update scene to ensure matrices are current
    bpy.context.view_layer.update()

    # Project keypoints
    kp_3d, kp_2d, vis = _project_keypoints(
        bpy.context.scene, cam_obj, keypoints, resolution,
    )

    metadata = {
        "strategy": strategy,
        "position": [x, y, z],
        "rotation": list(cam_obj.rotation_euler),
        "focal_length": focal_length,
        "keypoints_3d": kp_3d,
        "keypoints_2d": kp_2d,
        "visibility": vis,
    }

    return cam_obj, metadata
```

- [ ] **Step 4: Test in Blender**

```python
import sys
sys.path.insert(0, r"C:\1NGWZ\1NGWZ\1-NTU\Projects\badminton-court-finder")
from src.tools.blender_court import build_court
from src.tools.blender_camera import place_camera

court = build_court()
cam, meta = place_camera(court["keypoints"], strategy="broadcast")
print(f"Strategy: {meta['strategy']}, Focal: {meta['focal_length']:.1f}mm")
print(f"Visible keypoints: {sum(meta['visibility'])}/30")
```
Expected: Camera placed, 20-30 keypoints visible from broadcast angle.

- [ ] **Step 5: Commit**

```bash
git add src/tools/blender_camera.py
git commit -m "feat(blender): add camera module with 6 placement strategies"
```

---

### Task 3: Lighting Module

**Files:**
- Create: `src/tools/blender_lighting.py`

**Interfaces:**
- Consumes: Nothing from other modules (standalone)
- Produces: `setup_lighting(preset: str, config: dict) -> dict` returning metadata with `preset` (str), `lights` (list of dicts with position/energy/color).

- [ ] **Step 1: Create module with preset definitions and helpers**

```python
"""Blender lighting presets for indoor badminton venues."""

import bpy
import random
import math

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1

PRESETS = ["fluorescent", "mixed", "dim", "harsh", "competition"]

DEFAULT_CONFIG = {
    "preset": "fluorescent",
    "intensity_jitter": 0.2,
    "temp_jitter": 300,
}


def _clear_lights():
    """Remove all existing lights and their collection."""
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    for light in list(bpy.data.lights):
        if light.users == 0:
            bpy.data.lights.remove(light)
    if "Lighting" in bpy.data.collections:
        col = bpy.data.collections["Lighting"]
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)


def _kelvin_to_rgb(temp):
    """Approximate color temperature to RGB (simplified)."""
    t = temp / 100.0
    if t <= 66:
        r = 1.0
        g = max(0, min(1, 0.39 * math.log(t) - 0.63))
        b = max(0, min(1, 0.54 * math.log(t - 10) - 1.19)) if t > 19 else 0
    else:
        r = max(0, min(1, 1.29 * ((t - 60) ** -0.13)))
        g = max(0, min(1, 1.13 * ((t - 60) ** -0.08)))
        b = 1.0
    return (r, g, b)


def _add_area_light(name, location, energy, size, color, collection):
    """Add an area light to the scene."""
    bpy.ops.object.light_add(type='AREA', location=location)
    light = bpy.context.active_object
    light.name = name
    light.data.energy = energy
    light.data.size = size
    light.data.color = color

    if light.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(light)
    collection.objects.link(light)

    return light
```

- [ ] **Step 2: Implement preset builders**

```python
def _build_fluorescent(collection, config):
    """Multiple evenly-spaced rectangular area lights, high-mounted."""
    base_temp = 5200 + random.uniform(-config["temp_jitter"], config["temp_jitter"])
    color = _kelvin_to_rgb(base_temp)
    jitter = config["intensity_jitter"]
    lights = []

    positions = [
        (COURT_LENGTH * 0.25, COURT_WIDTH / 2, 9.0),
        (COURT_LENGTH * 0.5, COURT_WIDTH / 2, 9.0),
        (COURT_LENGTH * 0.75, COURT_WIDTH / 2, 9.0),
        (COURT_LENGTH * 0.25, COURT_WIDTH * 0.15, 8.5),
        (COURT_LENGTH * 0.75, COURT_WIDTH * 0.85, 8.5),
    ]
    for i, pos in enumerate(positions):
        jittered_pos = (
            pos[0] + random.uniform(-0.5, 0.5),
            pos[1] + random.uniform(-0.3, 0.3),
            pos[2] + random.uniform(-0.3, 0.3),
        )
        energy = 500 * (1 + random.uniform(-jitter, jitter))
        light = _add_area_light(f"Light_Fluoro_{i}", jittered_pos, energy, 4.0, color, collection)
        lights.append({"name": light.name, "position": list(jittered_pos), "energy": energy})

    return lights


def _build_mixed(collection, config):
    """Overhead lights plus angled sun for window light."""
    jitter = config["intensity_jitter"]
    lights = []

    # Overhead
    warm = _kelvin_to_rgb(4500 + random.uniform(-200, 200))
    for i in range(3):
        x = COURT_LENGTH * (0.25 + 0.25 * i)
        pos = (x, COURT_WIDTH / 2, 8.0)
        energy = 350 * (1 + random.uniform(-jitter, jitter))
        light = _add_area_light(f"Light_Mixed_Over_{i}", pos, energy, 3.0, warm, collection)
        lights.append({"name": light.name, "position": list(pos), "energy": energy})

    # Sun lamp (window light)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.name = "Light_Mixed_Sun"
    sun.data.energy = 2.0 * (1 + random.uniform(-jitter, jitter))
    sun.data.color = _kelvin_to_rgb(6500)
    sun.rotation_euler = (math.radians(50), math.radians(random.uniform(-30, 30)), 0)
    if sun.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(sun)
    collection.objects.link(sun)
    lights.append({"name": sun.name, "position": [0, 0, 10], "energy": sun.data.energy})

    return lights


def _build_dim(collection, config):
    """Fewer, weaker lights with visible falloff."""
    jitter = config["intensity_jitter"]
    color = _kelvin_to_rgb(4000 + random.uniform(-200, 200))
    lights = []

    positions = [
        (COURT_LENGTH * 0.33, COURT_WIDTH / 2, 7.0),
        (COURT_LENGTH * 0.67, COURT_WIDTH / 2, 7.0),
    ]
    for i, pos in enumerate(positions):
        energy = 200 * (1 + random.uniform(-jitter, jitter))
        light = _add_area_light(f"Light_Dim_{i}", pos, energy, 2.5, color, collection)
        lights.append({"name": light.name, "position": list(pos), "energy": energy})

    return lights


def _build_harsh(collection, config):
    """Single strong directional light creating sharp shadows."""
    jitter = config["intensity_jitter"]
    color = _kelvin_to_rgb(5500)
    lights = []

    side = random.choice(["left", "right"])
    y = -3.0 if side == "left" else COURT_WIDTH + 3.0
    pos = (COURT_LENGTH / 2 + random.uniform(-2, 2), y, 8.0)
    energy = 1200 * (1 + random.uniform(-jitter, jitter))
    light = _add_area_light("Light_Harsh", pos, energy, 1.0, color, collection)
    lights.append({"name": light.name, "position": list(pos), "energy": energy})

    return lights


def _build_competition(collection, config):
    """Bright, even, multi-position lighting with minimal shadows."""
    jitter = config["intensity_jitter"]
    color = _kelvin_to_rgb(5500 + random.uniform(-100, 100))
    lights = []

    positions = [
        (COURT_LENGTH * 0.2, COURT_WIDTH * 0.2, 10.0),
        (COURT_LENGTH * 0.2, COURT_WIDTH * 0.8, 10.0),
        (COURT_LENGTH * 0.5, COURT_WIDTH * 0.2, 10.0),
        (COURT_LENGTH * 0.5, COURT_WIDTH * 0.8, 10.0),
        (COURT_LENGTH * 0.8, COURT_WIDTH * 0.2, 10.0),
        (COURT_LENGTH * 0.8, COURT_WIDTH * 0.8, 10.0),
    ]
    for i, pos in enumerate(positions):
        jittered_pos = (
            pos[0] + random.uniform(-0.3, 0.3),
            pos[1] + random.uniform(-0.3, 0.3),
            pos[2],
        )
        energy = 600 * (1 + random.uniform(-jitter, jitter))
        light = _add_area_light(f"Light_Comp_{i}", jittered_pos, energy, 5.0, color, collection)
        lights.append({"name": light.name, "position": list(jittered_pos), "energy": energy})

    return lights


PRESET_BUILDERS = {
    "fluorescent": _build_fluorescent,
    "mixed": _build_mixed,
    "dim": _build_dim,
    "harsh": _build_harsh,
    "competition": _build_competition,
}
```

- [ ] **Step 3: Implement main `setup_lighting` function**

```python
def setup_lighting(preset="fluorescent", config=None):
    """Set up venue lighting using a named preset.

    Args:
        preset: one of "fluorescent", "mixed", "dim", "harsh",
                "competition", or "random"
        config: dict with optional intensity_jitter, temp_jitter

    Returns:
        dict with "preset" and "lights" metadata
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if preset == "random":
        preset = random.choice(PRESETS)

    _clear_lights()

    col = bpy.data.collections.new("Lighting")
    bpy.context.scene.collection.children.link(col)

    builder = PRESET_BUILDERS[preset]
    lights = builder(col, cfg)

    return {"preset": preset, "lights": lights}
```

- [ ] **Step 4: Test in Blender**

```python
from src.tools.blender_lighting import setup_lighting
meta = setup_lighting("competition")
print(f"Preset: {meta['preset']}, Lights: {len(meta['lights'])}")
```
Expected: 6 lights placed for competition preset.

- [ ] **Step 5: Commit**

```bash
git add src/tools/blender_lighting.py
git commit -m "feat(blender): add lighting module with 5 venue presets"
```

---

### Task 4: Occluders Module

**Files:**
- Create: `src/tools/blender_occluders.py`

**Interfaces:**
- Consumes: Nothing from other modules (standalone, uses court dimension constants)
- Produces: `add_occluders(config: dict) -> dict` returning metadata with `occluders` (list of dicts with type/position).

- [ ] **Step 1: Create module with helpers**

```python
"""On-court occluders: players, umpire chair, equipment."""

import bpy
import random

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
NET_POS = 6.7

DEFAULT_CONFIG = {
    "max_players": 4,
    "umpire_chance": 0.2,
    "equipment_chance": 0.3,
    "no_occluder_chance": 0.3,
}


def _clear_occluders():
    """Remove existing occluder collection."""
    if "Occluders" in bpy.data.collections:
        col = bpy.data.collections["Occluders"]
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)


def _random_court_position():
    """Random position on the court, biased toward service areas."""
    x = random.gauss(COURT_LENGTH / 2, COURT_LENGTH / 4)
    x = max(0.5, min(COURT_LENGTH - 0.5, x))
    y = random.uniform(0.5, COURT_WIDTH - 0.5)
    return (x, y)


def _random_clothing_color():
    """Random dark/colored clothing tone."""
    colors = [
        (0.1, 0.1, 0.15, 1.0),   # dark navy
        (0.15, 0.1, 0.1, 1.0),   # dark red
        (0.1, 0.15, 0.1, 1.0),   # dark green
        (0.2, 0.2, 0.2, 1.0),    # dark grey
        (0.9, 0.9, 0.9, 1.0),    # white jersey
        (0.8, 0.6, 0.1, 1.0),    # yellow jersey
        (0.1, 0.3, 0.7, 1.0),    # blue jersey
    ]
    return random.choice(colors)
```

- [ ] **Step 2: Implement occluder builders**

```python
def _add_player(collection, index):
    """Add a simple capsule-shaped player figure."""
    x, y = _random_court_position()

    # Body (cylinder)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.2, depth=1.2, location=(x, y, 0.6),
    )
    body = bpy.context.active_object
    body.name = f"Player_{index}_Body"

    # Head (sphere)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.12, location=(x, y, 1.35),
    )
    head = bpy.context.active_object
    head.name = f"Player_{index}_Head"

    # Material
    mat = bpy.data.materials.new(f"PlayerMat_{index}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = _random_clothing_color()
    bsdf.inputs["Roughness"].default_value = 0.8

    skin_mat = bpy.data.materials.new(f"SkinMat_{index}")
    skin_mat.use_nodes = True
    bsdf_s = skin_mat.node_tree.nodes["Principled BSDF"]
    bsdf_s.inputs["Base Color"].default_value = (0.6, 0.45, 0.35, 1.0)
    bsdf_s.inputs["Roughness"].default_value = 0.6

    body.data.materials.append(mat)
    head.data.materials.append(skin_mat)

    for obj in [body, head]:
        if obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(obj)
        collection.objects.link(obj)

    return {"type": "player", "position": [x, y, 0], "index": index}


def _add_umpire_chair(collection):
    """Add a simple umpire chair at the net post position."""
    side = random.choice([-1.5, COURT_WIDTH + 1.5])

    # Seat (cube)
    bpy.ops.mesh.primitive_cube_add(
        size=0.6, location=(NET_POS, side, 1.8),
    )
    seat = bpy.context.active_object
    seat.name = "UmpireChair_Seat"
    seat.scale = (0.5, 0.5, 0.1)
    bpy.ops.object.transform_apply(scale=True)

    # Legs (thin cube)
    bpy.ops.mesh.primitive_cube_add(
        size=0.1, location=(NET_POS, side, 0.9),
    )
    legs = bpy.context.active_object
    legs.name = "UmpireChair_Legs"
    legs.scale = (0.4, 0.4, 9.0)
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.new("UmpireChairMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.35, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.6

    for obj in [seat, legs]:
        obj.data.materials.append(mat)
        if obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(obj)
        collection.objects.link(obj)

    return {"type": "umpire_chair", "position": [NET_POS, side, 0]}


def _add_equipment(collection, index):
    """Add a shuttlecock or racket on the court."""
    x, y = _random_court_position()
    item_type = random.choice(["shuttlecock", "racket"])

    if item_type == "shuttlecock":
        bpy.ops.mesh.primitive_cone_add(
            radius1=0.03, depth=0.07, location=(x, y, 0.035),
        )
    else:
        bpy.ops.mesh.primitive_plane_add(
            size=0.25, location=(x, y, 0.01),
        )
        obj = bpy.context.active_object
        obj.scale = (1, 0.4, 1)
        bpy.ops.object.transform_apply(scale=True)

    obj = bpy.context.active_object
    obj.name = f"Equipment_{index}_{item_type}"

    if obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(obj)
    collection.objects.link(obj)

    return {"type": item_type, "position": [x, y, 0]}
```

- [ ] **Step 3: Implement main `add_occluders` function**

```python
def add_occluders(config=None):
    """Add random on-court occluders.

    Args:
        config: dict with optional max_players, umpire_chance,
                equipment_chance, no_occluder_chance

    Returns:
        dict with "occluders" list of metadata dicts
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    _clear_occluders()

    if random.random() < cfg["no_occluder_chance"]:
        return {"occluders": []}

    col = bpy.data.collections.new("Occluders")
    bpy.context.scene.collection.children.link(col)

    occluders = []

    # Players
    num_players = random.randint(0, cfg["max_players"])
    for i in range(num_players):
        meta = _add_player(col, i)
        occluders.append(meta)

    # Umpire chair
    if random.random() < cfg["umpire_chance"]:
        meta = _add_umpire_chair(col)
        occluders.append(meta)

    # Equipment
    if random.random() < cfg["equipment_chance"]:
        num_items = random.randint(1, 3)
        for i in range(num_items):
            meta = _add_equipment(col, i)
            occluders.append(meta)

    return {"occluders": occluders}
```

- [ ] **Step 4: Test in Blender**

```python
from src.tools.blender_occluders import add_occluders
meta = add_occluders({"no_occluder_chance": 0})
print(f"Occluders: {len(meta['occluders'])}")
for occ in meta["occluders"]:
    print(f"  {occ['type']} at {occ['position']}")
```
Expected: 1-4 players + possibly umpire chair + possibly equipment.

- [ ] **Step 5: Commit**

```bash
git add src/tools/blender_occluders.py
git commit -m "feat(blender): add occluders module with players, umpire chair, equipment"
```

---

### Task 5: Environment Module

**Files:**
- Create: `src/tools/blender_environment.py`

**Interfaces:**
- Consumes: Nothing from other modules
- Produces: `build_environment(config: dict) -> dict` returning metadata with `adjacent_courts` (int), `venue_size` (str), `has_dividers` (bool), `has_spectators` (bool).

- [ ] **Step 1: Create module with helpers**

```python
"""Venue environment: surrounding floor, walls, adjacent courts, spectators."""

import bpy
import math
import random

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
COURT_GAP = 2.0

VENUE_MARGIN = 8.0

DEFAULT_CONFIG = {
    "max_adjacent_courts": 8,
    "spectator_chance": 0.4,
    "divider_chance": 0.5,
    "ceiling_height": {"small": 18, "medium": 22, "large": 28},
}

# 3x3 grid positions around the main court (row_offset, col_offset)
GRID_SLOTS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

FLOOR_COLORS = {
    "concrete": (0.45, 0.43, 0.40, 1.0),
    "rubber": (0.15, 0.15, 0.18, 1.0),
    "wood": (0.50, 0.35, 0.20, 1.0),
}


def _clear_environment():
    if "Environment" in bpy.data.collections:
        col = bpy.data.collections["Environment"]
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)
```

- [ ] **Step 2: Implement environment builders**

```python
def _build_venue_floor(collection, venue_w, venue_d):
    """Extended floor plane beyond the court."""
    cx = COURT_LENGTH / 2
    cy = COURT_WIDTH / 2
    color = random.choice(list(FLOOR_COLORS.values()))

    bpy.ops.mesh.primitive_plane_add(size=1, location=(cx, cy, -0.005))
    floor = bpy.context.active_object
    floor.name = "VenueFloor"
    floor.scale = (venue_d, venue_w, 1)
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.new("VenueFloorMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.8
    floor.data.materials.append(mat)

    if floor.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(floor)
    collection.objects.link(floor)

    return floor


def _build_walls_and_ceiling(collection, venue_w, venue_d, venue_h):
    """Walls and ceiling enclosure."""
    cx = COURT_LENGTH / 2
    cy = COURT_WIDTH / 2

    mat = bpy.data.materials.new("WallMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.75, 0.73, 0.70, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9

    walls = []
    # 4 walls
    wall_specs = [
        ("Wall_Back", (cx - venue_d / 2, cy, venue_h / 2), (0.1, venue_w, venue_h)),
        ("Wall_Front", (cx + venue_d / 2, cy, venue_h / 2), (0.1, venue_w, venue_h)),
        ("Wall_Left", (cx, cy - venue_w / 2, venue_h / 2), (venue_d, 0.1, venue_h)),
        ("Wall_Right", (cx, cy + venue_w / 2, venue_h / 2), (venue_d, 0.1, venue_h)),
    ]
    for name, loc, scale in wall_specs:
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        wall = bpy.context.active_object
        wall.name = name
        wall.scale = scale
        bpy.ops.object.transform_apply(scale=True)
        wall.data.materials.append(mat)
        if wall.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(wall)
        collection.objects.link(wall)
        walls.append(wall)

    # Ceiling
    bpy.ops.mesh.primitive_plane_add(size=1, location=(cx, cy, venue_h))
    ceiling = bpy.context.active_object
    ceiling.name = "Ceiling"
    ceiling.scale = (venue_d, venue_w, 1)
    bpy.ops.object.transform_apply(scale=True)
    ceiling.data.materials.append(mat)
    if ceiling.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(ceiling)
    collection.objects.link(ceiling)

    return walls + [ceiling]


def _build_adjacent_court(collection, index, y_offset):
    """A simplified adjacent court (surface + lines only)."""
    from src.tools.blender_court import SURFACE_COLORS, LINE_Z, LINE_WIDTH

    color_name = random.choice(list(SURFACE_COLORS.keys()))
    color = SURFACE_COLORS[color_name]

    bpy.ops.mesh.primitive_plane_add(
        size=1,
        location=(COURT_LENGTH / 2, y_offset + COURT_WIDTH / 2, 0),
    )
    surface = bpy.context.active_object
    surface.name = f"AdjCourt_{index}_Surface"
    surface.scale = (COURT_LENGTH, COURT_WIDTH, 1)
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.new(f"AdjCourtMat_{index}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.7
    surface.data.materials.append(mat)

    if surface.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(surface)
    collection.objects.link(surface)

    # Add basic sidelines for the adjacent court
    line_mat = bpy.data.materials.new(f"AdjLineMat_{index}")
    line_mat.use_nodes = True
    bsdf = line_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1.0)

    for li, y in enumerate([y_offset, y_offset + COURT_WIDTH]):
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            location=(COURT_LENGTH / 2, y, LINE_Z),
        )
        line = bpy.context.active_object
        line.name = f"AdjCourt_{index}_Line_{li}"
        line.scale = (COURT_LENGTH, LINE_WIDTH, 1)
        bpy.ops.object.transform_apply(scale=True)
        line.data.materials.append(line_mat)
        if line.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(line)
        collection.objects.link(line)

    return surface


def _build_spectators(collection, venue_w):
    """Low-poly bench rows behind one or both baselines."""
    mat = bpy.data.materials.new("BenchMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.3, 0.25, 0.2, 1.0)

    benches = []
    sides = random.sample(["back", "front"], k=random.randint(1, 2))
    for side in sides:
        x = -3.0 if side == "back" else COURT_LENGTH + 3.0
        for row in range(random.randint(1, 3)):
            bpy.ops.mesh.primitive_cube_add(
                size=1,
                location=(x - row * 0.8 if side == "back" else x + row * 0.8,
                          COURT_WIDTH / 2, 0.3 + row * 0.3),
            )
            bench = bpy.context.active_object
            bench.name = f"Bench_{side}_{row}"
            bench.scale = (0.3, min(venue_w * 0.6, 8.0), 0.15)
            bpy.ops.object.transform_apply(scale=True)
            bench.data.materials.append(mat)
            if bench.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(bench)
            collection.objects.link(bench)
            benches.append(bench)

    return benches


def _build_dividers(collection, y_positions):
    """Curtain dividers between courts."""
    mat = bpy.data.materials.new("DividerMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.2, 0.25, 0.35, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.95

    dividers = []
    for i, y in enumerate(y_positions):
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            location=(COURT_LENGTH / 2, y, 1.5),
        )
        div = bpy.context.active_object
        div.name = f"Divider_{i}"
        div.scale = (COURT_LENGTH, 0.01, 3.0)
        bpy.ops.object.transform_apply(scale=True)
        div.data.materials.append(mat)
        if div.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(div)
        collection.objects.link(div)
        dividers.append(div)

    return dividers


def _build_scoreboard(collection):
    """Flat rectangle scoreboard on a stand at courtside."""
    side = random.choice([-3.0, COURT_WIDTH + 3.0])
    x = COURT_LENGTH / 2 + random.uniform(-2, 2)

    bpy.ops.mesh.primitive_plane_add(
        size=1, location=(x, side, 2.0),
    )
    board = bpy.context.active_object
    board.name = "Scoreboard"
    board.scale = (1.5, 0.05, 1.0)
    board.rotation_euler = (0, 0, 0 if side < 0 else math.pi)
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.new("ScoreboardMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.1, 0.1, 0.12, 1.0)
    board.data.materials.append(mat)

    if board.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(board)
    collection.objects.link(board)

    return board
```

- [ ] **Step 3: Implement main `build_environment` function**

```python
def build_environment(config=None):
    """Build venue environment around the main court.

    Args:
        config: dict with optional max_adjacent_courts, spectator_chance,
                divider_chance, venue_sizes

    Returns:
        dict with environment metadata
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    _clear_environment()

    col = bpy.data.collections.new("Environment")
    bpy.context.scene.collection.children.link(col)

    # Pick venue size
    venue_size = random.choice(list(cfg["venue_sizes"].keys()))
    venue_d, venue_w, venue_h = cfg["venue_sizes"][venue_size]

    _build_venue_floor(col, venue_w, venue_d)
    _build_walls_and_ceiling(col, venue_w, venue_d, venue_h)

    # Adjacent courts
    num_adj = random.randint(0, cfg["max_adjacent_courts"])
    divider_y_positions = []
    for i in range(num_adj):
        direction = 1 if i % 2 == 0 else -1
        offset_y = COURT_WIDTH + COURT_GAP
        if direction == -1:
            offset_y = -(COURT_WIDTH + COURT_GAP)
        actual_y = offset_y if i < 2 else offset_y * 2
        _build_adjacent_court(col, i, actual_y)
        divider_y = COURT_WIDTH + COURT_GAP / 2 if direction == 1 else -COURT_GAP / 2
        divider_y_positions.append(divider_y)

    # Dividers
    has_dividers = False
    if divider_y_positions and random.random() < cfg["divider_chance"]:
        _build_dividers(col, divider_y_positions)
        has_dividers = True

    # Spectators
    has_spectators = random.random() < cfg["spectator_chance"]
    if has_spectators:
        _build_spectators(col, venue_w)

    # Scoreboard (~30% chance)
    has_scoreboard = random.random() < 0.3
    if has_scoreboard:
        _build_scoreboard(col)

    return {
        "adjacent_courts": num_adj,
        "venue_size": venue_size,
        "has_dividers": has_dividers,
        "has_spectators": has_spectators,
        "has_scoreboard": has_scoreboard,
    }
```

- [ ] **Step 4: Test in Blender**

```python
from src.tools.blender_environment import build_environment
meta = build_environment({"spectator_chance": 1.0, "max_adjacent_courts": 2})
print(f"Venue: {meta['venue_size']}, Adjacent: {meta['adjacent_courts']}, "
      f"Spectators: {meta['has_spectators']}, Dividers: {meta['has_dividers']}")
```
Expected: Venue floor, walls, ceiling, 0-2 adjacent courts, spectator benches visible.

- [ ] **Step 5: Commit**

```bash
git add src/tools/blender_environment.py
git commit -m "feat(blender): add environment module with venue, adjacent courts, spectators"
```

---

### Task 6: Renderer & Orchestrator

**Files:**
- Create: `src/tools/blender_render.py`

**Interfaces:**
- Consumes: `build_court()`, `place_camera()`, `setup_lighting()`, `add_occluders()`, `build_environment()` — all from tasks 1-5
- Produces: Rendered images in `data/blender/raw/images/` and metadata JSONs in `data/blender/raw/metadata/`. CLI entry point for headless batch rendering.

- [ ] **Step 1: Create module with config and argument parsing**

```python
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
SEED = int(_get_arg("--seed", "-1"))
ENGINE = _get_arg("--engine", "CYCLES")
SAMPLES = int(_get_arg("--samples", "64"))

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
```

- [ ] **Step 2: Implement the render loop**

```python
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
    bpy.context.scene.render.resolution_x = 640
    bpy.context.scene.render.resolution_y = 640
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    from src.tools.blender_court import build_court
    from src.tools.blender_camera import place_camera
    from src.tools.blender_lighting import setup_lighting
    from src.tools.blender_occluders import add_occluders
    from src.tools.blender_environment import build_environment

    print(f"\n=== Rendering {COUNT} images ===")
    print(f"  Engine: {ENGINE}, Samples: {SAMPLES}")
    print(f"  Output: {OUTPUT_DIR}\n")

    for i in range(COUNT):
        t0 = time.time()
        idx = f"{i + 1:04d}"

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
        cam, cam_meta = place_camera(court["keypoints"], strategy=strategy)

        # 4. Setup lighting
        preset = random.choice(LIGHTING_PRESETS)
        light_meta = setup_lighting(preset)

        # 5. Add occluders
        occ_meta = add_occluders()

        # 6. Render
        img_name = f"blender_{idx}.png"
        img_path = os.path.join(IMAGES_DIR, img_name)
        bpy.context.scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)

        # 7. Export metadata
        metadata = {
            "image_file": img_name,
            "resolution": [640, 640],
            "camera": cam_meta,
            "lighting": light_meta,
            "occluders": occ_meta["occluders"],
            "court": {
                "surface_color": court["surface_color"],
                "line_color": court["line_color"],
            },
            "environment": env_meta,
        }

        meta_path = os.path.join(METADATA_DIR, f"blender_{idx}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - t0
        vis = sum(cam_meta["visibility"])
        print(f"  [{i + 1}/{COUNT}] {img_name} | {strategy} | {preset} | "
              f"{vis}/30 kp | {elapsed:.1f}s")

    print(f"\nDone: {COUNT} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    render_batch()
```

- [ ] **Step 3: Test with a small batch**

```bash
blender --background data/blender/court_template.blend --python src/tools/blender_render.py -- --count 3 --engine EEVEE --samples 16
```
Expected: 3 images + 3 metadata JSONs in `data/blender/raw/`. Each metadata JSON has camera, lighting, occluder, court, and environment fields.

- [ ] **Step 4: Verify a metadata JSON**

```bash
python -c "import json; m=json.load(open('data/blender/raw/metadata/blender_0001.json')); print(f'Strategy: {m[\"camera\"][\"strategy\"]}'); print(f'Visible: {sum(m[\"camera\"][\"visibility\"])}/30'); print(f'Court: {m[\"court\"]}')"
```
Expected: Valid metadata with strategy name, visibility count, court colors.

- [ ] **Step 5: Commit**

```bash
git add src/tools/blender_render.py
git commit -m "feat(blender): add batch renderer orchestrating all modules"
```

---

### Task 7: Converter (blender_to_cvn.py)

**Files:**
- Create: `src/tools/blender_to_cvn.py`
- Create: `tests/test_blender_to_cvn.py`

**Interfaces:**
- Consumes: Metadata JSONs from `data/blender/raw/metadata/` and images from `data/blender/raw/images/`
- Produces: CVN-format annotations in `data/blender/annotations/` and copied images in `data/blender/images/`. Same format as `coco_to_cvn.py` output.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for blender_to_cvn converter."""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from src.tools.blender_to_cvn import convert_metadata, convert_all


@pytest.fixture
def tmp_dirs():
    """Create temporary directory structure mimicking data/blender/raw/."""
    base = tempfile.mkdtemp()
    raw_img = os.path.join(base, "raw", "images")
    raw_meta = os.path.join(base, "raw", "metadata")
    out_img = os.path.join(base, "images")
    out_ann = os.path.join(base, "annotations")
    os.makedirs(raw_img)
    os.makedirs(raw_meta)
    yield {
        "base": base,
        "raw_img": raw_img,
        "raw_meta": raw_meta,
        "out_img": out_img,
        "out_ann": out_ann,
    }
    shutil.rmtree(base)


def _make_sample_metadata(visible_count=20):
    """Create a sample metadata dict."""
    kp_2d = [[i * 20.0, i * 15.0] for i in range(30)]
    vis = [1] * visible_count + [0] * (30 - visible_count)
    return {
        "image_file": "blender_0001.png",
        "resolution": [640, 640],
        "camera": {
            "strategy": "broadcast",
            "keypoints_2d": kp_2d,
            "keypoints_3d": [[0, 0, 0]] * 30,
            "visibility": vis,
        },
    }


def test_convert_metadata_normalizes_coordinates():
    meta = _make_sample_metadata(visible_count=10)
    result = convert_metadata(meta)

    for i in range(10):
        expected_x = (i * 20.0) / 640.0
        expected_y = (i * 15.0) / 640.0
        assert abs(result["keypoints"][i][0] - expected_x) < 1e-6
        assert abs(result["keypoints"][i][1] - expected_y) < 1e-6

    for i in range(10, 30):
        assert result["keypoints"][i] == [-1.0, -1.0]
        assert result["visibility"][i] == 0


def test_convert_metadata_computes_bounding_box():
    meta = _make_sample_metadata(visible_count=5)
    result = convert_metadata(meta)
    bb = result["bounding_box"]
    assert len(bb) == 4
    assert bb[0] >= 0
    assert bb[1] >= 0
    assert bb[2] > 0
    assert bb[3] > 0


def test_convert_metadata_returns_none_below_min_visible():
    meta = _make_sample_metadata(visible_count=2)
    result = convert_metadata(meta, min_visible=4)
    assert result is None


def test_convert_metadata_passes_at_min_visible():
    meta = _make_sample_metadata(visible_count=4)
    result = convert_metadata(meta, min_visible=4)
    assert result is not None
    assert result["image_path"] == "blender_0001.png"


def test_convert_all_end_to_end(tmp_dirs):
    # Write a fake image
    img_path = os.path.join(tmp_dirs["raw_img"], "blender_0001.png")
    with open(img_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    # Write metadata
    meta = _make_sample_metadata(visible_count=20)
    meta_path = os.path.join(tmp_dirs["raw_meta"], "blender_0001.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    count = convert_all(
        raw_dir=os.path.join(tmp_dirs["base"], "raw"),
        out_images=tmp_dirs["out_img"],
        out_annotations=tmp_dirs["out_ann"],
        min_visible=4,
    )

    assert count == 1
    assert os.path.isfile(os.path.join(tmp_dirs["out_img"], "blender_0001.png"))

    ann_path = os.path.join(tmp_dirs["out_ann"], "blender_0001.json")
    assert os.path.isfile(ann_path)
    with open(ann_path) as f:
        ann = json.load(f)
    assert ann["image_path"] == "blender_0001.png"
    assert ann["image_size"] == [640, 640]
    assert len(ann["keypoints"]) == 30
    assert len(ann["visibility"]) == 30
    assert len(ann["bounding_box"]) == 4


def test_convert_all_skips_low_visibility(tmp_dirs):
    img_path = os.path.join(tmp_dirs["raw_img"], "blender_0002.png")
    with open(img_path, "wb") as f:
        f.write(b"\x89PNG" + b"\x00" * 100)

    meta = _make_sample_metadata(visible_count=2)
    meta["image_file"] = "blender_0002.png"
    meta_path = os.path.join(tmp_dirs["raw_meta"], "blender_0002.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    count = convert_all(
        raw_dir=os.path.join(tmp_dirs["base"], "raw"),
        out_images=tmp_dirs["out_img"],
        out_annotations=tmp_dirs["out_ann"],
        min_visible=4,
    )

    assert count == 0
    assert not os.path.isfile(os.path.join(tmp_dirs["out_ann"], "blender_0002.json"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_blender_to_cvn.py -v
```
Expected: All 6 tests FAIL with `ModuleNotFoundError: No module named 'src.tools.blender_to_cvn'`.

- [ ] **Step 3: Implement the converter**

```python
"""Convert Blender render metadata to CVN annotation format.

Reads from data/blender/raw/ (images + metadata JSONs), writes
CVN-format annotations to data/blender/annotations/ and copies
images to data/blender/images/.

Usage:
    python src/tools/blender_to_cvn.py [--min-visible 4]
"""

import json
import os
import shutil
import sys

NUM_KEYPOINTS = 30


def convert_metadata(meta, min_visible=4):
    """Convert a single Blender metadata dict to CVN annotation format.

    Args:
        meta: dict from blender_render.py metadata JSON
        min_visible: minimum visible keypoints to accept

    Returns:
        CVN annotation dict, or None if below min_visible threshold
    """
    cam = meta["camera"]
    resolution = meta["resolution"]
    res_x, res_y = resolution

    raw_kp = cam["keypoints_2d"]
    raw_vis = cam["visibility"]

    keypoints = []
    visibility = []

    for i in range(NUM_KEYPOINTS):
        if raw_vis[i]:
            nx = raw_kp[i][0] / res_x
            ny = raw_kp[i][1] / res_y
            keypoints.append([nx, ny])
            visibility.append(1)
        else:
            keypoints.append([-1.0, -1.0])
            visibility.append(0)

    n_visible = sum(visibility)
    if n_visible < min_visible:
        return None

    vis_kps = [i for i in range(NUM_KEYPOINTS) if visibility[i]]
    xs = [keypoints[i][0] for i in vis_kps]
    ys = [keypoints[i][1] for i in vis_kps]
    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]

    return {
        "image_path": meta["image_file"],
        "image_size": [res_y, res_x],
        "keypoints": keypoints,
        "visibility": visibility,
        "bounding_box": bbox,
    }


def convert_all(raw_dir, out_images, out_annotations, min_visible=4):
    """Convert all Blender raw output to CVN format.

    Args:
        raw_dir: path to data/blender/raw/
        out_images: path to data/blender/images/
        out_annotations: path to data/blender/annotations/
        min_visible: minimum visible keypoints

    Returns:
        number of images converted
    """
    raw_images = os.path.join(raw_dir, "images")
    raw_metadata = os.path.join(raw_dir, "metadata")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_annotations, exist_ok=True)

    count = 0
    for fname in sorted(os.listdir(raw_metadata)):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(raw_metadata, fname)) as f:
            meta = json.load(f)

        ann = convert_metadata(meta, min_visible=min_visible)
        if ann is None:
            continue

        img_name = meta["image_file"]
        src_img = os.path.join(raw_images, img_name)
        if not os.path.isfile(src_img):
            continue

        shutil.copy2(src_img, os.path.join(out_images, img_name))

        stem = os.path.splitext(img_name)[0]
        ann_path = os.path.join(out_annotations, f"{stem}.json")
        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2)

        count += 1

    return count


def main():
    min_visible = 4
    if "--min-visible" in sys.argv:
        idx = sys.argv.index("--min-visible")
        min_visible = int(sys.argv[idx + 1])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(script_dir))
    blender_dir = os.path.join(root, "data", "blender")

    raw_dir = os.path.join(blender_dir, "raw")
    out_images = os.path.join(blender_dir, "images")
    out_annotations = os.path.join(blender_dir, "annotations")

    if not os.path.isdir(raw_dir):
        print(f"No raw directory found at {raw_dir}")
        print("Run blender_render.py first.")
        sys.exit(1)

    print(f"Converting Blender raw output to CVN format...")
    print(f"  Raw:         {raw_dir}")
    print(f"  Images out:  {out_images}")
    print(f"  Ann out:     {out_annotations}")
    print(f"  Min visible: {min_visible}")

    count = convert_all(raw_dir, out_images, out_annotations, min_visible)
    print(f"\nDone: {count} images converted")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_blender_to_cvn.py -v
```
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/blender_to_cvn.py tests/test_blender_to_cvn.py
git commit -m "feat(blender): add converter from raw Blender metadata to CVN format"
```

---

### Task 8: Integration Test — Full Pipeline

**Files:**
- No new files — this task validates the end-to-end pipeline

**Interfaces:**
- Consumes: All modules from tasks 1-7

- [ ] **Step 1: Run a small batch render**

```bash
blender --background data/blender/court_template.blend --python src/tools/blender_render.py -- --count 5 --engine EEVEE --samples 16 --seed 42
```
Expected: 5 images + 5 metadata JSONs in `data/blender/raw/`.

- [ ] **Step 2: Convert to CVN format**

```bash
python src/tools/blender_to_cvn.py --min-visible 4
```
Expected: Prints conversion count. Images + annotations appear in `data/blender/images/` and `data/blender/annotations/`.

- [ ] **Step 3: Verify dashboard integration**

Start the annotator server and check the dashboard:
```bash
python src/tools/serve_annotator.py
```
Open `http://localhost:8000/dashboard` — the "Blender Synthetic" source should appear with the correct count and thumbnails should load.

- [ ] **Step 4: Verify annotation format matches CourtDataset expectations**

```bash
python -c "
import json, os, glob
ann_dir = 'data/blender/annotations'
for f in sorted(glob.glob(os.path.join(ann_dir, '*.json')))[:2]:
    with open(f) as fh:
        a = json.load(fh)
    print(f'{os.path.basename(f)}:')
    print(f'  image_path: {a[\"image_path\"]}')
    print(f'  image_size: {a[\"image_size\"]}')
    print(f'  keypoints: {len(a[\"keypoints\"])} points')
    print(f'  visibility: {sum(a[\"visibility\"])}/30 visible')
    print(f'  bbox: {a[\"bounding_box\"]}')
"
```
Expected: Each annotation has 30 keypoints, valid visibility, bounding box with positive width/height.

- [ ] **Step 5: Commit integration test results**

```bash
git add -A
git commit -m "feat(blender): complete synthetic data pipeline — render + convert + dashboard"
```

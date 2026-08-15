# Environment Realism Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add procedural floor textures, wall/ceiling details, articulated mannequin players, improved seating, net post improvements, light housings, line judge chairs, and courtside clutter to the Blender synthetic data pipeline — making renders closely resemble real badminton venue photographs.

**Architecture:** 2 new modules (`blender_mannequin.py`, `blender_venue_details.py`) and 4 enhanced modules (`blender_environment.py`, `blender_court.py`, `blender_occluders.py`, `blender_render.py`). All materials are fully procedural (shader nodes, no external textures). Every element is randomized per render via the existing seed system. A Blender smoke test script validates the full pipeline end-to-end.

**Tech Stack:** Blender 5.2+ Python API (`bpy`), procedural shader nodes (Noise Texture, Voronoi Texture, Wave Texture, Musgrave Texture, Gradient Texture, ColorRamp, MixRGB)

## Global Constraints

- Blender 5.2+ — use `users_collection` pattern for collection management, `BLENDER_EEVEE` engine name
- All materials fully procedural — no external image texture files
- No changes to keypoint positions, visibility computation, or CVN annotation format
- Metadata additions are backward-compatible (new fields only)
- All randomization seeded via the existing `random.seed(SEED)` in the render loop
- Never add Claude as co-author/contributor in commits or repo
- Court constants: `COURT_LENGTH=13.4`, `COURT_WIDTH=6.1`, `NET_POS=6.7` (meters, 1 BU = 1m)

---

### Task 1: Procedural Floor Textures and Environment Return Value

**Files:**
- Modify: `src/tools/blender_environment.py:27-63` (FLOOR_COLORS dict, `_build_venue_floor_at` function, `build_environment` return)

**Interfaces:**
- Consumes: existing `build_environment(config)` API
- Produces: `build_environment(config)` now returns additional keys: `"floor_type"` (str), `"venue_bounds"` (dict with `cx`, `cy`, `d`, `w`, `h`)

**Context:** Currently `_build_venue_floor_at` creates a flat-color `Principled BSDF` material. We replace this with procedural node graphs for three floor types: concrete, rubber, and wood. We also update `build_environment()` to return `floor_type` and `venue_bounds` which downstream modules (`build_venue_details`) need.

- [ ] **Step 1: Update FLOOR_COLORS to FLOOR_TYPES and refactor `_build_venue_floor_at` to use procedural materials**

Replace the flat FLOOR_COLORS dict and the `_build_venue_floor_at` function. The new function creates procedural shader node graphs for each floor type.

In `src/tools/blender_environment.py`, replace the `FLOOR_COLORS` dict (line 27-31) with:

```python
FLOOR_TYPES = ["concrete", "rubber", "wood"]
```

Replace `_build_venue_floor_at` (lines 42-63) with:

```python
def _build_venue_floor_at(collection, cx, cy, venue_d, venue_w):
    """Extended floor plane with procedural material."""
    floor_type = random.choice(FLOOR_TYPES)

    bpy.ops.mesh.primitive_plane_add(size=1, location=(cx, cy, -0.005))
    floor = bpy.context.active_object
    floor.name = "VenueFloor"
    floor.scale = (venue_d, venue_w, 1)
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.new("VenueFloorMat")
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    bsdf = nodes["Principled BSDF"]

    if floor_type == "concrete":
        _build_concrete_floor(tree, nodes, links, bsdf)
    elif floor_type == "rubber":
        _build_rubber_floor(tree, nodes, links, bsdf)
    else:
        _build_wood_floor(tree, nodes, links, bsdf)

    floor.data.materials.append(mat)

    for col_ in list(floor.users_collection):
        col_.objects.unlink(floor)
    collection.objects.link(floor)

    return floor, floor_type


def _build_concrete_floor(tree, nodes, links, bsdf):
    """Concrete: noise grain + musgrave wear patches."""
    base_color = (
        0.45 + random.uniform(-0.05, 0.05),
        0.43 + random.uniform(-0.05, 0.05),
        0.40 + random.uniform(-0.05, 0.05),
        1.0,
    )

    tex_coord = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = random.uniform(50, 100)
    noise.inputs["Detail"].default_value = 6.0

    musgrave = nodes.new("ShaderNodeTexMusgrave")
    musgrave.inputs["Scale"].default_value = random.uniform(200, 400)
    musgrave.inputs["Detail"].default_value = 4.0

    base_rgb = nodes.new("ShaderNodeRGB")
    base_rgb.outputs[0].default_value = base_color

    mix_grain = nodes.new("ShaderNodeMixRGB")
    mix_grain.blend_type = 'MIX'
    mix_grain.inputs["Fac"].default_value = 0.15

    mix_wear = nodes.new("ShaderNodeMixRGB")
    mix_wear.blend_type = 'MULTIPLY'
    mix_wear.inputs["Fac"].default_value = 0.1

    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], musgrave.inputs["Vector"])
    links.new(base_rgb.outputs[0], mix_grain.inputs["Color1"])
    links.new(noise.outputs["Fac"], mix_grain.inputs["Color2"])
    links.new(mix_grain.outputs[0], mix_wear.inputs["Color1"])
    links.new(musgrave.outputs["Fac"], mix_wear.inputs["Color2"])
    links.new(mix_wear.outputs[0], bsdf.inputs["Base Color"])

    noise_rough = nodes.new("ShaderNodeTexNoise")
    noise_rough.inputs["Scale"].default_value = random.uniform(30, 60)
    links.new(tex_coord.outputs["Object"], noise_rough.inputs["Vector"])

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 0.0
    map_range.inputs["From Max"].default_value = 1.0
    map_range.inputs["To Min"].default_value = 0.7
    map_range.inputs["To Max"].default_value = 0.95
    links.new(noise_rough.outputs["Fac"], map_range.inputs["Value"])
    links.new(map_range.outputs[0], bsdf.inputs["Roughness"])


def _build_rubber_floor(tree, nodes, links, bsdf):
    """Rubber: voronoi tile pattern with color variation."""
    base_color = (
        0.15 + random.uniform(-0.03, 0.03),
        0.15 + random.uniform(-0.03, 0.03),
        0.18 + random.uniform(-0.03, 0.03),
        1.0,
    )

    tex_coord = nodes.new("ShaderNodeTexCoord")
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.inputs["Scale"].default_value = random.uniform(30, 60)

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = base_color
    lighter = tuple(min(1.0, c + 0.06) for c in base_color[:3]) + (1.0,)
    ramp.color_ramp.elements[1].color = lighter

    links.new(tex_coord.outputs["Object"], voronoi.inputs["Vector"])
    links.new(voronoi.outputs["Distance"], ramp.inputs["Fac"])
    links.new(ramp.outputs[0], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = random.uniform(0.85, 0.95)


def _build_wood_floor(tree, nodes, links, bsdf):
    """Wood: wave grain + noise knots."""
    base_color = (
        0.50 + random.uniform(-0.05, 0.05),
        0.35 + random.uniform(-0.05, 0.05),
        0.20 + random.uniform(-0.05, 0.05),
        1.0,
    )

    tex_coord = nodes.new("ShaderNodeTexCoord")
    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = 'BANDS'
    wave.inputs["Scale"].default_value = random.uniform(20, 40)
    wave.inputs["Distortion"].default_value = 3.0

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = random.uniform(40, 80)
    noise.inputs["Detail"].default_value = 4.0

    base_rgb = nodes.new("ShaderNodeRGB")
    base_rgb.outputs[0].default_value = base_color

    mix_grain = nodes.new("ShaderNodeMixRGB")
    mix_grain.blend_type = 'MIX'
    mix_grain.inputs["Fac"].default_value = 0.2

    mix_knots = nodes.new("ShaderNodeMixRGB")
    mix_knots.blend_type = 'MIX'
    mix_knots.inputs["Fac"].default_value = 0.1

    links.new(tex_coord.outputs["Object"], wave.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])
    links.new(base_rgb.outputs[0], mix_grain.inputs["Color1"])
    links.new(wave.outputs["Fac"], mix_grain.inputs["Color2"])
    links.new(mix_grain.outputs[0], mix_knots.inputs["Color1"])
    links.new(noise.outputs["Fac"], mix_knots.inputs["Color2"])
    links.new(mix_knots.outputs[0], bsdf.inputs["Base Color"])

    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 0.0
    map_range.inputs["From Max"].default_value = 1.0
    map_range.inputs["To Min"].default_value = 0.5
    map_range.inputs["To Max"].default_value = 0.7
    links.new(wave.outputs["Fac"], map_range.inputs["Value"])
    links.new(map_range.outputs[0], bsdf.inputs["Roughness"])
```

- [ ] **Step 2: Update `build_environment()` to return venue_bounds and floor_type**

In `build_environment()`, update the `_build_venue_floor_at` call (line 316) to capture the floor_type return:

```python
    floor, floor_type = _build_venue_floor_at(col, venue_cx, venue_cy, venue_d, venue_w)
```

Update the return dict at line 342 to include the new fields:

```python
    return {
        "adjacent_courts": len(slots),
        "venue_size": venue_size,
        "has_dividers": has_dividers,
        "has_spectators": has_spectators,
        "has_scoreboard": has_scoreboard,
        "floor_type": floor_type,
        "venue_bounds": {
            "cx": venue_cx,
            "cy": venue_cy,
            "d": venue_d,
            "w": venue_w,
            "h": venue_h,
        },
    }
```

- [ ] **Step 3: Run Blender smoke test**

Run: `"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python src/tools/blender_render.py -- --count 1 --engine BLENDER_EEVEE --samples 16 --seed 42`

Expected: Renders 1 image without errors. Check the metadata JSON in `data/blender/raw/metadata/` contains `floor_type` and `venue_bounds` keys in the environment section.

Note: `venue_bounds` won't appear in metadata yet until Task 8 updates the render loop, but the function should return it correctly. Verify no Blender Python errors in console output.

- [ ] **Step 4: Commit**

```bash
git add src/tools/blender_environment.py
git commit -m "feat(blender): add procedural floor textures and venue_bounds to environment"
```

---

### Task 2: Court Surface Texture and Net Post Improvements

**Files:**
- Modify: `src/tools/blender_court.py:70-90` (`_build_surface`), `src/tools/blender_court.py:186-222` (`_build_net`)

**Interfaces:**
- Consumes: existing `build_court(config)` API
- Produces: `build_court(config)` — same return shape, net posts now have base plates + tension cable + optional padding

**Context:** The court surface currently uses a flat `Principled BSDF`. We add subtle noise texture for surface variation and optional tape residue rectangles. Net posts get base plates, a tension cable connecting tops, and optional padding.

- [ ] **Step 1: Add subtle noise texture to court surface**

In `src/tools/blender_court.py`, replace `_build_surface` (lines 70-90) with:

```python
def _build_surface(collection, config):
    """Create the court floor plane with procedural surface texture."""
    color_name = config.get("surface_color", "green")
    if color_name == "random":
        color_name = random.choice(list(SURFACE_COLORS.keys()))
    color = SURFACE_COLORS[color_name]

    bpy.ops.mesh.primitive_plane_add(size=1, location=(COURT_LENGTH / 2, COURT_WIDTH / 2, 0))
    surface = bpy.context.active_object
    surface.name = "CourtSurface"
    surface.scale = (COURT_LENGTH, COURT_WIDTH, 1)
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.new("CourtSurfaceMat")
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.7

    tex_coord = nodes.new("ShaderNodeTexCoord")

    base_rgb = nodes.new("ShaderNodeRGB")
    base_rgb.outputs[0].default_value = color

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = random.uniform(100, 200)
    noise.inputs["Detail"].default_value = 4.0

    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = 'MIX'
    mix.inputs["Fac"].default_value = 0.05

    gradient = nodes.new("ShaderNodeTexGradient")
    mix_grad = nodes.new("ShaderNodeMixRGB")
    mix_grad.blend_type = 'MIX'
    mix_grad.inputs["Fac"].default_value = 0.03

    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], gradient.inputs["Vector"])
    links.new(base_rgb.outputs[0], mix.inputs["Color1"])
    links.new(noise.outputs["Fac"], mix.inputs["Color2"])
    links.new(mix.outputs[0], mix_grad.inputs["Color1"])
    links.new(gradient.outputs["Fac"], mix_grad.inputs["Color2"])
    links.new(mix_grad.outputs[0], bsdf.inputs["Base Color"])

    surface.data.materials.append(mat)

    for col in list(surface.users_collection):
        col.objects.unlink(surface)
    collection.objects.link(surface)

    return surface, color_name
```

- [ ] **Step 2: Add tape residue rectangles**

Add this function before `_build_surface` in `blender_court.py`:

```python
def _add_tape_residue(collection):
    """Add 0-5 small tape residue rectangles on the court surface."""
    if random.random() > 0.4:
        return
    count = random.randint(1, 5)
    mat = bpy.data.materials.new("TapeResidueMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.75, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6

    for i in range(count):
        x = random.uniform(0.5, COURT_LENGTH - 0.5)
        y = random.uniform(0.5, COURT_WIDTH - 0.5)
        w = random.uniform(0.1, 0.3)
        h = random.uniform(0.1, 0.3)
        bpy.ops.mesh.primitive_plane_add(size=1, location=(x, y, 0.0005))
        obj = bpy.context.active_object
        obj.name = f"TapeResidue_{i}"
        obj.scale = (w, h, 1)
        bpy.ops.object.transform_apply(scale=True)
        obj.data.materials.append(mat)
        for col in list(obj.users_collection):
            col.objects.unlink(obj)
        collection.objects.link(obj)
```

Call it in `build_court()` after `_build_lines`, before `_build_keypoints`:

```python
    _add_tape_residue(court_col)
```

- [ ] **Step 3: Add net post base plates and tension cable**

In `_build_net`, after creating the two posts (after the `for` loop at line 206), add base plates and cable:

```python
    # Base plates
    for i, y_pos in enumerate([-POST_EXTENSION, COURT_WIDTH + POST_EXTENSION]):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.15, depth=0.02,
            location=(NET_POS, y_pos, 0.01),
        )
        plate = bpy.context.active_object
        plate.name = f"PostBasePlate_{i}"
        plate.data.materials.append(mat_post)
        for col in list(plate.users_collection):
            col.objects.unlink(plate)
        collection.objects.link(plate)

    # Tension cable connecting post tops
    cable_y_start = -POST_EXTENSION
    cable_y_end = COURT_WIDTH + POST_EXTENSION
    cable_len = cable_y_end - cable_y_start
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.005, depth=cable_len,
        location=(NET_POS, (cable_y_start + cable_y_end) / 2, NET_HEIGHT_EDGE),
        rotation=(math.pi / 2, 0, 0),
    )
    cable = bpy.context.active_object
    cable.name = "NetTensionCable"
    cable.data.materials.append(mat_post)
    for col in list(cable.users_collection):
        col.objects.unlink(cable)
    collection.objects.link(cable)
```

Add `import math` at the top of `blender_court.py` (it's not currently imported).

- [ ] **Step 4: Add optional post padding**

After the tension cable code, add:

```python
    # Post padding (~30% chance)
    if random.random() < 0.3:
        pad_colors = [
            (0.8, 0.7, 0.1, 1.0),   # yellow
            (0.1, 0.3, 0.8, 1.0),   # blue
            (0.8, 0.15, 0.1, 1.0),  # red
        ]
        pad_color = random.choice(pad_colors)
        mat_pad = _make_material("PostPaddingMat", pad_color, roughness=0.9)
        for i, y_pos in enumerate([-POST_EXTENSION, COURT_WIDTH + POST_EXTENSION]):
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.08, depth=1.0,
                location=(NET_POS, y_pos, 0.5),
            )
            pad = bpy.context.active_object
            pad.name = f"PostPadding_{i}"
            pad.data.materials.append(mat_pad)
            for col in list(pad.users_collection):
                col.objects.unlink(pad)
            collection.objects.link(pad)
```

- [ ] **Step 5: Run Blender smoke test**

Run: `"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python src/tools/blender_render.py -- --count 1 --engine BLENDER_EEVEE --samples 16 --seed 42`

Expected: Renders without errors. Net posts now have base plates and tension cable visible.

- [ ] **Step 6: Commit**

```bash
git add src/tools/blender_court.py
git commit -m "feat(blender): add court surface texture, tape residue, and net post improvements"
```

---

### Task 3: Floor Markings

**Files:**
- Modify: `src/tools/blender_environment.py` (add `_build_floor_markings` function, call it in `build_environment`)

**Interfaces:**
- Consumes: venue_cx, venue_cy, venue_d, venue_w from `build_environment` locals
- Produces: `build_environment` return dict gains `"has_floor_markings"` (bool)

**Context:** Add safety zone tape outlines, court number markers, and warm-up area rectangles to the venue floor outside the main court area. These are colored markings commonly found in real venues.

- [ ] **Step 1: Add `_build_floor_markings` function**

Add this function in `blender_environment.py` before `build_environment`:

```python
def _build_floor_markings(collection, venue_cx, venue_cy, venue_d, venue_w):
    """Add safety zones, court number, and warm-up areas to the venue floor."""
    markings_added = False

    # Safety zone tape (~60% chance)
    if random.random() < 0.6:
        markings_added = True
        tape_color_options = [
            (0.8, 0.7, 0.1, 1.0),   # yellow
            (0.7, 0.1, 0.1, 1.0),   # red
            (0.1, 0.2, 0.7, 1.0),   # blue
        ]
        tape_color = random.choice(tape_color_options)
        mat = bpy.data.materials.new("SafetyTapeMat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = tape_color
        bsdf.inputs["Roughness"].default_value = 0.6

        tape_w = 0.05  # 5cm wide tape
        offset = random.uniform(1.0, 2.0)
        # Rectangle around main court
        tape_rects = [
            # (cx, cy, sx, sy) — position and scale
            (COURT_LENGTH / 2, -offset, COURT_LENGTH + 2 * offset, tape_w),
            (COURT_LENGTH / 2, COURT_WIDTH + offset, COURT_LENGTH + 2 * offset, tape_w),
            (-offset, COURT_WIDTH / 2, tape_w, COURT_WIDTH + 2 * offset),
            (COURT_LENGTH + offset, COURT_WIDTH / 2, tape_w, COURT_WIDTH + 2 * offset),
        ]
        for i, (tx, ty, sx, sy) in enumerate(tape_rects):
            bpy.ops.mesh.primitive_plane_add(size=1, location=(tx, ty, 0.0003))
            obj = bpy.context.active_object
            obj.name = f"SafetyTape_{i}"
            obj.scale = (sx, sy, 1)
            bpy.ops.object.transform_apply(scale=True)
            obj.data.materials.append(mat)
            for col_ in list(obj.users_collection):
                col_.objects.unlink(obj)
            collection.objects.link(obj)

    # Court number marker (~40% chance)
    if random.random() < 0.4:
        markings_added = True
        mat = bpy.data.materials.new("CourtNumberMat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.1, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5

        side = random.choice([0, COURT_LENGTH])
        bpy.ops.mesh.primitive_plane_add(
            size=1, location=(side + (0.5 if side == 0 else -0.5),
                              COURT_WIDTH / 2, 0.0003),
        )
        obj = bpy.context.active_object
        obj.name = "CourtNumber"
        obj.scale = (0.8, 0.4, 1)
        bpy.ops.object.transform_apply(scale=True)
        obj.data.materials.append(mat)
        for col_ in list(obj.users_collection):
            col_.objects.unlink(obj)
        collection.objects.link(obj)

    # Warm-up areas (~30% chance)
    if random.random() < 0.3:
        markings_added = True
        warmup_color = random.choice([
            (0.2, 0.5, 0.2, 1.0),
            (0.2, 0.2, 0.6, 1.0),
        ])
        mat = bpy.data.materials.new("WarmupAreaMat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = warmup_color
        bsdf.inputs["Roughness"].default_value = 0.75

        corners = [
            (venue_cx - venue_d / 2 + 2, venue_cy - venue_w / 2 + 2),
            (venue_cx + venue_d / 2 - 2, venue_cy + venue_w / 2 - 2),
        ]
        num = random.randint(1, 2)
        chosen = random.sample(corners, k=min(num, len(corners)))
        for i, (wx, wy) in enumerate(chosen):
            bpy.ops.mesh.primitive_plane_add(size=1, location=(wx, wy, 0.0002))
            obj = bpy.context.active_object
            obj.name = f"WarmupArea_{i}"
            obj.scale = (3, 2, 1)
            bpy.ops.object.transform_apply(scale=True)
            obj.data.materials.append(mat)
            for col_ in list(obj.users_collection):
                col_.objects.unlink(obj)
            collection.objects.link(obj)

    return markings_added
```

- [ ] **Step 2: Call `_build_floor_markings` in `build_environment` and update return**

In `build_environment()`, after the `_build_walls_and_ceiling_at` call and before the dividers section, add:

```python
    has_floor_markings = _build_floor_markings(col, venue_cx, venue_cy, venue_d, venue_w)
```

Add `"has_floor_markings": has_floor_markings` to the return dict.

- [ ] **Step 3: Run Blender smoke test**

Run: `"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python src/tools/blender_render.py -- --count 1 --engine BLENDER_EEVEE --samples 16 --seed 42`

Expected: No errors. Floor markings appear depending on random seed.

- [ ] **Step 4: Commit**

```bash
git add src/tools/blender_environment.py
git commit -m "feat(blender): add floor markings (safety tape, court numbers, warm-up areas)"
```

---

### Task 4: Articulated Mannequin Players

**Files:**
- Create: `src/tools/blender_mannequin.py`

**Interfaces:**
- Consumes: `bpy`, `random`
- Produces: `build_mannequin(collection, index, config)` → dict with `type`, `position`, `pose`, `has_racket`, `facing_angle`

**Context:** New module that builds articulated player figures from cylinders and spheres. Parent-child hierarchy so rotating a joint moves all children. Five pose presets with random jitter. Expanded clothing colors and skin tones.

- [ ] **Step 1: Create the mannequin module with constants and materials**

Create `src/tools/blender_mannequin.py`:

```python
"""Articulated mannequin players for badminton court scenes."""

import bpy
import math
import random

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1

POSE_PRESETS = ["standing", "ready", "lunging", "serving", "walking"]

SHIRT_COLORS = [
    (0.05, 0.05, 0.25, 1.0),   # navy
    (0.6, 0.1, 0.08, 1.0),     # red
    (0.08, 0.4, 0.12, 1.0),    # green
    (0.3, 0.3, 0.3, 1.0),      # grey
    (0.9, 0.9, 0.9, 1.0),      # white
    (0.85, 0.75, 0.1, 1.0),    # yellow
    (0.1, 0.3, 0.7, 1.0),      # blue
    (0.8, 0.4, 0.1, 1.0),      # orange
    (0.8, 0.4, 0.6, 1.0),      # pink
    (0.1, 0.5, 0.5, 1.0),      # teal
    (0.08, 0.08, 0.08, 1.0),   # black
]

SHORTS_COLORS = [
    (0.08, 0.08, 0.08, 1.0),   # black
    (0.05, 0.05, 0.25, 1.0),   # navy
    (0.9, 0.9, 0.9, 1.0),      # white
    (0.3, 0.3, 0.3, 1.0),      # grey
]

SKIN_TONES = [
    (0.95, 0.82, 0.70, 1.0),
    (0.80, 0.62, 0.47, 1.0),
    (0.65, 0.48, 0.35, 1.0),
    (0.50, 0.35, 0.25, 1.0),
    (0.38, 0.26, 0.18, 1.0),
    (0.28, 0.18, 0.12, 1.0),
]


def _make_mat(name, color, roughness=0.8):
    """Create a simple material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat
```

- [ ] **Step 2: Add body part builder with parent-child hierarchy**

Append to `blender_mannequin.py`:

```python
def _add_cylinder(name, r, h, location, parent, collection, mat):
    """Add a cylinder body part parented to another object."""
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    if parent:
        obj.parent = parent
        obj.location = (
            location[0] - parent.location[0],
            location[1] - parent.location[1],
            location[2] - parent.location[2],
        )
    for col_ in list(obj.users_collection):
        col_.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def _add_sphere(name, r, location, parent, collection, mat):
    """Add a sphere body part parented to another object."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    if parent:
        obj.parent = parent
        obj.location = (
            location[0] - parent.location[0],
            location[1] - parent.location[1],
            location[2] - parent.location[2],
        )
    for col_ in list(obj.users_collection):
        col_.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def _add_cube(name, dims, location, parent, collection, mat):
    """Add a cube body part parented to another object."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (dims[0], dims[1], dims[2])
    bpy.ops.object.transform_apply(scale=True)
    obj.data.materials.append(mat)
    if parent:
        obj.parent = parent
        obj.location = (
            location[0] - parent.location[0],
            location[1] - parent.location[1],
            location[2] - parent.location[2],
        )
    for col_ in list(obj.users_collection):
        col_.objects.unlink(obj)
    collection.objects.link(obj)
    return obj
```

- [ ] **Step 3: Add pose preset functions**

Append to `blender_mannequin.py`:

```python
def _jitter(deg, amount=7):
    """Add random jitter to an angle in degrees."""
    return deg + random.uniform(-amount, amount)


def _apply_pose(parts, preset, mirror=False):
    """Apply joint rotations for a pose preset.

    parts dict keys: torso, head, upper_arm_l, upper_arm_r,
    forearm_l, forearm_r, upper_leg_l, upper_leg_r,
    lower_leg_l, lower_leg_r
    """
    sign = -1 if mirror else 1

    if preset == "standing":
        parts["upper_arm_l"].rotation_euler.x = math.radians(_jitter(5))
        parts["upper_arm_r"].rotation_euler.x = math.radians(_jitter(5))
        parts["forearm_l"].rotation_euler.x = math.radians(_jitter(10))
        parts["forearm_r"].rotation_euler.x = math.radians(_jitter(10))

    elif preset == "ready":
        parts["torso"].rotation_euler.x = math.radians(_jitter(5))
        parts["upper_arm_l"].rotation_euler.x = math.radians(_jitter(40))
        parts["upper_arm_r"].rotation_euler.x = math.radians(_jitter(40))
        parts["forearm_l"].rotation_euler.x = math.radians(_jitter(90))
        parts["forearm_r"].rotation_euler.x = math.radians(_jitter(90))
        parts["upper_leg_l"].rotation_euler.x = math.radians(_jitter(20))
        parts["upper_leg_r"].rotation_euler.x = math.radians(_jitter(20))
        parts["lower_leg_l"].rotation_euler.x = math.radians(_jitter(-30))
        parts["lower_leg_r"].rotation_euler.x = math.radians(_jitter(-30))

    elif preset == "lunging":
        parts["torso"].rotation_euler.x = math.radians(_jitter(15))
        # Lead leg
        lead_leg = "upper_leg_r" if not mirror else "upper_leg_l"
        trail_leg = "upper_leg_l" if not mirror else "upper_leg_r"
        lead_lower = "lower_leg_r" if not mirror else "lower_leg_l"
        trail_lower = "lower_leg_l" if not mirror else "lower_leg_r"
        parts[lead_leg].rotation_euler.x = math.radians(_jitter(70))
        parts[lead_lower].rotation_euler.x = math.radians(_jitter(-90))
        parts[trail_leg].rotation_euler.x = math.radians(_jitter(-10))
        parts[trail_lower].rotation_euler.x = math.radians(_jitter(-5))
        # Racket arm
        racket_arm = "upper_arm_r" if not mirror else "upper_arm_l"
        off_arm = "upper_arm_l" if not mirror else "upper_arm_r"
        racket_fore = "forearm_r" if not mirror else "forearm_l"
        off_fore = "forearm_l" if not mirror else "forearm_r"
        parts[racket_arm].rotation_euler.x = math.radians(_jitter(60))
        parts[racket_arm].rotation_euler.y = math.radians(_jitter(30 * sign))
        parts[racket_fore].rotation_euler.x = math.radians(_jitter(120))
        parts[off_arm].rotation_euler.x = math.radians(_jitter(-30))
        parts[off_fore].rotation_euler.x = math.radians(_jitter(45))

    elif preset == "serving":
        parts["torso"].rotation_euler.x = math.radians(_jitter(-10))
        parts["upper_arm_r"].rotation_euler.x = math.radians(_jitter(160))
        parts["forearm_r"].rotation_euler.x = math.radians(_jitter(90))
        parts["upper_arm_l"].rotation_euler.x = math.radians(_jitter(70))
        parts["forearm_l"].rotation_euler.x = math.radians(_jitter(10))
        parts["upper_leg_l"].rotation_euler.x = math.radians(_jitter(5))
        parts["upper_leg_r"].rotation_euler.x = math.radians(_jitter(-5))
        parts["lower_leg_l"].rotation_euler.x = math.radians(_jitter(-10))
        parts["lower_leg_r"].rotation_euler.x = math.radians(_jitter(-10))

    elif preset == "walking":
        phase = random.choice([1, -1])
        parts["upper_arm_l"].rotation_euler.x = math.radians(_jitter(15 * phase))
        parts["upper_arm_r"].rotation_euler.x = math.radians(_jitter(-15 * phase))
        parts["forearm_l"].rotation_euler.x = math.radians(_jitter(20))
        parts["forearm_r"].rotation_euler.x = math.radians(_jitter(20))
        parts["upper_leg_l"].rotation_euler.x = math.radians(_jitter(-20 * phase))
        parts["upper_leg_r"].rotation_euler.x = math.radians(_jitter(20 * phase))
        parts["lower_leg_l"].rotation_euler.x = math.radians(_jitter(-10 if phase > 0 else 0))
        parts["lower_leg_r"].rotation_euler.x = math.radians(_jitter(-10 if phase < 0 else 0))
```

- [ ] **Step 4: Add `build_mannequin` main function**

Append to `blender_mannequin.py`:

```python
def build_mannequin(collection, index, config=None):
    """Build an articulated mannequin player figure.

    Args:
        collection: Blender collection to add objects to
        index: player index for naming
        config: optional dict (unused for now, reserved for future overrides)

    Returns:
        dict with type, position, pose, has_racket, facing_angle metadata
    """
    # Position on court (Gaussian bias toward center)
    x = random.gauss(COURT_LENGTH / 2, COURT_LENGTH / 4)
    x = max(0.5, min(COURT_LENGTH - 0.5, x))
    y = random.uniform(0.5, COURT_WIDTH - 0.5)

    # Materials
    skin_tone = random.choice(SKIN_TONES)
    skin_mat = _make_mat(f"Skin_{index}", skin_tone, roughness=0.6)
    shirt_mat = _make_mat(f"Shirt_{index}", random.choice(SHIRT_COLORS))
    shorts_mat = _make_mat(f"Shorts_{index}", random.choice(SHORTS_COLORS))
    shoe_color = random.choice([(0.9, 0.9, 0.9, 1.0), (0.1, 0.1, 0.1, 1.0)])
    shoe_mat = _make_mat(f"Shoe_{index}", shoe_color, roughness=0.7)

    # Torso (root)
    torso = _add_cylinder(f"Player_{index}_Torso", 0.18, 0.50,
                          (x, y, 1.05), None, collection, shirt_mat)

    # Head
    head = _add_sphere(f"Player_{index}_Head", 0.11,
                       (x, y, 1.50), torso, collection, skin_mat)

    # Upper arms
    upper_arm_l = _add_cylinder(f"Player_{index}_UpperArm_L", 0.05, 0.28,
                                (x - 0.25, y, 1.20), torso, collection, shirt_mat)
    upper_arm_r = _add_cylinder(f"Player_{index}_UpperArm_R", 0.05, 0.28,
                                (x + 0.25, y, 1.20), torso, collection, shirt_mat)

    # Forearms
    forearm_l = _add_cylinder(f"Player_{index}_Forearm_L", 0.04, 0.25,
                              (x - 0.25, y, 0.93), upper_arm_l, collection, skin_mat)
    forearm_r = _add_cylinder(f"Player_{index}_Forearm_R", 0.04, 0.25,
                              (x + 0.25, y, 0.93), upper_arm_r, collection, skin_mat)

    # Upper legs
    upper_leg_l = _add_cylinder(f"Player_{index}_UpperLeg_L", 0.07, 0.40,
                                (x - 0.10, y, 0.60), torso, collection, shorts_mat)
    upper_leg_r = _add_cylinder(f"Player_{index}_UpperLeg_R", 0.07, 0.40,
                                (x + 0.10, y, 0.60), torso, collection, shorts_mat)

    # Lower legs
    lower_leg_l = _add_cylinder(f"Player_{index}_LowerLeg_L", 0.05, 0.38,
                                (x - 0.10, y, 0.21), upper_leg_l, collection, skin_mat)
    lower_leg_r = _add_cylinder(f"Player_{index}_LowerLeg_R", 0.05, 0.38,
                                (x + 0.10, y, 0.21), upper_leg_r, collection, skin_mat)

    # Shoes
    _add_cube(f"Player_{index}_Shoe_L", (0.12, 0.08, 0.05),
              (x - 0.10, y, 0.025), lower_leg_l, collection, shoe_mat)
    _add_cube(f"Player_{index}_Shoe_R", (0.12, 0.08, 0.05),
              (x + 0.10, y, 0.025), lower_leg_r, collection, shoe_mat)

    # Hair (~60% chance)
    if random.random() < 0.6:
        hair_color = random.choice([
            (0.05, 0.03, 0.02, 1.0),
            (0.15, 0.10, 0.05, 1.0),
            (0.02, 0.02, 0.02, 1.0),
        ])
        hair_mat = _make_mat(f"Hair_{index}", hair_color)
        _add_sphere(f"Player_{index}_Hair", 0.09,
                    (x, y, 1.60), head, collection, hair_mat)

    # Racket (~30% chance)
    has_racket = random.random() < 0.3
    if has_racket:
        racket_mat = bpy.data.materials.new(f"Racket_{index}")
        racket_mat.use_nodes = True
        bsdf = racket_mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.15, 0.15, 0.2, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.3

        bpy.ops.mesh.primitive_plane_add(size=1, location=(x + 0.25, y, 0.70))
        racket = bpy.context.active_object
        racket.name = f"Player_{index}_Racket"
        racket.scale = (0.25, 0.10, 1)
        bpy.ops.object.transform_apply(scale=True)
        racket.data.materials.append(racket_mat)
        racket.parent = forearm_r
        racket.location = (0, 0, -0.18)
        for col_ in list(racket.users_collection):
            col_.objects.unlink(racket)
        collection.objects.link(racket)

    # Apply pose
    pose = random.choice(POSE_PRESETS)
    mirror = random.choice([True, False])
    parts = {
        "torso": torso, "head": head,
        "upper_arm_l": upper_arm_l, "upper_arm_r": upper_arm_r,
        "forearm_l": forearm_l, "forearm_r": forearm_r,
        "upper_leg_l": upper_leg_l, "upper_leg_r": upper_leg_r,
        "lower_leg_l": lower_leg_l, "lower_leg_r": lower_leg_r,
    }
    _apply_pose(parts, pose, mirror)

    # Random Y-axis rotation biased toward net
    net_dir = math.atan2(COURT_WIDTH / 2 - y, COURT_LENGTH / 2 - x)
    facing = net_dir + random.uniform(-math.radians(30), math.radians(30))
    torso.rotation_euler.z = facing

    return {
        "type": "player",
        "position": [x, y, 0],
        "pose": pose,
        "has_racket": has_racket,
        "facing_angle": round(facing, 3),
    }
```

- [ ] **Step 5: Smoke test the mannequin module in isolation**

Create a quick test script and run it:

Run: `"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python -c "import sys; sys.path.insert(0, '.'); exec(open('src/tools/blender_mannequin.py').read()); col = bpy.data.collections.new('Test'); bpy.context.scene.collection.children.link(col); result = build_mannequin(col, 0); print('OK:', result)"`

If that doesn't work due to inline `-c`, create a small test script temporarily, run it, then delete it. The important thing is that `build_mannequin` runs without error and returns a valid metadata dict.

- [ ] **Step 6: Commit**

```bash
git add src/tools/blender_mannequin.py
git commit -m "feat(blender): add articulated mannequin player module with pose presets"
```

---

### Task 5: Wire Mannequin into Occluders

**Files:**
- Modify: `src/tools/blender_occluders.py:49-88` (replace `_add_player`), `src/tools/blender_occluders.py:155-195` (update `add_occluders`)

**Interfaces:**
- Consumes: `build_mannequin(collection, index, config)` from `blender_mannequin.py`
- Produces: `add_occluders(config)` — occluder metadata now includes `pose`, `has_racket`, `facing_angle` per player

**Context:** Replace the simple cylinder+sphere player builder with the articulated mannequin. The old `_add_player` function and its helpers (`_random_clothing_color`) are deleted. Player metadata in the return dict is enriched with pose information.

- [ ] **Step 1: Replace `_add_player` with mannequin builder**

In `src/tools/blender_occluders.py`:

1. Remove `_random_clothing_color` function (lines 36-47)
2. Remove `_add_player` function (lines 49-88)
3. Add import at the top (after `import random`):

```python
from src.tools.blender_mannequin import build_mannequin
```

4. In `add_occluders`, replace the player creation loop (lines 178-180):

```python
    # Players
    num_players = random.randint(0, cfg["max_players"])
    for i in range(num_players):
        meta = build_mannequin(col, i)
        occluders.append(meta)
```

This is a drop-in replacement since `build_mannequin` returns a dict with `type: "player"` and `position`, matching the old interface plus extra fields.

- [ ] **Step 2: Run Blender smoke test**

Run: `"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python src/tools/blender_render.py -- --count 2 --engine BLENDER_EEVEE --samples 16 --seed 42`

Expected: Renders 2 images without errors. Check metadata JSONs — player occluders should now have `pose`, `has_racket`, and `facing_angle` fields.

- [ ] **Step 3: Commit**

```bash
git add src/tools/blender_occluders.py
git commit -m "feat(blender): replace simple player figures with articulated mannequins"
```

---

### Task 6: Venue Details Module — Walls and Ceiling

**Files:**
- Create: `src/tools/blender_venue_details.py`

**Interfaces:**
- Consumes: `config` dict with `venue_bounds` (`cx`, `cy`, `d`, `w`, `h`) from `build_environment`
- Produces: `build_venue_details(config)` → dict with detail flags (`has_trusses`, `has_ducts`, `has_banners`, `has_windows`, `num_wall_banners`, `has_light_housings`, `has_line_judges`, `seating_type`, `spectator_count`, `clutter_items`)

**Context:** New module for wall banners, exit signs, windows, scoreboard upgrade, ceiling trusses, ventilation ducts, hanging banners, and light housings. Runs after lighting so it can read light positions for housing placement.

- [ ] **Step 1: Create module skeleton with wall details**

Create `src/tools/blender_venue_details.py`:

```python
"""Venue details: wall banners, ceiling trusses, light housings, seating, clutter."""

import bpy
import math
import random

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1


def _clear_venue_details():
    """Remove existing venue details collection."""
    if "VenueDetails" in bpy.data.collections:
        col = bpy.data.collections["VenueDetails"]
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)


def _make_mat(name, color, roughness=0.8, metallic=0.0, emission=0.0):
    """Create a material with optional emission."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission > 0:
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def _link_to_collection(obj, collection):
    """Move object from default collection to target."""
    for col_ in list(obj.users_collection):
        col_.objects.unlink(obj)
    collection.objects.link(obj)


def _build_wall_banners(collection, bounds):
    """Sponsor banners on 1-3 walls."""
    cx, cy = bounds["cx"], bounds["cy"]
    d, w, h = bounds["d"], bounds["w"], bounds["h"]
    half_d, half_w = d / 2, w / 2

    wall_positions = [
        ("back", cx - half_d + 0.05, cy, 0, 0, 0),
        ("front", cx + half_d - 0.05, cy, 0, 0, math.pi),
        ("left", cx, cy - half_w + 0.05, 0, 0, math.pi / 2),
        ("right", cx, cy + half_w - 0.05, 0, 0, -math.pi / 2),
    ]

    num_walls = random.randint(1, 3)
    chosen_walls = random.sample(wall_positions, k=num_walls)
    count = 0

    for wall_name, wx, wy, rx, ry, rz in chosen_walls:
        num_banners = random.randint(1, 3)
        for i in range(num_banners):
            banner_w = random.uniform(1.0, 3.0)
            banner_h = random.uniform(0.5, 1.5)
            banner_y_off = random.uniform(-w * 0.3, w * 0.3)
            banner_z = random.uniform(1.5, 3.0)

            bpy.ops.mesh.primitive_plane_add(size=1)
            obj = bpy.context.active_object
            obj.name = f"WallBanner_{count}"
            obj.scale = (banner_w, banner_h, 1)
            bpy.ops.object.transform_apply(scale=True)

            if "back" in wall_name or "front" in wall_name:
                obj.location = (wx, wy + banner_y_off, banner_z)
                obj.rotation_euler = (math.pi / 2, 0, rz)
            else:
                obj.location = (wx + banner_y_off, wy, banner_z)
                obj.rotation_euler = (math.pi / 2, 0, rz)

            # Procedural banner: wave + colorramp
            mat = bpy.data.materials.new(f"BannerMat_{count}")
            mat.use_nodes = True
            tree = mat.node_tree
            nodes = tree.nodes
            links_ = tree.links
            bsdf = nodes["Principled BSDF"]

            wave = nodes.new("ShaderNodeTexWave")
            wave.inputs["Scale"].default_value = random.uniform(3, 8)

            ramp = nodes.new("ShaderNodeValToRGB")
            c1 = (random.uniform(0.2, 0.9), random.uniform(0.1, 0.8),
                   random.uniform(0.1, 0.8), 1.0)
            c2 = (random.uniform(0.2, 0.9), random.uniform(0.1, 0.8),
                   random.uniform(0.1, 0.8), 1.0)
            ramp.color_ramp.elements[0].color = c1
            ramp.color_ramp.elements[1].color = c2

            links_.new(wave.outputs["Fac"], ramp.inputs["Fac"])
            links_.new(ramp.outputs[0], bsdf.inputs["Base Color"])

            obj.data.materials.append(mat)
            _link_to_collection(obj, collection)
            count += 1

    return count


def _build_exit_signs(collection, bounds):
    """Small emissive exit sign rectangles near ceiling."""
    cx, cy = bounds["cx"], bounds["cy"]
    d, w, h = bounds["d"], bounds["w"], bounds["h"]
    half_d = d / 2

    num = random.randint(1, 2)
    for i in range(num):
        color = random.choice([
            (0.1, 0.8, 0.1, 1.0),   # green
            (0.8, 0.1, 0.1, 1.0),   # red
        ])
        mat = _make_mat(f"ExitSignMat_{i}", color, emission=random.uniform(2, 5))

        wall_x = cx + random.choice([-half_d + 0.1, half_d - 0.1])
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            location=(wall_x, cy + random.uniform(-w * 0.2, w * 0.2), h - 0.5),
        )
        obj = bpy.context.active_object
        obj.name = f"ExitSign_{i}"
        obj.scale = (0.3, 0.15, 1)
        obj.rotation_euler = (math.pi / 2, 0, 0)
        bpy.ops.object.transform_apply(scale=True)
        obj.data.materials.append(mat)
        _link_to_collection(obj, collection)


def _build_windows(collection, bounds):
    """Light rectangles on upper walls simulating windows (~40%)."""
    if random.random() > 0.4:
        return False

    cx, cy = bounds["cx"], bounds["cy"]
    d, w, h = bounds["d"], bounds["w"], bounds["h"]
    half_d, half_w = d / 2, w / 2

    mat = _make_mat("WindowMat", (0.85, 0.88, 0.92, 1.0),
                    roughness=0.3, emission=random.uniform(0.5, 1.5))

    num_walls = random.randint(1, 2)
    walls = random.sample([
        (cx - half_d + 0.06, cy, True),
        (cx + half_d - 0.06, cy, True),
        (cx, cy - half_w + 0.06, False),
        (cx, cy + half_w - 0.06, False),
    ], k=num_walls)

    for i, (wx, wy, is_xwall) in enumerate(walls):
        win_w = random.uniform(2, 4)
        win_h = random.uniform(1, 2)
        win_z = h * 0.7
        bpy.ops.mesh.primitive_plane_add(size=1, location=(wx, wy, win_z))
        obj = bpy.context.active_object
        obj.name = f"Window_{i}"
        obj.scale = (win_w if is_xwall else 0.01, 0.01 if is_xwall else win_w, win_h)
        obj.rotation_euler = (math.pi / 2, 0, 0) if is_xwall else (0, 0, math.pi / 2)
        bpy.ops.object.transform_apply(scale=True)
        obj.data.materials.append(mat)
        _link_to_collection(obj, collection)

    return True
```

- [ ] **Step 2: Add ceiling details (trusses, ducts, hanging banners)**

Append to `blender_venue_details.py`:

```python
def _build_trusses(collection, bounds):
    """Exposed ceiling trusses (~50% chance)."""
    if random.random() > 0.5:
        return False

    cx, cy = bounds["cx"], bounds["cy"]
    d, w, h = bounds["d"], bounds["w"], bounds["h"]

    mat = _make_mat("TrussMat", (0.5, 0.5, 0.55, 1.0), roughness=0.4, metallic=0.7)

    num_beams = random.randint(3, 6)
    spacing = d / (num_beams + 1)

    for i in range(num_beams):
        beam_x = cx - d / 2 + spacing * (i + 1)
        # Main beam
        bpy.ops.mesh.primitive_cube_add(size=1, location=(beam_x, cy, h - 0.5))
        obj = bpy.context.active_object
        obj.name = f"Truss_Beam_{i}"
        obj.scale = (0.15, w * 0.9, 0.15)
        bpy.ops.object.transform_apply(scale=True)
        obj.data.materials.append(mat)
        _link_to_collection(obj, collection)

        # Cross braces (connect to next beam if not last)
        if i < num_beams - 1:
            next_x = cx - d / 2 + spacing * (i + 2)
            mid_x = (beam_x + next_x) / 2
            bpy.ops.mesh.primitive_cube_add(
                size=1, location=(mid_x, cy, h - 0.5),
            )
            brace = bpy.context.active_object
            brace.name = f"Truss_Brace_{i}"
            brace_len = ((next_x - beam_x) ** 2 + (w * 0.3) ** 2) ** 0.5
            brace.scale = (brace_len, 0.05, 0.05)
            brace.rotation_euler = (0, 0, math.atan2(w * 0.3, next_x - beam_x))
            bpy.ops.object.transform_apply(scale=True, rotation=True)
            brace.data.materials.append(mat)
            _link_to_collection(brace, collection)

    return True


def _build_ducts(collection, bounds):
    """Ventilation ducts (~40% chance)."""
    if random.random() > 0.4:
        return False

    cx, cy = bounds["cx"], bounds["cy"]
    d, w, h = bounds["d"], bounds["w"], bounds["h"]

    mat = _make_mat("DuctMat", (0.6, 0.6, 0.65, 1.0), roughness=0.35, metallic=0.6)

    num = random.randint(1, 3)
    for i in range(num):
        duct_len = random.uniform(3, 8)
        offset_y = random.uniform(-w * 0.3, w * 0.3)
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(cx + random.uniform(-d * 0.2, d * 0.2),
                      cy + offset_y, h - 0.7),
        )
        obj = bpy.context.active_object
        obj.name = f"Duct_{i}"
        obj.scale = (duct_len, 0.4, 0.4)
        bpy.ops.object.transform_apply(scale=True)
        obj.data.materials.append(mat)
        _link_to_collection(obj, collection)

    return True


def _build_hanging_banners(collection, bounds):
    """Hanging banners from ceiling (~30% chance)."""
    if random.random() > 0.3:
        return False

    cx, cy = bounds["cx"], bounds["cy"]
    d, w, h = bounds["d"], bounds["w"], bounds["h"]

    num = random.randint(1, 3)
    for i in range(num):
        banner_w = random.uniform(1, 2)
        banner_h = random.uniform(2, 4)
        bx = cx + random.uniform(-d * 0.4, d * 0.4)
        by = cy + random.choice([-1, 1]) * (w * 0.35 + random.uniform(0, w * 0.1))

        # Cable
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.01, depth=1.5,
            location=(bx, by, h - 0.75),
        )
        cable = bpy.context.active_object
        cable.name = f"HangBanner_Cable_{i}"
        _link_to_collection(cable, collection)

        # Banner
        mat = bpy.data.materials.new(f"HangBannerMat_{i}")
        mat.use_nodes = True
        tree = mat.node_tree
        bsdf = tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (
            random.uniform(0.3, 0.9), random.uniform(0.1, 0.8),
            random.uniform(0.1, 0.8), 1.0,
        )

        bpy.ops.mesh.primitive_plane_add(
            size=1, location=(bx, by, h - 1.5 - banner_h / 2),
        )
        obj = bpy.context.active_object
        obj.name = f"HangBanner_{i}"
        obj.scale = (banner_w, 0.01, banner_h)
        bpy.ops.object.transform_apply(scale=True)
        obj.data.materials.append(mat)
        _link_to_collection(obj, collection)

    return True
```

- [ ] **Step 3: Add light housings**

Append to `blender_venue_details.py`:

```python
def _build_light_housings(collection):
    """Rectangular housings at each area light position."""
    if "Lighting" not in bpy.data.collections:
        return False

    lighting_col = bpy.data.collections["Lighting"]
    mat = _make_mat("LightHousingMat", (0.85, 0.85, 0.88, 1.0),
                    roughness=0.5, emission=0.3)

    count = 0
    for obj in lighting_col.objects:
        if obj.type != 'LIGHT':
            continue
        lx, ly, lz = obj.location

        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(lx, ly, lz + 0.15),
        )
        housing = bpy.context.active_object
        housing.name = f"LightHousing_{count}"
        housing.scale = (1.5, 0.8, 0.3)
        bpy.ops.object.transform_apply(scale=True)
        housing.data.materials.append(mat)
        _link_to_collection(housing, collection)
        count += 1

    return count > 0
```

- [ ] **Step 4: Add `build_venue_details` main function (walls + ceiling only for now)**

Append to `blender_venue_details.py`:

```python
def build_venue_details(config=None):
    """Build venue wall/ceiling details, seating, and clutter.

    Args:
        config: dict with "venue_bounds" key containing
                {cx, cy, d, w, h}

    Returns:
        dict with detail flags for metadata
    """
    cfg = config or {}
    bounds = cfg.get("venue_bounds", {
        "cx": COURT_LENGTH / 2, "cy": COURT_WIDTH / 2,
        "d": COURT_LENGTH + 16, "w": COURT_WIDTH + 16, "h": 22,
    })

    _clear_venue_details()

    col = bpy.data.collections.new("VenueDetails")
    bpy.context.scene.collection.children.link(col)

    # Wall details
    num_wall_banners = _build_wall_banners(col, bounds)
    _build_exit_signs(col, bounds)
    has_windows = _build_windows(col, bounds)

    # Ceiling details
    has_trusses = _build_trusses(col, bounds)
    has_ducts = _build_ducts(col, bounds)
    has_hanging_banners = _build_hanging_banners(col, bounds)
    has_light_housings = _build_light_housings(col)

    return {
        "num_wall_banners": num_wall_banners,
        "has_windows": has_windows or False,
        "has_trusses": has_trusses or False,
        "has_ducts": has_ducts or False,
        "has_banners": has_hanging_banners or False,
        "has_light_housings": has_light_housings or False,
    }
```

- [ ] **Step 5: Commit**

```bash
git add src/tools/blender_venue_details.py
git commit -m "feat(blender): add venue details module with wall banners, ceiling trusses, light housings"
```

---

### Task 7: Venue Details — Seating, Clutter, and Line Judges

**Files:**
- Modify: `src/tools/blender_venue_details.py` (add seating, clutter, line judge functions; update `build_venue_details`)
- Modify: `src/tools/blender_environment.py` (remove old `_build_spectators`, update `build_environment`)

**Interfaces:**
- Consumes: `bounds` dict in `build_venue_details`
- Produces: `build_venue_details` return dict gains `seating_type`, `spectator_count`, `has_line_judges`, `clutter_items`

**Context:** Tiered bleacher seating replaces the current flat bench rows in `blender_environment.py`. Seated spectator figures, courtside furniture, equipment bags, and courtside clutter items are added. Line judge chairs correlate with competition lighting. The old `_build_spectators` in `blender_environment.py` is removed.

- [ ] **Step 1: Add bleacher seating to `blender_venue_details.py`**

Add before `build_venue_details`:

```python
def _build_bleacher_seating(collection, bounds):
    """Tiered bleacher seating behind baselines, optionally along sideline."""
    seat_colors = [
        (0.15, 0.2, 0.6, 1.0),    # blue
        (0.6, 0.1, 0.08, 1.0),    # red
        (0.4, 0.4, 0.4, 1.0),     # grey
        (0.7, 0.35, 0.05, 1.0),   # orange
        (0.1, 0.45, 0.15, 1.0),   # green
    ]
    seat_color = random.choice(seat_colors)
    mat = _make_mat("BleacherMat", seat_color, roughness=0.75)

    num_rows = random.randint(1, 5)
    has_seat_backs = random.random() < 0.4
    seating_type = "bleacher_with_seats" if has_seat_backs else "bleacher_bench"

    placements = []
    # Behind baselines (always at least one side)
    sides = random.sample(["back", "front"], k=random.randint(1, 2))
    for side in sides:
        base_x = -4.0 if side == "back" else COURT_LENGTH + 4.0
        direction = -1 if side == "back" else 1
        placements.append((base_x, direction))

    # Optional sideline (~20% chance)
    if random.random() < 0.2:
        side_y = random.choice([-4.0, COURT_WIDTH + 4.0])
        placements.append((side_y, None))

    spectator_count = 0

    for placement in placements:
        if placement[1] is not None:
            base_x, direction = placement
            for row in range(num_rows):
                row_x = base_x + direction * row * 0.6
                row_z = 0.2 + row * 0.4
                bench_width = min(bounds["w"] * 0.5, 8.0)

                bpy.ops.mesh.primitive_cube_add(
                    size=1,
                    location=(row_x, COURT_WIDTH / 2, row_z),
                )
                bench = bpy.context.active_object
                bench.name = f"Bleacher_{base_x:.0f}_{row}"
                bench.scale = (0.5, bench_width, 0.15)
                bpy.ops.object.transform_apply(scale=True)
                bench.data.materials.append(mat)
                _link_to_collection(bench, collection)

                if has_seat_backs:
                    for s in range(int(bench_width / 0.5)):
                        seat_y = COURT_WIDTH / 2 - bench_width / 2 + s * 0.5 + 0.25
                        bpy.ops.mesh.primitive_cube_add(
                            size=1,
                            location=(row_x - direction * 0.2, seat_y, row_z + 0.25),
                        )
                        back = bpy.context.active_object
                        back.name = f"SeatBack_{base_x:.0f}_{row}_{s}"
                        back.scale = (0.05, 0.3, 0.3)
                        bpy.ops.object.transform_apply(scale=True)
                        back.data.materials.append(mat)
                        _link_to_collection(back, collection)

                # Seated spectators (0-30% occupancy)
                occupancy = random.uniform(0, 0.3)
                seat_count = int(bench_width / 0.5)
                for s in range(seat_count):
                    if random.random() > occupancy:
                        continue
                    spectator_count += 1
                    sy = COURT_WIDTH / 2 - bench_width / 2 + s * 0.5 + 0.25
                    # Simple torso + head
                    torso_color = random.choice([
                        (0.3, 0.3, 0.5, 1.0), (0.5, 0.2, 0.2, 1.0),
                        (0.8, 0.8, 0.8, 1.0), (0.2, 0.4, 0.2, 1.0),
                    ])
                    sp_mat = _make_mat(f"SpectatorMat_{spectator_count}", torso_color)
                    bpy.ops.mesh.primitive_cylinder_add(
                        radius=0.12, depth=0.4,
                        location=(row_x, sy, row_z + 0.45),
                    )
                    torso = bpy.context.active_object
                    torso.name = f"Spectator_{spectator_count}_Torso"
                    torso.data.materials.append(sp_mat)
                    _link_to_collection(torso, collection)

                    skin = _make_mat(f"SpectatorSkin_{spectator_count}",
                                     (0.7, 0.55, 0.4, 1.0), roughness=0.6)
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        radius=0.08,
                        location=(row_x, sy, row_z + 0.75),
                    )
                    head = bpy.context.active_object
                    head.name = f"Spectator_{spectator_count}_Head"
                    head.data.materials.append(skin)
                    _link_to_collection(head, collection)

    return seating_type, spectator_count
```

- [ ] **Step 2: Add courtside furniture and clutter**

Add before `build_venue_details`:

```python
def _build_courtside_furniture(collection):
    """Player chairs, drink station, equipment bags."""
    items = []

    # Player chairs (~60% chance)
    if random.random() < 0.6:
        num_chairs = random.randint(1, 2)
        chair_mat = _make_mat("ChairMat", (0.3, 0.3, 0.35, 1.0),
                              roughness=0.5, metallic=0.4)
        for i in range(num_chairs):
            cx = COURT_LENGTH / 2 + random.uniform(-1, 1)
            cy = random.choice([-2.0, COURT_WIDTH + 2.0])

            # Seat
            bpy.ops.mesh.primitive_cube_add(
                size=1, location=(cx, cy, 0.45),
            )
            seat = bpy.context.active_object
            seat.name = f"PlayerChair_{i}_Seat"
            seat.scale = (0.4, 0.4, 0.05)
            bpy.ops.object.transform_apply(scale=True)
            seat.data.materials.append(chair_mat)
            _link_to_collection(seat, collection)

            # Back
            bpy.ops.mesh.primitive_cube_add(
                size=1, location=(cx, cy + (0.2 if cy < 0 else -0.2), 0.65),
            )
            back = bpy.context.active_object
            back.name = f"PlayerChair_{i}_Back"
            back.scale = (0.4, 0.05, 0.3)
            bpy.ops.object.transform_apply(scale=True)
            back.data.materials.append(chair_mat)
            _link_to_collection(back, collection)

            # 4 legs
            for li, (lx_off, ly_off) in enumerate([(-0.15, -0.15), (0.15, -0.15),
                                                     (-0.15, 0.15), (0.15, 0.15)]):
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=0.015, depth=0.45,
                    location=(cx + lx_off, cy + ly_off, 0.225),
                )
                leg = bpy.context.active_object
                leg.name = f"PlayerChair_{i}_Leg_{li}"
                leg.data.materials.append(chair_mat)
                _link_to_collection(leg, collection)

            # Towel (~50%)
            if random.random() < 0.5:
                towel_color = (random.uniform(0.3, 0.9), random.uniform(0.3, 0.9),
                               random.uniform(0.3, 0.9), 1.0)
                towel_mat = _make_mat(f"TowelMat_{i}", towel_color)
                bpy.ops.mesh.primitive_plane_add(
                    size=1,
                    location=(cx, cy + (0.21 if cy < 0 else -0.21), 0.7),
                )
                towel = bpy.context.active_object
                towel.name = f"Towel_{i}"
                towel.scale = (0.3, 0.02, 0.4)
                bpy.ops.object.transform_apply(scale=True)
                towel.data.materials.append(towel_mat)
                _link_to_collection(towel, collection)

        # Drink station (~40%)
        if random.random() < 0.4:
            table_mat = _make_mat("DrinkTableMat", (0.35, 0.35, 0.38, 1.0))
            tx = COURT_LENGTH / 2 + random.uniform(-0.5, 0.5)
            ty = random.choice([-2.5, COURT_WIDTH + 2.5])
            bpy.ops.mesh.primitive_cube_add(
                size=1, location=(tx, ty, 0.35),
            )
            table = bpy.context.active_object
            table.name = "DrinkTable"
            table.scale = (0.6, 0.4, 0.7)
            bpy.ops.object.transform_apply(scale=True)
            table.data.materials.append(table_mat)
            _link_to_collection(table, collection)

            bottle_mat = _make_mat("BottleMat", (0.3, 0.5, 0.8, 1.0), roughness=0.3)
            for bi in range(random.randint(1, 3)):
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=0.03, depth=0.2,
                    location=(tx + random.uniform(-0.15, 0.15), ty, 0.8),
                )
                bottle = bpy.context.active_object
                bottle.name = f"Bottle_{bi}"
                bottle.data.materials.append(bottle_mat)
                _link_to_collection(bottle, collection)

    # Equipment bags (~50%)
    if random.random() < 0.5:
        bag_colors = [
            (0.05, 0.05, 0.2, 1.0),
            (0.08, 0.08, 0.08, 1.0),
            (0.25, 0.25, 0.28, 1.0),
        ]
        num_bags = random.randint(1, 3)
        for i in range(num_bags):
            bag_mat = _make_mat(f"BagMat_{i}", random.choice(bag_colors))
            bx = random.uniform(-3, COURT_LENGTH + 3)
            by = random.choice([-3.5, COURT_WIDTH + 3.5])
            bpy.ops.mesh.primitive_cube_add(
                size=1, location=(bx, by, 0.15),
            )
            bag = bpy.context.active_object
            bag.name = f"EquipBag_{i}"
            bag.scale = (0.5, 0.3, 0.3)
            bpy.ops.object.transform_apply(scale=True)
            bag.data.materials.append(bag_mat)
            _link_to_collection(bag, collection)


def _build_clutter(collection, bounds):
    """Courtside clutter items with independent spawn chances."""
    items = []

    # Mop bucket (15%)
    if random.random() < 0.15:
        mat = _make_mat("MopBucketMat", (0.4, 0.4, 0.45, 1.0))
        bx = bounds["cx"] + random.choice([-1, 1]) * (bounds["d"] / 2 - 1)
        by = bounds["cy"] + random.uniform(-bounds["w"] * 0.3, bounds["w"] * 0.3)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.15, depth=0.3, location=(bx, by, 0.15),
        )
        obj = bpy.context.active_object
        obj.name = "MopBucket"
        obj.data.materials.append(mat)
        _link_to_collection(obj, collection)
        # Handle
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.01, depth=0.8, location=(bx, by, 0.7),
        )
        handle = bpy.context.active_object
        handle.name = "MopHandle"
        handle.rotation_euler = (math.radians(15), 0, 0)
        _link_to_collection(handle, collection)
        items.append("mop_bucket")

    # Rolled-up net (10%)
    if random.random() < 0.10:
        mat = _make_mat("RolledNetMat", (0.2, 0.2, 0.22, 1.0))
        bx = bounds["cx"] + random.choice([-1, 1]) * (bounds["d"] / 2 - 1.5)
        by = bounds["cy"] + random.uniform(-2, 2)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.15, depth=2.0,
            location=(bx, by, 0.15),
            rotation=(0, math.pi / 2, random.uniform(0, math.pi)),
        )
        obj = bpy.context.active_object
        obj.name = "RolledNet"
        obj.data.materials.append(mat)
        _link_to_collection(obj, collection)
        items.append("rolled_net")

    # Stacked chairs (15%)
    if random.random() < 0.15:
        mat = _make_mat("StackedChairMat", (0.35, 0.35, 0.38, 1.0), metallic=0.3)
        bx = bounds["cx"] + random.choice([-1, 1]) * (bounds["d"] / 2 - 1)
        by = bounds["cy"] + random.choice([-1, 1]) * (bounds["w"] / 2 - 1)
        for ci in range(random.randint(2, 3)):
            bpy.ops.mesh.primitive_cube_add(
                size=0.4,
                location=(bx, by, 0.2 + ci * 0.35),
            )
            obj = bpy.context.active_object
            obj.name = f"StackedChair_{ci}"
            obj.data.materials.append(mat)
            _link_to_collection(obj, collection)
        items.append("stacked_chairs")

    # Training cones (20%)
    if random.random() < 0.20:
        cone_mat = _make_mat("ConeMat", (0.9, 0.5, 0.05, 1.0))
        num_cones = random.randint(1, 4)
        for ci in range(num_cones):
            cx = random.uniform(max(-3, COURT_LENGTH / 2 - 5),
                                min(COURT_LENGTH + 3, COURT_LENGTH / 2 + 5))
            cy = random.choice([-1.5, COURT_WIDTH + 1.5]) + random.uniform(-0.5, 0.5)
            bpy.ops.mesh.primitive_cone_add(
                radius1=0.1, depth=0.2, location=(cx, cy, 0.1),
            )
            obj = bpy.context.active_object
            obj.name = f"TrainingCone_{ci}"
            obj.data.materials.append(cone_mat)
            _link_to_collection(obj, collection)
        items.append("training_cones")

    return items


def _build_line_judge_chairs(collection, lighting_preset=None):
    """Line judge chairs at court corners (~25% chance, higher with competition lighting)."""
    chance = 0.5 if lighting_preset == "competition" else 0.25
    if random.random() > chance:
        return False

    mat_frame = _make_mat("LineJudgeChairMat", (0.4, 0.4, 0.45, 1.0),
                          roughness=0.4, metallic=0.5)
    mat_seat = _make_mat("LineJudgeSeatMat", (0.15, 0.15, 0.18, 1.0))

    corners = [
        (-1.5, -1.5), (-1.5, COURT_WIDTH + 1.5),
        (COURT_LENGTH + 1.5, -1.5), (COURT_LENGTH + 1.5, COURT_WIDTH + 1.5),
    ]
    num = random.randint(2, 4)
    chosen = random.sample(corners, k=num)

    for i, (jx, jy) in enumerate(chosen):
        # Chair seat
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(jx, jy, 0.4),
        )
        seat = bpy.context.active_object
        seat.name = f"LineJudgeChair_{i}_Seat"
        seat.scale = (0.4, 0.4, 0.05)
        bpy.ops.object.transform_apply(scale=True)
        seat.data.materials.append(mat_seat)
        _link_to_collection(seat, collection)

        # Back
        bpy.ops.mesh.primitive_cube_add(
            size=1, location=(jx, jy, 0.55),
        )
        back = bpy.context.active_object
        back.name = f"LineJudgeChair_{i}_Back"
        back.scale = (0.4, 0.02, 0.3)
        bpy.ops.object.transform_apply(scale=True)
        back.data.materials.append(mat_frame)
        _link_to_collection(back, collection)

        # 4 legs
        for li, (lx_off, ly_off) in enumerate([(-0.15, -0.15), (0.15, -0.15),
                                                 (-0.15, 0.15), (0.15, 0.15)]):
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.012, depth=0.4,
                location=(jx + lx_off, jy + ly_off, 0.2),
            )
            leg = bpy.context.active_object
            leg.name = f"LineJudgeChair_{i}_Leg_{li}"
            leg.data.materials.append(mat_frame)
            _link_to_collection(leg, collection)

        # Optional seated figure (~50%)
        if random.random() < 0.5:
            sp_color = random.choice([
                (0.1, 0.1, 0.15, 1.0), (0.2, 0.2, 0.25, 1.0),
            ])
            sp_mat = _make_mat(f"LineJudgeMat_{i}", sp_color)
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.12, depth=0.4,
                location=(jx, jy, 0.65),
            )
            torso = bpy.context.active_object
            torso.name = f"LineJudge_{i}_Torso"
            torso.data.materials.append(sp_mat)
            _link_to_collection(torso, collection)

            skin = _make_mat(f"LineJudgeSkin_{i}", (0.65, 0.48, 0.35, 1.0))
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=0.08, location=(jx, jy, 0.95),
            )
            head = bpy.context.active_object
            head.name = f"LineJudge_{i}_Head"
            head.data.materials.append(skin)
            _link_to_collection(head, collection)

    return True
```

- [ ] **Step 3: Update `build_venue_details` to call new functions**

Replace the `build_venue_details` function body to include seating, clutter, and line judges:

```python
def build_venue_details(config=None):
    """Build venue wall/ceiling details, seating, and clutter.

    Args:
        config: dict with "venue_bounds" (cx, cy, d, w, h)
                and optionally "lighting_preset" (str)

    Returns:
        dict with detail flags for metadata
    """
    cfg = config or {}
    bounds = cfg.get("venue_bounds", {
        "cx": COURT_LENGTH / 2, "cy": COURT_WIDTH / 2,
        "d": COURT_LENGTH + 16, "w": COURT_WIDTH + 16, "h": 22,
    })
    lighting_preset = cfg.get("lighting_preset")

    _clear_venue_details()

    col = bpy.data.collections.new("VenueDetails")
    bpy.context.scene.collection.children.link(col)

    # Wall details
    num_wall_banners = _build_wall_banners(col, bounds)
    _build_exit_signs(col, bounds)
    has_windows = _build_windows(col, bounds)

    # Ceiling details
    has_trusses = _build_trusses(col, bounds)
    has_ducts = _build_ducts(col, bounds)
    has_hanging_banners = _build_hanging_banners(col, bounds)
    has_light_housings = _build_light_housings(col)

    # Seating
    seating_type, spectator_count = _build_bleacher_seating(col, bounds)

    # Courtside furniture
    _build_courtside_furniture(col)

    # Line judge chairs
    has_line_judges = _build_line_judge_chairs(col, lighting_preset)

    # Clutter
    clutter_items = _build_clutter(col, bounds)

    return {
        "num_wall_banners": num_wall_banners,
        "has_windows": has_windows or False,
        "has_trusses": has_trusses or False,
        "has_ducts": has_ducts or False,
        "has_banners": has_hanging_banners or False,
        "has_light_housings": has_light_housings or False,
        "seating_type": seating_type,
        "spectator_count": spectator_count,
        "has_line_judges": has_line_judges or False,
        "clutter_items": clutter_items,
    }
```

- [ ] **Step 4: Remove old `_build_spectators` and upgrade scoreboard in `blender_environment.py`**

In `src/tools/blender_environment.py`:

1. Remove the `_build_spectators` function (lines 188-215)
2. In `build_environment()`, remove the spectator creation code (lines 333-335):

```python
    # Remove these lines:
    has_spectators = random.random() < cfg["spectator_chance"]
    if has_spectators:
        _build_spectators(col, venue_w)
```

3. Remove `"has_spectators": has_spectators` from the return dict
4. Remove `"spectator_chance"` from `DEFAULT_CONFIG`

The spectator functionality is now handled by `build_venue_details` via bleacher seating.

5. Upgrade `_build_scoreboard` to add a frame border, emissive face, and stand legs. Replace the function (lines 245-269) with:

```python
def _build_scoreboard(collection):
    """Scoreboard with frame border, emissive face, and stand legs."""
    side = random.choice([-3.0, COURT_WIDTH + 3.0])
    x = COURT_LENGTH / 2 + random.uniform(-2, 2)

    # Face (emissive)
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
    bsdf.inputs["Emission Color"].default_value = (0.15, 0.2, 0.3, 1.0)
    bsdf.inputs["Emission Strength"].default_value = random.uniform(1, 3)
    board.data.materials.append(mat)

    for col_ in list(board.users_collection):
        col_.objects.unlink(board)
    collection.objects.link(board)

    # Frame border (4 thin cubes)
    frame_mat = bpy.data.materials.new("ScoreboardFrameMat")
    frame_mat.use_nodes = True
    bsdf_f = frame_mat.node_tree.nodes["Principled BSDF"]
    bsdf_f.inputs["Base Color"].default_value = (0.3, 0.3, 0.35, 1.0)
    bsdf_f.inputs["Metallic"].default_value = 0.5

    frame_parts = [
        (f"ScoreboardFrame_T", (x, side, 2.5), (1.55, 0.03, 0.03)),
        (f"ScoreboardFrame_B", (x, side, 1.5), (1.55, 0.03, 0.03)),
        (f"ScoreboardFrame_L", (x - 0.75, side, 2.0), (0.03, 0.03, 1.03)),
        (f"ScoreboardFrame_R", (x + 0.75, side, 2.0), (0.03, 0.03, 1.03)),
    ]
    for name, loc, scale in frame_parts:
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = scale
        bpy.ops.object.transform_apply(scale=True)
        obj.data.materials.append(frame_mat)
        for col_ in list(obj.users_collection):
            col_.objects.unlink(obj)
        collection.objects.link(obj)

    # Two stand legs
    for i, x_off in enumerate([-0.5, 0.5]):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.025, depth=1.5,
            location=(x + x_off, side, 0.75),
        )
        leg = bpy.context.active_object
        leg.name = f"ScoreboardLeg_{i}"
        leg.data.materials.append(frame_mat)
        for col_ in list(leg.users_collection):
            col_.objects.unlink(leg)
        collection.objects.link(leg)

    return board
```

- [ ] **Step 5: Run Blender smoke test**

Run: `"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python src/tools/blender_render.py -- --count 1 --engine BLENDER_EEVEE --samples 16 --seed 42`

Expected: No errors. Note that `build_venue_details` isn't called in the render loop yet (Task 8), but the old spectators are removed and the environment module works without them.

- [ ] **Step 6: Commit**

```bash
git add src/tools/blender_venue_details.py src/tools/blender_environment.py
git commit -m "feat(blender): add bleacher seating, courtside furniture, clutter, and line judges"
```

---

### Task 8: Render Loop Integration

**Files:**
- Modify: `src/tools/blender_render.py:92-166` (add `build_venue_details` call, update metadata export)

**Interfaces:**
- Consumes: `build_venue_details(config)` from `blender_venue_details.py`, updated `build_environment()` return with `venue_bounds` and `floor_type`
- Produces: Updated metadata JSON with all new fields

**Context:** Wire `build_venue_details()` into the render loop between `setup_lighting` and `add_occluders`. Pass venue bounds from environment and lighting preset. Merge venue details metadata into the environment section of the output JSON.

- [ ] **Step 1: Add import and update render loop**

In `src/tools/blender_render.py`, add the import (after line 96):

```python
    from src.tools.blender_venue_details import build_venue_details
```

In the render loop (inside `for i in range(COUNT):`), after `light_meta = setup_lighting(preset)` (line 133) and before `occ_meta = add_occluders()` (line 135), add:

```python
        # 5. Build venue details (after lighting, needs light positions)
        details_meta = build_venue_details({
            "venue_bounds": env_meta.get("venue_bounds", {}),
            "lighting_preset": preset,
        })
```

- [ ] **Step 2: Update metadata export to merge venue details**

Replace the metadata assembly (lines 144-155) with:

```python
        # 7. Export metadata
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
```

- [ ] **Step 3: Update the print line to include detail count**

Replace the print line (line 163) with:

```python
        details_count = sum(1 for k in details_meta if details_meta[k])
        print(f"  [{i + 1}/{COUNT}] {img_name} | {strategy} | {preset} | "
              f"{vis}/30 kp | {details_count} details | {elapsed:.1f}s")
```

- [ ] **Step 4: Run full Blender smoke test (3 images)**

Run: `"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python src/tools/blender_render.py -- --count 3 --engine BLENDER_EEVEE --samples 16 --seed 42`

Expected: 3 images render without errors. Check metadata JSONs:
- `environment` section contains `floor_type`, `has_floor_markings`, `num_wall_banners`, `has_trusses`, `has_ducts`, `has_banners`, `has_windows`, `has_light_housings`, `seating_type`, `spectator_count`, `has_line_judges`, `clutter_items`
- `occluders` entries with `type: "player"` have `pose`, `has_racket`, `facing_angle` fields

- [ ] **Step 5: Run the converter to verify backward compatibility**

Run: `python src/tools/blender_to_cvn.py --min-visible 4`

Expected: Converter runs successfully. The new metadata fields are in the raw metadata only — the CVN format output is unchanged (image_path, image_size, keypoints, visibility, bounding_box).

- [ ] **Step 6: Commit**

```bash
git add src/tools/blender_render.py
git commit -m "feat(blender): integrate venue details into render loop with enriched metadata"
```

---

### Task 9: Blender Integration Smoke Test Script

**Files:**
- Create: `tests/test_blender_smoke.py` (script that runs inside Blender to validate the full pipeline)

**Interfaces:**
- Consumes: all Blender modules
- Produces: pass/fail output and validation of metadata structure

**Context:** Since the Blender modules require `bpy` and can't run under normal pytest, we create a dedicated smoke test script that runs inside Blender and validates the complete pipeline: scene builds without errors, metadata has the expected fields, all collections exist.

- [ ] **Step 1: Create the Blender smoke test**

Create `tests/test_blender_smoke.py`:

```python
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
```

- [ ] **Step 2: Run the smoke test**

Run: `"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python tests/test_blender_smoke.py`

Expected: All checks pass, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add tests/test_blender_smoke.py
git commit -m "test(blender): add integration smoke test for full pipeline with realism improvements"
```

"""Venue details: wall banners, ceiling trusses, light housings, seating, clutter."""

import bpy
import math
import random

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1

# Materials created with fixed (non-unique) names that must be cleaned up
# alongside the objects, so re-runs don't accumulate orphan datablocks.
_FIXED_MATERIAL_NAMES = [
    "TrussMat",
    "DuctMat",
    "WindowMat",
    "LightHousingMat",
]
_FIXED_MATERIAL_PREFIXES = [
    "ExitSignMat_",
    "BannerMat_",
    "HangBannerMat_",
]


def _clear_venue_details():
    """Remove existing venue details collection and its fixed-name materials."""
    if "VenueDetails" in bpy.data.collections:
        col = bpy.data.collections["VenueDetails"]
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)

    for name in _FIXED_MATERIAL_NAMES:
        mat = bpy.data.materials.get(name)
        if mat is not None:
            bpy.data.materials.remove(mat)

    for mat in list(bpy.data.materials):
        for prefix in _FIXED_MATERIAL_PREFIXES:
            if mat.name.startswith(prefix):
                bpy.data.materials.remove(mat)
                break


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
        if is_xwall:
            obj.scale = (win_w, 0.01, win_h)
            obj.rotation_euler = (math.pi / 2, 0, 0)
        else:
            obj.scale = (0.01, win_w, win_h)
            obj.rotation_euler = (0, math.pi / 2, 0)
        bpy.ops.object.transform_apply(scale=True)
        obj.data.materials.append(mat)
        _link_to_collection(obj, collection)

    return True


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
        "has_line_judges": False,
        "seating_type": None,
        "spectator_count": 0,
        "clutter_items": [],
    }

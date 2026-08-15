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

FLOOR_TYPES = ["concrete", "rubber", "wood"]


def _clear_environment():
    if "Environment" in bpy.data.collections:
        col = bpy.data.collections["Environment"]
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)


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
    """Concrete: noise grain + noise wear patches."""
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

    # Wear patches (second noise node — Musgrave was removed in Blender 4.0+)
    wear_noise = nodes.new("ShaderNodeTexNoise")
    wear_noise.inputs["Scale"].default_value = random.uniform(200, 400)
    wear_noise.inputs["Detail"].default_value = 4.0
    wear_noise.inputs["Roughness"].default_value = 0.7

    base_rgb = nodes.new("ShaderNodeRGB")
    base_rgb.outputs[0].default_value = base_color

    mix_grain = nodes.new("ShaderNodeMixRGB")
    mix_grain.blend_type = 'MIX'
    mix_grain.inputs["Fac"].default_value = 0.15

    mix_wear = nodes.new("ShaderNodeMixRGB")
    mix_wear.blend_type = 'MULTIPLY'
    mix_wear.inputs["Fac"].default_value = 0.1

    links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], wear_noise.inputs["Vector"])
    links.new(base_rgb.outputs[0], mix_grain.inputs["Color1"])
    links.new(noise.outputs["Fac"], mix_grain.inputs["Color2"])
    links.new(mix_grain.outputs[0], mix_wear.inputs["Color1"])
    links.new(wear_noise.outputs["Fac"], mix_wear.inputs["Color2"])
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


def _build_walls_and_ceiling_at(collection, cx, cy, venue_d, venue_w, venue_h):
    """Walls and ceiling enclosure centered at (cx, cy)."""
    mat = bpy.data.materials.new("WallMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.75, 0.73, 0.70, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9

    walls = []
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
        for col_ in list(wall.users_collection):
            col_.objects.unlink(wall)
        collection.objects.link(wall)
        walls.append(wall)

    bpy.ops.mesh.primitive_plane_add(size=1, location=(cx, cy, venue_h))
    ceiling = bpy.context.active_object
    ceiling.name = "Ceiling"
    ceiling.scale = (venue_d, venue_w, 1)
    bpy.ops.object.transform_apply(scale=True)
    ceiling.data.materials.append(mat)
    for col_ in list(ceiling.users_collection):
        col_.objects.unlink(ceiling)
    collection.objects.link(ceiling)

    return walls + [ceiling]


def _build_adjacent_court(collection, index, x_offset, y_offset):
    """Build a full adjacent court with surface and all lines."""
    from src.tools.blender_court import (
        SURFACE_COLORS, LINE_COLORS, LINE_Z, LINE_WIDTH,
        NET_POS, SHORT_SERVICE, LONG_SERVICE_DBL, SINGLES_OFFSET,
    )

    color_name = random.choice(list(SURFACE_COLORS.keys()))
    color = SURFACE_COLORS[color_name]

    cx = x_offset + COURT_LENGTH / 2
    cy = y_offset + COURT_WIDTH / 2

    bpy.ops.mesh.primitive_plane_add(size=1, location=(cx, cy, 0))
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

    for col_ in list(surface.users_collection):
        col_.objects.unlink(surface)
    collection.objects.link(surface)

    line_color_name = random.choice(list(LINE_COLORS.keys()))
    line_color = LINE_COLORS[line_color_name]
    line_mat = bpy.data.materials.new(f"AdjLineMat_{index}")
    line_mat.use_nodes = True
    bsdf = line_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = line_color
    bsdf.inputs["Roughness"].default_value = 0.5

    ss = NET_POS - SHORT_SERVICE
    rs = NET_POS + SHORT_SERVICE
    rls = COURT_LENGTH - LONG_SERVICE_DBL
    half_w = COURT_WIDTH / 2
    sb = COURT_WIDTH - SINGLES_OFFSET

    line_defs = [
        ("h", 0, 0, 0, COURT_WIDTH),
        ("h", COURT_LENGTH, 0, COURT_LENGTH, COURT_WIDTH),
        ("h", LONG_SERVICE_DBL, 0, LONG_SERVICE_DBL, COURT_WIDTH),
        ("h", rls, 0, rls, COURT_WIDTH),
        ("h", ss, 0, ss, COURT_WIDTH),
        ("h", rs, 0, rs, COURT_WIDTH),
        ("v", 0, 0, COURT_LENGTH, 0),
        ("v", 0, COURT_WIDTH, COURT_LENGTH, COURT_WIDTH),
        ("v", 0, SINGLES_OFFSET, COURT_LENGTH, SINGLES_OFFSET),
        ("v", 0, sb, COURT_LENGTH, sb),
        ("v", 0, half_w, ss, half_w),
        ("v", rs, half_w, COURT_LENGTH, half_w),
    ]

    for li, (orient, x1, y1, x2, y2) in enumerate(line_defs):
        dx = x2 - x1
        dy = y2 - y1
        length = (dx ** 2 + dy ** 2) ** 0.5
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            location=(x_offset + (x1 + x2) / 2, y_offset + (y1 + y2) / 2, LINE_Z),
        )
        obj = bpy.context.active_object
        obj.name = f"AdjCourt_{index}_Line_{li}"
        if orient == "h":
            obj.scale = (LINE_WIDTH, length, 1)
        else:
            obj.scale = (length, LINE_WIDTH, 1)
        bpy.ops.object.transform_apply(scale=True)
        obj.data.materials.append(line_mat)
        for col_ in list(obj.users_collection):
            col_.objects.unlink(obj)
        collection.objects.link(obj)

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
            for col_ in list(bench.users_collection):
                col_.objects.unlink(bench)
            collection.objects.link(bench)
            benches.append(bench)

    return benches


def _build_dividers(collection, positions):
    """Curtain dividers between courts at (x, y) positions."""
    mat = bpy.data.materials.new("DividerMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.2, 0.25, 0.35, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.95

    dividers = []
    for i, (x, y) in enumerate(positions):
        bpy.ops.mesh.primitive_plane_add(
            size=1,
            location=(x, y, 1.5),
        )
        div = bpy.context.active_object
        div.name = f"Divider_{i}"
        div.scale = (COURT_LENGTH, 0.01, 3.0)
        bpy.ops.object.transform_apply(scale=True)
        div.data.materials.append(mat)
        for col_ in list(div.users_collection):
            col_.objects.unlink(div)
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

    for col_ in list(board.users_collection):
        col_.objects.unlink(board)
    collection.objects.link(board)

    return board


def build_environment(config=None):
    """Build venue environment around the main court.

    Args:
        config: dict with optional max_adjacent_courts, spectator_chance,
                divider_chance, ceiling_height

    Returns:
        dict with environment metadata
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    _clear_environment()

    col = bpy.data.collections.new("Environment")
    bpy.context.scene.collection.children.link(col)

    # Pick adjacent courts from 3x3 grid
    num_adj = random.randint(0, cfg["max_adjacent_courts"])
    slots = random.sample(GRID_SLOTS, k=min(num_adj, len(GRID_SLOTS)))

    court_positions = []
    for i, (row_off, col_off) in enumerate(slots):
        x_offset = row_off * (COURT_LENGTH + COURT_GAP)
        y_offset = col_off * (COURT_WIDTH + COURT_GAP)
        _build_adjacent_court(col, i, x_offset, y_offset)
        court_positions.append((x_offset, y_offset))

    # Compute venue bounds from all courts (main + adjacent)
    all_x = [0.0] + [p[0] for p in court_positions]
    all_y = [0.0] + [p[1] for p in court_positions]
    min_x = min(all_x) - VENUE_MARGIN
    max_x = max(all_x) + COURT_LENGTH + VENUE_MARGIN
    min_y = min(all_y) - VENUE_MARGIN
    max_y = max(all_y) + COURT_WIDTH + VENUE_MARGIN

    venue_d = max_x - min_x
    venue_w = max_y - min_y
    venue_cx = (min_x + max_x) / 2
    venue_cy = (min_y + max_y) / 2

    venue_size = random.choice(list(cfg["ceiling_height"].keys()))
    venue_h = cfg["ceiling_height"][venue_size]

    floor, floor_type = _build_venue_floor_at(col, venue_cx, venue_cy, venue_d, venue_w)
    _build_walls_and_ceiling_at(col, venue_cx, venue_cy, venue_d, venue_w, venue_h)

    # Dividers between adjacent courts sharing a boundary
    divider_positions = []
    for x_off, y_off in court_positions:
        if y_off > 0:
            divider_positions.append((x_off + COURT_LENGTH / 2, y_off - COURT_GAP / 2))
        elif y_off < 0:
            divider_positions.append((x_off + COURT_LENGTH / 2, y_off + COURT_WIDTH + COURT_GAP / 2))

    has_dividers = False
    if divider_positions and random.random() < cfg["divider_chance"]:
        _build_dividers(col, divider_positions)
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

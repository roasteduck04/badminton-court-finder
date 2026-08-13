"""Venue environment: surrounding floor, walls, adjacent courts, spectators."""

import bpy
import math
import random

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
COURT_GAP = 2.0

DEFAULT_CONFIG = {
    "max_adjacent_courts": 2,
    "spectator_chance": 0.4,
    "divider_chance": 0.5,
    "venue_sizes": {"small": (18, 12, 7), "medium": (25, 18, 9), "large": (35, 25, 12)},
}

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

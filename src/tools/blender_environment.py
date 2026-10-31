"""Venue environment: surrounding floor, walls, adjacent courts, dividers."""

import random

import bpy

from src.tools.blender_common import (
    COURT_LENGTH, COURT_WIDTH, COURT_GAP,
    add_box, add_floor_rect, add_floor_quad, make_material, new_collection,
    wood_planks_socket,
)

VENUE_MARGIN = (6.0, 10.0)  # free floor between the outermost courts and the walls

DEFAULT_CONFIG = {
    "max_adjacent_courts": 8,
    "divider_chance": 0.5,
    "ceiling_height": {"small": 9, "medium": 12, "large": 16},
    "court_style": None,
}

# 3x3 grid positions around the main court (row_offset, col_offset)
GRID_SLOTS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

FLOOR_TYPES = ["concrete", "rubber", "wood"]

WALL_COLORS = [
    (0.10, 0.12, 0.45, 1.0), (0.15, 0.18, 0.50, 1.0), (0.35, 0.35, 0.40, 1.0),
    (0.30, 0.55, 0.35, 1.0), (0.80, 0.78, 0.72, 1.0), (0.60, 0.60, 0.62, 1.0),
    (0.55, 0.45, 0.30, 1.0), (0.85, 0.85, 0.85, 1.0), (0.20, 0.20, 0.22, 1.0),
    (0.65, 0.25, 0.15, 1.0),
]


def _venue_floor_material(floor_type):
    mat = bpy.data.materials.new("VenueFloorMat")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    tex = nodes.new("ShaderNodeTexCoord")
    base = nodes.new("ShaderNodeRGB")
    if floor_type == "concrete":
        v = random.uniform(0.3, 0.55)
        base.outputs[0].default_value = (v, v * 0.97, v * 0.92, 1.0)
        bsdf.inputs["Roughness"].default_value = random.uniform(0.6, 0.9)
    elif floor_type == "rubber":
        v = random.uniform(0.08, 0.25)
        base.outputs[0].default_value = (v, v, v * 1.1, 1.0)
        bsdf.inputs["Roughness"].default_value = random.uniform(0.8, 0.95)
    else:
        base.outputs[0].default_value = (0.50 + random.uniform(-0.08, 0.08),
                                         0.35 + random.uniform(-0.06, 0.06),
                                         0.20 + random.uniform(-0.05, 0.05), 1.0)
        bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 0.6)
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = random.uniform(30, 120)
    noise.inputs["Detail"].default_value = 5.0
    links.new(tex.outputs["Object"], noise.inputs["Vector"])
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = 'OVERLAY'
    mix.inputs["Fac"].default_value = random.uniform(0.05, 0.2)
    color = base.outputs[0]
    if floor_type == "wood":
        color = wood_planks_socket(nodes, links, base.outputs[0].default_value, random)
    links.new(color, mix.inputs["Color1"])
    links.new(noise.outputs["Color"], mix.inputs["Color2"])
    links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    return mat


def _build_walls_and_ceiling(collection, x0, y0, x1, y1, h):
    """Two-tone walls (boxes) and a ceiling around the venue rectangle."""
    split = random.uniform(2.5, 5.0)
    lower = make_material("WallMatLower", random.choice(WALL_COLORS), roughness=0.85)
    upper = make_material("WallMat", random.choice(WALL_COLORS), roughness=0.9)
    t = 0.2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d, w = x1 - x0, y1 - y0
    for part, mat, zb, zt in (("Lower", lower, 0.0, split), ("Upper", upper, split, h)):
        hz, cz = zt - zb, (zb + zt) / 2
        add_box(f"Wall_Back_{part}", (x0 - t / 2, cy, cz), (t, w + 2 * t, hz), collection, mat)
        add_box(f"Wall_Front_{part}", (x1 + t / 2, cy, cz), (t, w + 2 * t, hz), collection, mat)
        add_box(f"Wall_Left_{part}", (cx, y0 - t / 2, cz), (d, t, hz), collection, mat)
        add_box(f"Wall_Right_{part}", (cx, y1 + t / 2, cz), (d, t, hz), collection, mat)
    ceil_mat = make_material("CeilingMat", random.choice(WALL_COLORS), roughness=0.9)
    add_box("Ceiling", (cx, cy, h + t / 2), (d + 2 * t, w + 2 * t, t), collection, ceil_mat)


def _build_adjacent_court(collection, index, x_off, y_off, style):
    from src.tools.blender_court import build_court_markings, build_net
    build_court_markings(collection, style, x_off, y_off, prefix=f"AdjCourt_{index}_")
    if random.random() < 0.5:
        build_net(collection, x_off, y_off, prefix=f"AdjNet_{index}_", simple=True)


def _build_dividers(collection, segments):
    """Curtain dividers: thin upright boxes. segments = (x0, x1, y)."""
    color = random.choice([(0.2, 0.25, 0.35, 1.0), (0.1, 0.1, 0.1, 1.0),
                           (0.12, 0.3, 0.15, 1.0), (0.45, 0.45, 0.48, 1.0)])
    mat = make_material("DividerMat", color, roughness=0.95)
    height = random.uniform(2.5, 4.0)
    for i, (xa, xb, y) in enumerate(segments):
        add_box(f"Divider_{i}", ((xa + xb) / 2, y, height / 2), (xb - xa, 0.02, height),
                collection, mat)
    return height


def _build_scoreboard(collection):
    """Flip scoreboard beside the net, inside the inter-court gap."""
    side = random.choice([-1, 1])
    y = -random.uniform(1.0, 1.6) if side < 0 else COURT_WIDTH + random.uniform(1.0, 1.6)
    x = COURT_LENGTH / 2 + random.uniform(-2.5, 2.5)
    face_mat = make_material("ScoreboardMat", (0.1, 0.1, 0.12, 1.0), roughness=0.5)
    bsdf = face_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Emission Color"].default_value = (0.15, 0.2, 0.3, 1.0)
    bsdf.inputs["Emission Strength"].default_value = random.uniform(0.5, 2.5)
    frame_mat = make_material("ScoreboardFrameMat", (0.3, 0.3, 0.35, 1.0), metallic=0.5)
    add_box("Scoreboard", (x, y, 1.15), (0.9, 0.05, 0.6), collection, face_mat)
    add_box("ScoreboardFrame", (x, y + 0.03 * side, 1.15), (0.96, 0.02, 0.66), collection, frame_mat)
    for i, dx in enumerate((-0.35, 0.35)):
        add_box(f"ScoreboardLeg_{i}", (x + dx, y, 0.425), (0.04, 0.04, 0.85), collection, frame_mat)


def _build_floor_markings(collection, x0, y0, x1, y1):
    """Safety tape, court number and warm-up areas on the venue floor."""
    added = False
    if random.random() < 0.4:
        added = True
        mat = make_material("SafetyTapeMat", random.choice([
            (0.8, 0.7, 0.1, 1.0), (0.7, 0.1, 0.1, 1.0), (0.1, 0.2, 0.7, 1.0)]), 0.6)
        off = random.uniform(0.8, 1.8)
        L, W = COURT_LENGTH, COURT_WIDTH
        corners = [(-off, -off), (L + off, -off), (L + off, W + off), (-off, W + off)]
        for i in range(4):
            add_floor_quad(f"SafetyTape_{i}", corners[i], corners[(i + 1) % 4], 0.05,
                           0.0003, collection, mat)
    if random.random() < 0.3:
        added = True
        mat = make_material("CourtNumberMat", (0.8, 0.8, 0.1, 1.0), 0.5)
        x = random.choice([-0.9, COURT_LENGTH + 0.5])
        add_floor_rect("CourtNumber", x, COURT_WIDTH / 2 - 0.2, x + 0.4, COURT_WIDTH / 2 + 0.2,
                       0.0003, collection, mat)
    if random.random() < 0.3:
        added = True
        mat = make_material("WarmupAreaMat", random.choice([
            (0.2, 0.5, 0.2, 1.0), (0.2, 0.2, 0.6, 1.0)]), 0.75)
        for i, (wx, wy) in enumerate([(x0 + 2.5, y0 + 2), (x1 - 2.5, y1 - 2)]):
            if random.random() < 0.6:
                add_floor_rect(f"WarmupArea_{i}", wx - 1.5, wy - 1, wx + 1.5, wy + 1,
                               0.0002, collection, mat)
    return added


def build_environment(config=None):
    """Build the venue around the main court.

    Returns:
        dict with metadata plus "venue_bounds" (cx, cy, d, w, h, x0, y0, x1,
        y1), "footprint" (x0, y0, x1, y1 of all courts) and "court_offsets"
        (list of (x_off, y_off) of the adjacent courts).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    col = new_collection("Environment")

    from src.tools.blender_court import sample_court_style
    main_style = cfg["court_style"]

    num_adj = random.randint(0, cfg["max_adjacent_courts"])
    slots = random.sample(GRID_SLOTS, k=min(num_adj, len(GRID_SLOTS)))
    court_offsets = []
    for i, (row_off, col_off) in enumerate(slots):
        x_off = row_off * (COURT_LENGTH + COURT_GAP)
        y_off = col_off * (COURT_WIDTH + COURT_GAP)
        style = main_style if (main_style and random.random() < 0.7) else sample_court_style()
        style = dict(style, overlay=None, margin=[0.0, 0.0, 0.0, 0.0])
        _build_adjacent_court(col, i, x_off, y_off, style)
        court_offsets.append((x_off, y_off))

    fx0 = min([0.0] + [p[0] for p in court_offsets])
    fy0 = min([0.0] + [p[1] for p in court_offsets])
    fx1 = max([0.0] + [p[0] for p in court_offsets]) + COURT_LENGTH
    fy1 = max([0.0] + [p[1] for p in court_offsets]) + COURT_WIDTH
    margins = [random.uniform(*VENUE_MARGIN) for _ in range(4)]
    x0, x1 = fx0 - margins[0], fx1 + margins[1]
    y0, y1 = fy0 - margins[2], fy1 + margins[3]

    venue_size = random.choice(list(cfg["ceiling_height"].keys()))
    venue_h = cfg["ceiling_height"][venue_size] + random.uniform(-1.5, 3.0)

    floor_type = random.choice(FLOOR_TYPES)
    add_floor_rect("VenueFloor", x0, y0, x1, y1, -0.002, col, _venue_floor_material(floor_type))
    _build_walls_and_ceiling(col, x0, y0, x1, y1, venue_h)
    has_floor_markings = _build_floor_markings(col, x0, y0, x1, y1)

    # Dividers in the gaps between laterally adjacent court rows (including
    # the main court's row), spanning the courts on either side of the gap.
    all_courts = [(0.0, 0.0)] + court_offsets
    gaps = {}
    for x_off, y_off in all_courts:
        for nx, ny in all_courts:
            if nx == x_off and abs(ny - (y_off + COURT_WIDTH + COURT_GAP)) < 1e-6:
                gaps[(x_off, y_off + COURT_WIDTH + COURT_GAP / 2)] = True
    has_dividers = False
    if gaps and random.random() < cfg["divider_chance"]:
        _build_dividers(col, [(x - 1.0, x + COURT_LENGTH + 1.0, y) for (x, y) in gaps])
        has_dividers = True

    has_scoreboard = random.random() < 0.3
    if has_scoreboard:
        _build_scoreboard(col)

    return {
        "adjacent_courts": len(slots),
        "venue_size": venue_size,
        "has_dividers": has_dividers,
        "has_scoreboard": has_scoreboard,
        "has_floor_markings": has_floor_markings,
        "floor_type": floor_type,
        "venue_bounds": {
            "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
            "d": x1 - x0, "w": y1 - y0, "h": venue_h,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        },
        "footprint": {"x0": fx0, "y0": fy0, "x1": fx1, "y1": fy1},
        "court_offsets": court_offsets,
    }

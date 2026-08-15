"""Blender court geometry builder.

Builds a 3D badminton court with randomizable surface/line colors,
optional net, and 30 keypoint empties at line intersections.
"""

import bpy
import math
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
    "green": (0.10, 0.35, 0.13, 1.0),
    "blue": (0.08, 0.18, 0.45, 1.0),
    "red": (0.45, 0.10, 0.08, 1.0),
    "wood": (0.42, 0.26, 0.12, 1.0),
    "grey": (0.22, 0.24, 0.26, 1.0),
}

LINE_COLORS = {
    "white": (1.0, 1.0, 1.0, 1.0),
    "yellow": (1.0, 0.95, 0.55, 1.0),
    "light_grey": (0.88, 0.88, 0.88, 1.0),
}

DEFAULT_CONFIG = {
    "surface_color": "green",
    "line_color": "white",
    "include_net": True,
}


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


def _add_tape_residue(collection):
    """Add 0-5 small tape residue rectangles on the court surface."""
    if random.random() > 0.4:
        return
    count = random.randint(1, 5)
    mat = _make_material("TapeResidueMat", (0.8, 0.8, 0.75, 1.0), roughness=0.6)

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

    if "CourtSurfaceMat" in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials["CourtSurfaceMat"])
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

        for col in list(obj.users_collection):
            col.objects.unlink(obj)
        collection.objects.link(obj)
        lines.append(obj)

    return lines, color_name


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

            for col in list(empty.users_collection):
                col.objects.unlink(empty)
            collection.objects.link(empty)
            keypoints.append(empty)

    return keypoints


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
        for col in list(post.users_collection):
            col.objects.unlink(post)
        collection.objects.link(post)
        posts.append(post)

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

    bpy.ops.mesh.primitive_plane_add(
        size=1,
        location=(NET_POS, COURT_WIDTH / 2, NET_HEIGHT_CENTER / 2 + 0.2),
    )
    net = bpy.context.active_object
    net.name = "Net"
    net.scale = (0.02, COURT_WIDTH + 2 * POST_EXTENSION, NET_HEIGHT_CENTER * 0.65)
    bpy.ops.object.transform_apply(scale=True)
    net.data.materials.append(mat_net)
    for col in list(net.users_collection):
        col.objects.unlink(net)
    collection.objects.link(net)

    return net, posts


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
    _add_tape_residue(court_col)
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

"""Blender court geometry builder.

Builds a badminton court (surface, painted lines, optional other-sport line
overlays, optional net) with randomized appearance. Keypoint world positions
follow the template order of src/court_geometry.py: K = row * 5 + col at
world (x, y) = template (x along length, y along width), z = LINE_Z.
"""

import colorsys
import math
import random

import bpy

from src.tools.blender_common import (
    COURT_LENGTH, COURT_WIDTH, SINGLES_OFFSET, NET_POS, SHORT_SERVICE,
    LONG_SERVICE_DBL, LINE_Z, NUM_KEYPOINTS, KEYPOINTS_TEMPLATE,
    NET_HEIGHT_EDGE, NET_HEIGHT_CENTER, NET_DEPTH,
    add_box, add_cylinder, add_floor_quad, add_floor_rect, make_material,
    new_collection, wood_planks_socket, _mesh_object,
)

# Base colours per surface family (linear RGB); jittered in HSV per image.
SURFACE_FAMILIES = {
    "green": [(0.10, 0.42, 0.20), (0.15, 0.55, 0.30), (0.08, 0.30, 0.16), (0.20, 0.50, 0.35)],
    "blue": [(0.05, 0.18, 0.50), (0.08, 0.28, 0.60), (0.03, 0.10, 0.32), (0.10, 0.35, 0.55)],
    "red": [(0.50, 0.10, 0.08), (0.40, 0.12, 0.06), (0.55, 0.20, 0.10)],
    "grey": [(0.30, 0.31, 0.33), (0.18, 0.18, 0.20), (0.45, 0.45, 0.46)],
    "wood": [(0.55, 0.36, 0.18), (0.62, 0.44, 0.24), (0.42, 0.26, 0.12)],
}
SURFACE_FAMILY_WEIGHTS = {"green": 0.40, "blue": 0.18, "red": 0.10, "grey": 0.10, "wood": 0.22}

LINE_COLORS = {
    "white": (0.95, 0.95, 0.93, 1.0),
    "yellow": (0.90, 0.75, 0.08, 1.0),
    "black": (0.03, 0.03, 0.03, 1.0),
    "red": (0.70, 0.08, 0.06, 1.0),
    "light_blue": (0.35, 0.65, 0.90, 1.0),
}
OVERLAY_COLORS = {
    "yellow": (0.85, 0.70, 0.05, 1.0),
    "red": (0.65, 0.07, 0.05, 1.0),
    "blue": (0.05, 0.15, 0.60, 1.0),
    "black": (0.04, 0.04, 0.04, 1.0),
    "white": (0.90, 0.90, 0.90, 1.0),
    "orange": (0.85, 0.35, 0.05, 1.0),
}

DEFAULT_CONFIG = {
    "include_net": True,
    "style": None,  # None = sample a new style
}


# --------------------------------------------------------------------------
# Style sampling
# --------------------------------------------------------------------------

def _jitter_color(rgb, dh=0.02, ds=0.15, dv=0.2):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    h = (h + random.uniform(-dh, dh)) % 1.0
    s = min(1.0, max(0.0, s * (1 + random.uniform(-ds, ds))))
    v = min(1.0, max(0.01, v * (1 + random.uniform(-dv, dv))))
    return colorsys.hsv_to_rgb(h, s, v) + (1.0,)


def sample_court_style():
    """Sample the visual style of a court (shared with adjacent courts)."""
    fams = list(SURFACE_FAMILY_WEIGHTS)
    family = random.choices(fams, weights=[SURFACE_FAMILY_WEIGHTS[f] for f in fams])[0]
    surface_rgba = _jitter_color(random.choice(SURFACE_FAMILIES[family]))

    r = random.random()
    if r < 0.85:
        line_name = "white"
    elif r < 0.93:
        line_name = "yellow"
    else:
        line_name = random.choice(["black", "red", "light_blue"])
    if family == "wood" and line_name == "white" and random.random() < 0.3:
        line_name = random.choice(["black", "red", "yellow"])

    overlay = None
    if random.random() < (0.45 if family == "wood" else 0.2):
        choices = [c for c in OVERLAY_COLORS if c != line_name]
        overlay = {
            "sport": random.choice(["basketball", "volleyball"]),
            "color": random.choice(choices),
            "rot90": random.random() < 0.6,
            "dx": random.uniform(-4.0, 4.0),
            "dy": random.uniform(-3.0, 3.0),
        }

    return {
        "surface_family": family,
        "surface_rgba": surface_rgba,
        "roughness": random.uniform(0.35, 0.85),
        "grain_scale": random.uniform(40, 300),
        "grain_strength": random.uniform(0.02, 0.12),
        "mottle_scale": random.uniform(0.2, 1.5),
        "mottle_strength": random.uniform(0.0, 0.12),
        "margin": [random.uniform(0.0, 2.0) for _ in range(4)],  # -x, +x, -y, +y
        "mat_seams": family != "wood" and random.random() < 0.3,
        "line_color": line_name,
        "line_rgba": _jitter_color(LINE_COLORS[line_name][:3], 0.005, 0.05, 0.08),
        "line_width": random.uniform(0.035, 0.05),
        "line_wear": random.uniform(0.2, 0.7) if random.random() < 0.25 else 0.0,
        "taped": random.random() < 0.15,
        "overlay": overlay,
    }


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------

def _surface_material(style, name="CourtSurfaceMat"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = style["roughness"]
    tex = nodes.new("ShaderNodeTexCoord")
    base = nodes.new("ShaderNodeRGB")
    base.outputs[0].default_value = style["surface_rgba"]
    color_out = base.outputs[0]

    if style["surface_family"] == "wood":
        color_out = wood_planks_socket(nodes, links, style["surface_rgba"], random)

    grain = nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = style["grain_scale"]
    grain.inputs["Detail"].default_value = 4.0
    links.new(tex.outputs["Object"], grain.inputs["Vector"])
    mixg = nodes.new("ShaderNodeMixRGB")
    mixg.blend_type = 'OVERLAY'
    mixg.inputs["Fac"].default_value = style["grain_strength"]
    links.new(color_out, mixg.inputs["Color1"])
    links.new(grain.outputs["Color"], mixg.inputs["Color2"])
    color_out = mixg.outputs[0]

    if style["mottle_strength"] > 0:
        mottle = nodes.new("ShaderNodeTexNoise")
        mottle.inputs["Scale"].default_value = style["mottle_scale"]
        links.new(tex.outputs["Object"], mottle.inputs["Vector"])
        mixm = nodes.new("ShaderNodeMixRGB")
        mixm.blend_type = 'MULTIPLY'
        mixm.inputs["Fac"].default_value = style["mottle_strength"]
        links.new(color_out, mixm.inputs["Color1"])
        links.new(mottle.outputs["Fac"], mixm.inputs["Color2"])
        color_out = mixm.outputs[0]

    links.new(color_out, bsdf.inputs["Base Color"])
    return mat


def _line_material(style, name="CourtLineMat"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.25 if style["taped"] else 0.55
    if style["line_wear"] > 0:
        # Worn paint: noise mixes the line colour toward the surface colour.
        tex = nodes.new("ShaderNodeTexCoord")
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = random.uniform(8, 40)
        noise.inputs["Detail"].default_value = 6.0
        links.new(tex.outputs["Object"], noise.inputs["Vector"])
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.5 - style["line_wear"] * 0.4
        ramp.color_ramp.elements[1].position = 0.5 + style["line_wear"] * 0.2
        ramp.color_ramp.elements[0].color = style["surface_rgba"]
        ramp.color_ramp.elements[1].color = style["line_rgba"]
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = style["line_rgba"]
    return mat


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def court_line_segments():
    """(name, (x1, y1), (x2, y2)) for every painted badminton line."""
    ss = NET_POS - SHORT_SERVICE
    rs = NET_POS + SHORT_SERVICE
    rls = COURT_LENGTH - LONG_SERVICE_DBL
    cy = COURT_WIDTH / 2
    sb = COURT_WIDTH - SINGLES_OFFSET
    return [
        ("BaselineL", (0, 0), (0, COURT_WIDTH)),
        ("BaselineR", (COURT_LENGTH, 0), (COURT_LENGTH, COURT_WIDTH)),
        ("LongSvcL", (LONG_SERVICE_DBL, 0), (LONG_SERVICE_DBL, COURT_WIDTH)),
        ("LongSvcR", (rls, 0), (rls, COURT_WIDTH)),
        ("ShortSvcL", (ss, 0), (ss, COURT_WIDTH)),
        ("ShortSvcR", (rs, 0), (rs, COURT_WIDTH)),
        ("DblTop", (0, 0), (COURT_LENGTH, 0)),
        ("DblBot", (0, COURT_WIDTH), (COURT_LENGTH, COURT_WIDTH)),
        ("SglTop", (0, SINGLES_OFFSET), (COURT_LENGTH, SINGLES_OFFSET)),
        ("SglBot", (0, sb), (COURT_LENGTH, sb)),
        ("CenterL", (0, cy), (ss, cy)),
        ("CenterR", (rs, cy), (COURT_LENGTH, cy)),
    ]


def _overlay_segments(sport):
    """Line segments (local frame, centred at origin, long axis = x)."""
    segs = []

    def rect(x0, y0, x1, y1):
        segs.extend([((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                     ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))])

    def arc(cx, cy, r, a0, a1, n=24):
        pts = [(cx + r * math.cos(a0 + (a1 - a0) * i / n),
                cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
        segs.extend(zip(pts[:-1], pts[1:]))

    if sport == "basketball":
        L, W = 28.0, 15.0
        rect(-L / 2, -W / 2, L / 2, W / 2)
        segs.append(((0, -W / 2), (0, W / 2)))
        arc(0, 0, 1.8, 0, 2 * math.pi, 32)
        for sgn in (-1, 1):
            bx = sgn * L / 2
            rect(min(bx, bx - sgn * 5.8), -2.45, max(bx, bx - sgn * 5.8), 2.45)
            hoop = bx - sgn * 1.575
            if sgn < 0:
                arc(hoop, 0, 6.75, -1.35, 1.35)
            else:
                arc(hoop, 0, 6.75, math.pi - 1.35, math.pi + 1.35)
    else:  # volleyball
        L, W = 18.0, 9.0
        rect(-L / 2, -W / 2, L / 2, W / 2)
        segs.append(((0, -W / 2), (0, W / 2)))
        segs.append(((-3, -W / 2), (-3, W / 2)))
        segs.append(((3, -W / 2), (3, W / 2)))
    return segs


def _build_overlay(collection, overlay):
    mat = make_material("OverlayLineMat", OVERLAY_COLORS[overlay["color"]], roughness=0.5)
    cx = COURT_LENGTH / 2 + overlay["dx"]
    cy = COURT_WIDTH / 2 + overlay["dy"]
    rot = math.pi / 2 if overlay["rot90"] else 0.0
    c, s = math.cos(rot), math.sin(rot)
    for i, (p1, p2) in enumerate(_overlay_segments(overlay["sport"])):
        q1 = (cx + c * p1[0] - s * p1[1], cy + s * p1[0] + c * p1[1])
        q2 = (cx + c * p2[0] - s * p2[1], cy + s * p2[0] + c * p2[1])
        add_floor_quad(f"Overlay_{i}", q1, q2, 0.05, 0.0006, collection, mat)


def build_court_markings(collection, style, x_off=0.0, y_off=0.0, prefix="Court",
                         surface_mat=None, line_mat=None):
    """Surface (with surround margin) + painted lines for one court."""
    surface_mat = surface_mat or _surface_material(style, f"{prefix}SurfaceMat")
    line_mat = line_mat or _line_material(style, f"{prefix}LineMat")
    m = style["margin"]
    surface = add_floor_rect(f"{prefix}Surface", x_off - m[0], y_off - m[2],
                             x_off + COURT_LENGTH + m[1], y_off + COURT_WIDTH + m[3],
                             0.0, collection, surface_mat)

    if style["mat_seams"]:
        seam_mat = make_material(f"{prefix}SeamMat",
                                 tuple(c * 0.6 for c in style["surface_rgba"][:3]) + (1.0,), 0.8)
        n = 4
        for i in range(1, n):
            y = y_off - m[2] + (COURT_WIDTH + m[2] + m[3]) * i / n
            add_floor_quad(f"{prefix}Seam_{i}", (x_off - m[0], y),
                           (x_off + COURT_LENGTH + m[1], y), 0.006, 0.0004,
                           collection, seam_mat)

    z = LINE_Z + (0.0005 if style["taped"] else 0.0)
    lines = []
    for name, p1, p2 in court_line_segments():
        lw = style["line_width"] * (random.uniform(0.95, 1.05) if style["taped"] else 1.0)
        # Extend each line by half a line width so corners are fully painted.
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        ln = math.hypot(dx, dy)
        ex, ey = dx / ln * lw / 2, dy / ln * lw / 2
        q1 = (x_off + p1[0] - ex, y_off + p1[1] - ey)
        q2 = (x_off + p2[0] + ex, y_off + p2[1] + ey)
        lines.append(add_floor_quad(f"{prefix}Line_{name}", q1, q2, lw, z, collection, line_mat))
    return surface, lines


def _add_tape_residue(collection):
    """0-5 small tape residue rectangles on the court surface."""
    if random.random() > 0.4:
        return
    mat = make_material("TapeResidueMat", (0.8, 0.8, 0.75, 1.0), roughness=0.6)
    for i in range(random.randint(1, 5)):
        x = random.uniform(0.5, COURT_LENGTH - 0.5)
        y = random.uniform(0.5, COURT_WIDTH - 0.5)
        w, h = random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)
        add_floor_rect(f"TapeResidue_{i}", x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                       0.0005, collection, mat)


def _net_top_z(t):
    """Net top height; t in [-1, 1] across the net (0 = centre)."""
    return NET_HEIGHT_CENTER + (NET_HEIGHT_EDGE - NET_HEIGHT_CENTER) * t * t


def _net_material(name):
    """Semi-transparent mesh: Transparent BSDF where the holes are."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = 'DITHERED'
    if hasattr(mat, "use_transparent_shadow"):
        mat.use_transparent_shadow = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    color = random.choice([(0.02, 0.02, 0.02, 1.0), (0.03, 0.08, 0.04, 1.0),
                           (0.10, 0.07, 0.04, 1.0), (0.05, 0.05, 0.12, 1.0)])
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.9
    out = nodes["Material Output"]
    transp = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")

    # Grid mask: 1 on the strings, 0 in the holes (mesh ~15-20 mm).
    tex = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tex.outputs["Object"], sep.inputs[0])
    cell = random.uniform(0.015, 0.02)
    thick = random.uniform(0.12, 0.25)
    masks = []
    for axis in ("Y", "Z"):
        scale = nodes.new("ShaderNodeMath")
        scale.operation = 'MULTIPLY'
        scale.inputs[1].default_value = 1.0 / cell
        links.new(sep.outputs[axis], scale.inputs[0])
        frac = nodes.new("ShaderNodeMath")
        frac.operation = 'FRACT'
        links.new(scale.outputs[0], frac.inputs[0])
        lt = nodes.new("ShaderNodeMath")
        lt.operation = 'LESS_THAN'
        lt.inputs[1].default_value = thick
        links.new(frac.outputs[0], lt.inputs[0])
        masks.append(lt)
    mx = nodes.new("ShaderNodeMath")
    mx.operation = 'MAXIMUM'
    links.new(masks[0].outputs[0], mx.inputs[0])
    links.new(masks[1].outputs[0], mx.inputs[1])

    links.new(mx.outputs[0], mix.inputs["Fac"])
    links.new(transp.outputs[0], mix.inputs[1])
    links.new(bsdf.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def _net_strip(name, y0, y1, x, z_top_fn, height, collection, mat, segments=24):
    """Upright strip following the net's sagging top edge (double-sided)."""
    verts, faces = [], []
    for i in range(segments + 1):
        t = i / segments
        y = y0 + (y1 - y0) * t
        zt = z_top_fn(2 * t - 1)
        verts.append((x, y, zt - height))
        verts.append((x, y, zt))
    for i in range(segments):
        a = 2 * i
        faces.append((a, a + 2, a + 3, a + 1))
    return _mesh_object(name, verts, faces, collection, mat, occluder=False)


def build_net(collection, x_off=0.0, y_off=0.0, prefix="Net", simple=False):
    """Posts + see-through net (0.76 m tall, top 1.55 m at posts, 1.524 m centre)."""
    post_off = 0.0 if random.random() < 0.5 else random.uniform(0.05, 0.4)
    x = x_off + NET_POS
    y0, y1 = y_off - post_off, y_off + COURT_WIDTH + post_off
    post_mat = make_material(f"{prefix}PostMat", random.choice([
        (0.6, 0.6, 0.65, 1.0), (0.05, 0.15, 0.5, 1.0), (0.1, 0.4, 0.15, 1.0),
        (0.7, 0.1, 0.08, 1.0)]), roughness=0.35, metallic=0.6)
    posts = []
    for i, y in enumerate((y0, y1)):
        posts.append(add_cylinder(f"{prefix}Post_{i}", 0.02, NET_HEIGHT_EDGE,
                                  (x, y, NET_HEIGHT_EDGE / 2), collection, post_mat))
        add_box(f"{prefix}PostBase_{i}", (x, y + (-0.25 if i == 0 else 0.25), 0.04),
                (0.35, 0.6, 0.08), collection, post_mat)

    net_mat = _net_material(f"{prefix}MeshMat")
    net = _net_strip(f"{prefix}Mesh", y0 + 0.01, y1 - 0.01, x, _net_top_z, NET_DEPTH,
                     collection, net_mat)
    tape_mat = make_material(f"{prefix}TapeMat", (0.92, 0.92, 0.92, 1.0), roughness=0.6)
    _net_strip(f"{prefix}Tape", y0 + 0.01, y1 - 0.01, x + 0.002,
               lambda t: _net_top_z(t) + 0.002, 0.038, collection, tape_mat)
    _net_strip(f"{prefix}TapeBack", y0 + 0.01, y1 - 0.01, x - 0.002,
               lambda t: _net_top_z(t) + 0.002, 0.038, collection, tape_mat)

    if not simple and random.random() < 0.3:
        pad_mat = make_material(f"{prefix}PadMat", random.choice([
            (0.8, 0.7, 0.1, 1.0), (0.1, 0.3, 0.8, 1.0), (0.8, 0.15, 0.1, 1.0)]), roughness=0.9)
        for i, y in enumerate((y0, y1)):
            add_cylinder(f"{prefix}PostPad_{i}", 0.06, 1.0, (x, y, 0.5), collection, pad_mat)
    return net, posts


def build_court(config=None):
    """Build the main badminton court.

    Args:
        config: dict with optional keys include_net (bool) and style (dict
            from sample_court_style(); sampled if None).

    Returns:
        dict with surface, lines, net, posts, keypoints (30 world (x, y, z)
        tuples in template order), style, surface_color, line_color,
        include_net.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    style = cfg["style"] or sample_court_style()

    court_col = new_collection("Court")
    surface, lines = build_court_markings(court_col, style, prefix="Court")
    if style["overlay"]:
        _build_overlay(court_col, style["overlay"])
    _add_tape_residue(court_col)

    net, posts = None, []
    if cfg["include_net"]:
        net_col = new_collection("NetAssembly")
        net, posts = build_net(net_col)

    keypoints = [(x, y, LINE_Z) for x, y in KEYPOINTS_TEMPLATE]
    assert len(keypoints) == NUM_KEYPOINTS

    return {
        "surface": surface,
        "lines": lines,
        "net": net,
        "posts": posts,
        "keypoints": keypoints,
        "style": style,
        "surface_color": style["surface_family"],
        "line_color": style["line_color"],
        "include_net": bool(cfg["include_net"]),
    }


def style_metadata(style):
    """JSON-safe summary of a court style."""
    return {
        "surface_family": style["surface_family"],
        "surface_rgb": [round(c, 4) for c in style["surface_rgba"][:3]],
        "line_color": style["line_color"],
        "line_width": round(style["line_width"], 4),
        "line_wear": round(style["line_wear"], 3),
        "taped": style["taped"],
        "mat_seams": style["mat_seams"],
        "overlay": style["overlay"]["sport"] + "/" + style["overlay"]["color"]
        if style["overlay"] else None,
    }

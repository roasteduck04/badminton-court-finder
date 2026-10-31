"""Venue details: wall banners, ceiling trusses, light housings, seating, clutter.

Upright flat items (banners, signs, windows, towels) are thin boxes, never
z-scaled planes (a plane has no z extent and collapses). Bleachers are placed
outside the footprint of all courts so they never land on an adjacent court.
"""

import math
import random

import bpy

from src.tools.blender_common import (
    COURT_LENGTH, COURT_WIDTH,
    add_box, add_cone, add_cylinder, add_sphere, make_material, new_collection,
)


def _banner_material(name):
    """Procedural two-colour sponsor banner."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    wave = nodes.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = random.uniform(3, 8)
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (random.uniform(0.2, 0.9), random.uniform(0.1, 0.8),
                                         random.uniform(0.1, 0.8), 1.0)
    ramp.color_ramp.elements[1].color = (random.uniform(0.2, 0.9), random.uniform(0.1, 0.8),
                                         random.uniform(0.1, 0.8), 1.0)
    links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs[0], bsdf.inputs["Base Color"])
    return mat


def _wall_item(name, bounds, wall, along, z, width, height, collection, mat, inset=0.02):
    """Thin box flush against the inner face of a wall.

    wall in {"back" (x=x0), "front" (x=x1), "left" (y=y0), "right" (y=y1)};
    ``along`` is the coordinate along the wall (y for back/front, x for
    left/right).
    """
    t = 0.02
    if wall == "back":
        return add_box(name, (bounds["x0"] + inset + t / 2, along, z), (t, width, height), collection, mat)
    if wall == "front":
        return add_box(name, (bounds["x1"] - inset - t / 2, along, z), (t, width, height), collection, mat)
    if wall == "left":
        return add_box(name, (along, bounds["y0"] + inset + t / 2, z), (width, t, height), collection, mat)
    return add_box(name, (along, bounds["y1"] - inset - t / 2, z), (width, t, height), collection, mat)


def _wall_along_range(bounds, wall, margin=1.5):
    if wall in ("back", "front"):
        return bounds["y0"] + margin, bounds["y1"] - margin
    return bounds["x0"] + margin, bounds["x1"] - margin


WALLS = ["back", "front", "left", "right"]


def _build_wall_banners(collection, bounds):
    """Sponsor banners on 1-3 walls."""
    count = 0
    for wall in random.sample(WALLS, k=random.randint(1, 3)):
        lo, hi = _wall_along_range(bounds, wall, 2.0)
        for _ in range(random.randint(1, 4)):
            w, h = random.uniform(1.0, 4.0), random.uniform(0.5, 1.5)
            _wall_item(f"WallBanner_{count}", bounds, wall, random.uniform(lo, hi),
                       random.uniform(1.0, 3.5), w, h, collection,
                       _banner_material(f"BannerMat_{count}"))
            count += 1
    return count


def _build_exit_signs(collection, bounds):
    """Small emissive exit signs above doors."""
    for i in range(random.randint(1, 2)):
        color = random.choice([(0.1, 0.8, 0.1, 1.0), (0.8, 0.1, 0.1, 1.0)])
        mat = make_material(f"ExitSignMat_{i}", color, emission=random.uniform(2, 5))
        wall = random.choice(WALLS)
        lo, hi = _wall_along_range(bounds, wall)
        _wall_item(f"ExitSign_{i}", bounds, wall, random.uniform(lo, hi),
                   random.uniform(2.3, 3.0), 0.35, 0.15, collection, mat, inset=0.03)


def _build_windows(collection, bounds):
    """Bright rectangles high on the walls simulating windows (~40%)."""
    if random.random() > 0.4:
        return False
    mat = make_material("WindowMat", (0.85, 0.88, 0.92, 1.0), roughness=0.3,
                        emission=random.uniform(0.5, 2.0))
    i = 0
    for wall in random.sample(WALLS, k=random.randint(1, 2)):
        lo, hi = _wall_along_range(bounds, wall, 3.0)
        n = random.randint(1, 5)
        z = bounds["h"] * random.uniform(0.55, 0.8)
        for k in range(n):
            along = lo + (hi - lo) * (k + 0.5) / n
            _wall_item(f"Window_{i}", bounds, wall, along, z, random.uniform(1.5, 4),
                       random.uniform(1, 2), collection, mat)
            i += 1
    return True


def _build_trusses(collection, bounds):
    """Exposed ceiling trusses (~50%)."""
    if random.random() > 0.5:
        return False
    cy, d, w, h = bounds["cy"], bounds["d"], bounds["w"], bounds["h"]
    mat = make_material("TrussMat", (0.5, 0.5, 0.55, 1.0), roughness=0.4, metallic=0.7)
    n = random.randint(3, 7)
    spacing = d / (n + 1)
    for i in range(n):
        bx = bounds["x0"] + spacing * (i + 1)
        add_box(f"Truss_Beam_{i}", (bx, cy, h - 0.5), (0.15, w * 0.98, 0.3), collection, mat)
        if i < n - 1:
            nx = bx + spacing
            ln = math.hypot(spacing, w * 0.3)
            add_box(f"Truss_Brace_{i}", ((bx + nx) / 2, cy, h - 0.5), (ln, 0.05, 0.05),
                    collection, mat, rot_z=math.atan2(w * 0.3, spacing))
    return True


def _build_ducts(collection, bounds):
    """Ventilation ducts (~40%)."""
    if random.random() > 0.4:
        return False
    mat = make_material("DuctMat", (0.6, 0.6, 0.65, 1.0), roughness=0.35, metallic=0.6)
    for i in range(random.randint(1, 3)):
        add_box(f"Duct_{i}", (bounds["cx"] + random.uniform(-0.2, 0.2) * bounds["d"],
                              bounds["cy"] + random.uniform(-0.3, 0.3) * bounds["w"],
                              bounds["h"] - 0.8),
                (random.uniform(3, 10), 0.5, 0.5), collection, mat)
    return True


def _build_hanging_banners(collection, bounds):
    """Banners hanging from the ceiling (~30%)."""
    if random.random() > 0.3:
        return False
    h = bounds["h"]
    for i in range(random.randint(1, 3)):
        bw, bh = random.uniform(1, 2), random.uniform(1.5, 3)
        bx = bounds["cx"] + random.uniform(-0.4, 0.4) * bounds["d"]
        by = bounds["cy"] + random.choice([-1, 1]) * bounds["w"] * random.uniform(0.3, 0.45)
        add_cylinder(f"HangBanner_Cable_{i}", 0.01, 1.5, (bx, by, h - 0.75), collection)
        add_box(f"HangBanner_{i}", (bx, by, h - 1.5 - bh / 2), (bw, 0.02, bh), collection,
                _banner_material(f"HangBannerMat_{i}"))
    return True


def _build_light_housings(collection):
    """Housings above each overhead area light."""
    if "Lighting" not in bpy.data.collections:
        return False
    mat = make_material("LightHousingMat", (0.85, 0.85, 0.88, 1.0), roughness=0.5, emission=0.3)
    count = 0
    for obj in bpy.data.collections["Lighting"].objects:
        if obj.type != 'LIGHT' or obj.data.type != 'AREA' or "Ambient" in obj.name:
            continue
        lx, ly, lz = obj.location
        add_box(f"LightHousing_{count}", (lx, ly, lz + 0.15), (1.2, 0.6, 0.25), collection, mat)
        count += 1
    return count > 0


def _build_bleacher_seating(collection, bounds, footprint):
    """Tiered stands outside the footprint of all courts.

    Returns (seating_type, spectator_count, placements). Each placement is a
    dict usable by the camera module for spectator viewpoints.
    """
    schemes = [((0.15, 0.35, 0.70, 1.0), (0.85, 0.70, 0.10, 1.0)),
               ((0.60, 0.10, 0.10, 1.0), (0.80, 0.80, 0.80, 1.0)),
               ((0.10, 0.30, 0.15, 1.0), (0.80, 0.65, 0.08, 1.0)),
               ((0.35, 0.35, 0.38, 1.0), (0.70, 0.70, 0.72, 1.0))]
    primary, secondary = random.choice(schemes)
    mat = make_material("BleacherMat", primary, roughness=0.75)
    mat_alt = make_material("BleacherMatAlt", secondary, roughness=0.75)
    has_backs = random.random() < 0.4
    seating_type = "bleacher_with_seats" if has_backs else "bleacher_bench"

    sides = [s for s in WALLS if random.random() < 0.35]
    if not sides and random.random() < 0.8:
        sides = [random.choice(WALLS)]

    row_d, row_h = 0.7, 0.4
    placements = []
    spectators = 0
    for side in sides:
        if side in ("back", "front"):
            free = (footprint["x0"] - bounds["x0"]) if side == "back" else (bounds["x1"] - footprint["x1"])
            span_lo, span_hi = footprint["y0"], footprint["y1"]
        else:
            free = (footprint["y0"] - bounds["y0"]) if side == "left" else (bounds["y1"] - footprint["y1"])
            span_lo, span_hi = footprint["x0"], footprint["x1"]
        gap = 1.5
        max_rows = int((free - gap - 0.3) / row_d)
        if max_rows < 1:
            continue
        rows = random.randint(1, min(6, max_rows))
        frac = random.uniform(0.4, 0.95)
        mid = (span_lo + span_hi) / 2 + random.uniform(-0.2, 0.2) * (span_hi - span_lo) * (1 - frac)
        span = (span_hi - span_lo) * frac
        a, b = mid - span / 2, mid + span / 2
        sign = -1 if side in ("back", "left") else 1
        if side == "back":
            start = footprint["x0"] - gap
        elif side == "front":
            start = footprint["x1"] + gap
        elif side == "left":
            start = footprint["y0"] - gap
        else:
            start = footprint["y1"] + gap

        for r in range(rows):
            depth_c = start + sign * (r + 0.5) * row_d
            top = (r + 1) * row_h
            if side in ("back", "front"):
                add_box(f"Bleacher_{side}_{r}", (depth_c, mid, top / 2), (row_d, span, top), collection, mat)
            else:
                add_box(f"Bleacher_{side}_{r}", (mid, depth_c, top / 2), (span, row_d, top), collection, mat)
            n_seats = int(span / 0.5)
            occupancy = random.uniform(0, 0.35)
            for s in range(n_seats):
                along = a + (s + 0.5) * 0.5
                back_c = depth_c + sign * 0.25
                if has_backs:
                    smat = mat if s % 2 == 0 else mat_alt
                    if side in ("back", "front"):
                        add_box(f"SeatBack_{side}_{r}_{s}", (back_c, along, top + 0.2), (0.05, 0.4, 0.4),
                                collection, smat)
                    else:
                        add_box(f"SeatBack_{side}_{r}_{s}", (along, back_c, top + 0.2), (0.4, 0.05, 0.4),
                                collection, smat)
                if random.random() < occupancy:
                    spectators += 1
                    sp_mat = make_material(f"SpectatorMat_{spectators}", random.choice([
                        (0.3, 0.3, 0.5, 1.0), (0.5, 0.2, 0.2, 1.0), (0.8, 0.8, 0.8, 1.0),
                        (0.2, 0.4, 0.2, 1.0), (0.1, 0.1, 0.1, 1.0)]))
                    loc = (depth_c, along) if side in ("back", "front") else (along, depth_c)
                    add_cylinder(f"Spectator_{spectators}_Torso", 0.15, 0.55,
                                 (loc[0], loc[1], top + 0.3), collection, sp_mat)
                    add_sphere(f"Spectator_{spectators}_Head", 0.1,
                               (loc[0], loc[1], top + 0.7), collection,
                               make_material(f"SpectatorSkin_{spectators}",
                                             (0.7, 0.55, 0.4, 1.0), roughness=0.6))
        placements.append({
            "side": side, "start": start, "direction": sign, "row_depth": row_d,
            "num_rows": rows, "top_z": rows * row_h, "span": [a, b],
        })
    return seating_type, spectators, placements


def _build_popup_banners(collection):
    """Pop-up sponsor banners around the court perimeter."""
    if random.random() > 0.6:
        return 0
    colors = [(0.85, 0.20, 0.15, 1.0), (0.15, 0.30, 0.70, 1.0), (0.90, 0.55, 0.10, 1.0),
              (0.10, 0.50, 0.25, 1.0), (0.95, 0.95, 0.95, 1.0), (0.50, 0.10, 0.55, 1.0)]
    positions = [(x, y, 0.0) for y in (-1.4, COURT_WIDTH + 1.4) for x in (2.5, 4.5, 9.0, 11.0)]
    positions += [(x, y, math.pi / 2) for x in (-1.4, COURT_LENGTH + 1.4) for y in (1.5, 4.6)]
    chosen = random.sample(positions, k=random.randint(2, 7))
    for i, (bx, by, rot) in enumerate(chosen):
        w, h = random.uniform(0.8, 2.0), random.uniform(0.5, 0.9)
        add_box(f"PopupBanner_{i}", (bx, by, h / 2), (w, 0.03, h), collection,
                make_material(f"PopupBannerMat_{i}", random.choice(colors), 0.7), rot_z=rot)
    return len(chosen)


def _chair(collection, prefix, x, y, facing_neg_y, mat, seat_mat=None):
    seat_mat = seat_mat or mat
    add_box(f"{prefix}_Seat", (x, y, 0.45), (0.42, 0.42, 0.05), collection, seat_mat)
    back_y = y + (0.2 if facing_neg_y else -0.2)
    add_box(f"{prefix}_Back", (x, back_y, 0.68), (0.42, 0.04, 0.4), collection, mat)
    for li, (dx, dy) in enumerate([(-0.18, -0.18), (0.18, -0.18), (-0.18, 0.18), (0.18, 0.18)]):
        add_box(f"{prefix}_Leg_{li}", (x + dx, y + dy, 0.215), (0.025, 0.025, 0.43), collection, mat)
    return back_y


def _gap_y(side_neg):
    """A y coordinate in the 2 m gap beside the main court."""
    return -random.uniform(0.9, 1.6) if side_neg else COURT_WIDTH + random.uniform(0.9, 1.6)


def _build_courtside_furniture(collection):
    """Player chairs, towels, drink station, equipment bags."""
    if random.random() < 0.6:
        chair_mat = make_material("ChairMat", (0.3, 0.3, 0.35, 1.0), roughness=0.5, metallic=0.4)
        for i in range(random.randint(1, 2)):
            neg = random.random() < 0.5
            cx, cy = COURT_LENGTH / 2 + random.uniform(-2.5, 2.5), _gap_y(neg)
            back_y = _chair(collection, f"PlayerChair_{i}", cx, cy, neg, chair_mat)
            if random.random() < 0.5:
                towel = make_material(f"TowelMat_{i}", (random.uniform(0.3, 0.9), random.uniform(0.3, 0.9),
                                                        random.uniform(0.3, 0.9), 1.0))
                add_box(f"Towel_{i}", (cx, back_y + (0.03 if neg else -0.03), 0.7),
                        (0.3, 0.015, 0.4), collection, towel)
        if random.random() < 0.4:
            neg = random.random() < 0.5
            tx, ty = COURT_LENGTH / 2 + random.uniform(-3, 3), _gap_y(neg)
            add_box("DrinkTable", (tx, ty, 0.35), (0.6, 0.4, 0.7), collection,
                    make_material("DrinkTableMat", (0.35, 0.35, 0.38, 1.0)))
            bottle = make_material("BottleMat", (0.3, 0.5, 0.8, 1.0), roughness=0.3)
            for bi in range(random.randint(1, 3)):
                add_cylinder(f"Bottle_{bi}", 0.03, 0.2, (tx + random.uniform(-0.2, 0.2), ty, 0.8),
                             collection, bottle)
    if random.random() < 0.5:
        for i in range(random.randint(1, 3)):
            bag = make_material(f"BagMat_{i}", random.choice([
                (0.05, 0.05, 0.2, 1.0), (0.08, 0.08, 0.08, 1.0), (0.6, 0.1, 0.1, 1.0)]))
            if random.random() < 0.6:
                bx, by = random.uniform(-1.0, COURT_LENGTH + 1.0), _gap_y(random.random() < 0.5)
            else:
                bx = random.choice([-random.uniform(0.8, 1.6), COURT_LENGTH + random.uniform(0.8, 1.6)])
                by = random.uniform(0, COURT_WIDTH)
            add_box(f"EquipBag_{i}", (bx, by, 0.15), (0.75, 0.3, 0.3), collection, bag,
                    rot_z=random.uniform(0, math.pi))


def _build_clutter(collection, bounds):
    """Courtside clutter items with independent spawn chances."""
    items = []
    if random.random() < 0.15:
        mat = make_material("MopBucketMat", (0.4, 0.4, 0.45, 1.0))
        bx = bounds["cx"] + random.choice([-1, 1]) * (bounds["d"] / 2 - 1)
        by = bounds["cy"] + random.uniform(-0.3, 0.3) * bounds["w"]
        add_cylinder("MopBucket", 0.15, 0.3, (bx, by, 0.15), collection, mat)
        add_cylinder("MopHandle", 0.01, 1.2, (bx, by, 0.75), collection, mat,
                     rotation=(math.radians(15), 0, 0))
        items.append("mop_bucket")
    if random.random() < 0.10:
        bx = bounds["cx"] + random.choice([-1, 1]) * (bounds["d"] / 2 - 1.5)
        add_cylinder("RolledNet", 0.15, 2.0, (bx, bounds["cy"] + random.uniform(-2, 2), 0.15),
                     collection, make_material("RolledNetMat", (0.2, 0.2, 0.22, 1.0)),
                     rotation=(0, math.pi / 2, random.uniform(0, math.pi)))
        items.append("rolled_net")
    if random.random() < 0.15:
        mat = make_material("StackedChairMat", (0.35, 0.35, 0.38, 1.0), metallic=0.3)
        bx = bounds["cx"] + random.choice([-1, 1]) * (bounds["d"] / 2 - 1)
        by = bounds["cy"] + random.choice([-1, 1]) * (bounds["w"] / 2 - 1)
        for ci in range(random.randint(2, 4)):
            add_box(f"StackedChair_{ci}", (bx, by, 0.2 + ci * 0.35), (0.4, 0.4, 0.35), collection, mat)
        items.append("stacked_chairs")
    if random.random() < 0.20:
        mat = make_material("ConeMat", (0.9, 0.5, 0.05, 1.0))
        for ci in range(random.randint(1, 4)):
            add_cone(f"TrainingCone_{ci}", 0.1, 0.2,
                     (random.uniform(-1.5, COURT_LENGTH + 1.5), _gap_y(random.random() < 0.5), 0.1),
                     collection, mat)
        items.append("training_cones")
    return items


def _build_line_judge_chairs(collection, lighting_preset=None):
    """Line judge chairs near the court corners."""
    if random.random() > (0.5 if lighting_preset == "competition" else 0.2):
        return False
    frame = make_material("LineJudgeChairMat", (0.4, 0.4, 0.45, 1.0), roughness=0.4, metallic=0.5)
    seat = make_material("LineJudgeSeatMat", (0.15, 0.15, 0.18, 1.0))
    corners = [(-1.3, -1.3), (-1.3, COURT_WIDTH + 1.3),
               (COURT_LENGTH + 1.3, -1.3), (COURT_LENGTH + 1.3, COURT_WIDTH + 1.3)]
    for i, (jx, jy) in enumerate(random.sample(corners, k=random.randint(2, 4))):
        _chair(collection, f"LineJudgeChair_{i}", jx, jy, jy < 0, frame, seat)
        if random.random() < 0.5:
            add_cylinder(f"LineJudge_{i}_Torso", 0.15, 0.55, (jx, jy, 0.75), collection,
                         make_material(f"LineJudgeMat_{i}", random.choice([
                             (0.1, 0.1, 0.15, 1.0), (0.2, 0.2, 0.25, 1.0)])))
            add_sphere(f"LineJudge_{i}_Head", 0.1, (jx, jy, 1.15), collection,
                       make_material(f"LineJudgeSkin_{i}", (0.65, 0.48, 0.35, 1.0)))
    return True


def build_venue_details(config=None):
    """Build venue wall/ceiling details, seating, and clutter.

    Args:
        config: dict with "venue_bounds" (cx, cy, d, w, h, x0, y0, x1, y1),
            "footprint" (x0, y0, x1, y1 of all courts) and optionally
            "lighting_preset".

    Returns:
        dict of detail flags for metadata, plus "bleacher_placements".
    """
    cfg = config or {}
    footprint = cfg.get("footprint", {"x0": 0.0, "y0": 0.0, "x1": COURT_LENGTH, "y1": COURT_WIDTH})
    bounds = cfg.get("venue_bounds") or {}
    if "x0" not in bounds:
        bounds = {"x0": footprint["x0"] - 8, "x1": footprint["x1"] + 8,
                  "y0": footprint["y0"] - 8, "y1": footprint["y1"] + 8, "h": 12.0}
        bounds.update(cx=(bounds["x0"] + bounds["x1"]) / 2, cy=(bounds["y0"] + bounds["y1"]) / 2,
                      d=bounds["x1"] - bounds["x0"], w=bounds["y1"] - bounds["y0"])

    col = new_collection("VenueDetails")

    num_wall_banners = _build_wall_banners(col, bounds)
    _build_exit_signs(col, bounds)
    has_windows = _build_windows(col, bounds)
    has_trusses = _build_trusses(col, bounds)
    has_ducts = _build_ducts(col, bounds)
    has_hanging = _build_hanging_banners(col, bounds)
    has_housings = _build_light_housings(col)
    seating_type, spectator_count, placements = _build_bleacher_seating(col, bounds, footprint)
    _build_courtside_furniture(col)
    num_popup = _build_popup_banners(col)
    has_line_judges = _build_line_judge_chairs(col, cfg.get("lighting_preset"))
    clutter = _build_clutter(col, bounds)

    return {
        "num_wall_banners": num_wall_banners,
        "num_popup_banners": num_popup,
        "has_windows": bool(has_windows),
        "has_trusses": bool(has_trusses),
        "has_ducts": bool(has_ducts),
        "has_banners": bool(has_hanging),
        "has_light_housings": bool(has_housings),
        "seating_type": seating_type if placements else "none",
        "spectator_count": spectator_count,
        "has_line_judges": bool(has_line_judges),
        "clutter_items": clutter,
        "bleacher_placements": placements,
    }

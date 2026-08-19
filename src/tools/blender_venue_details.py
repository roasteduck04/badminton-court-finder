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
    "BleacherMat",
    "BleacherMatAlt",
    "ChairMat",
    "DrinkTableMat",
    "BottleMat",
    "ConeMat",
    "MopBucketMat",
    "RolledNetMat",
    "StackedChairMat",
    "LineJudgeChairMat",
    "LineJudgeSeatMat",
]
_FIXED_MATERIAL_PREFIXES = [
    "ExitSignMat_",
    "BannerMat_",
    "HangBannerMat_",
    "SpectatorMat_",
    "SpectatorSkin_",
    "LineJudgeMat_",
    "LineJudgeSkin_",
    "BagMat_",
    "TowelMat_",
    "PopupBannerMat_",
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


def _build_bleacher_seating(collection, bounds):
    """Tiered bleacher seating behind baselines, optionally along sideline."""
    seat_color_schemes = [
        ((0.15, 0.35, 0.70, 1.0), (0.85, 0.70, 0.10, 1.0)),  # blue + yellow
        ((0.20, 0.40, 0.75, 1.0), (0.90, 0.75, 0.15, 1.0)),  # lighter blue + yellow
        ((0.10, 0.30, 0.65, 1.0), (0.80, 0.65, 0.08, 1.0)),  # darker blue + gold
        ((0.15, 0.35, 0.70, 1.0), (0.70, 0.70, 0.72, 1.0)),  # blue + grey
    ]
    primary_color, secondary_color = random.choice(seat_color_schemes)
    mat = _make_mat("BleacherMat", primary_color, roughness=0.75)
    mat_alt = _make_mat("BleacherMatAlt", secondary_color, roughness=0.75)

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
                        seat_mat = mat if s % 2 == 0 else mat_alt
                        back.data.materials.append(seat_mat)
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


def _build_popup_banners(collection):
    """Pop-up A-frame sponsor banners around the court perimeter."""
    if random.random() > 0.7:
        return 0

    banner_colors = [
        (0.85, 0.20, 0.15, 1.0),   # red
        (0.15, 0.30, 0.70, 1.0),   # blue
        (0.90, 0.55, 0.10, 1.0),   # orange
        (0.10, 0.50, 0.25, 1.0),   # green
        (0.95, 0.95, 0.95, 1.0),   # white
        (0.50, 0.10, 0.55, 1.0),   # purple
    ]

    positions = []
    # Behind net on both sides
    for y_off in [-1.5, COURT_WIDTH + 1.5]:
        for x in [3.0, 6.7, 10.4]:
            positions.append((x, y_off, 0))
    # Behind baselines
    for x_off in [-1.5, COURT_LENGTH + 1.5]:
        positions.append((x_off, COURT_WIDTH / 2, math.pi / 2))

    num_banners = random.randint(3, min(8, len(positions)))
    chosen = random.sample(positions, k=num_banners)

    count = 0
    for bx, by, rot_z in chosen:
        color = random.choice(banner_colors)
        mat = _make_mat(f"PopupBannerMat_{count}", color, roughness=0.7)

        banner_w = random.uniform(0.6, 1.0)
        banner_h = random.uniform(0.5, 0.8)

        bpy.ops.mesh.primitive_plane_add(
            size=1,
            location=(bx, by, banner_h / 2),
        )
        obj = bpy.context.active_object
        obj.name = f"PopupBanner_{count}"
        obj.scale = (banner_w, 0.02, banner_h)
        obj.rotation_euler = (0, 0, rot_z)
        bpy.ops.object.transform_apply(scale=True, rotation=True)
        obj.data.materials.append(mat)
        _link_to_collection(obj, collection)
        count += 1

    return count


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

    # Pop-up courtside banners
    num_popup_banners = _build_popup_banners(col)

    # Line judge chairs
    has_line_judges = _build_line_judge_chairs(col, lighting_preset)

    # Clutter
    clutter_items = _build_clutter(col, bounds)

    return {
        "num_wall_banners": num_wall_banners,
        "num_popup_banners": num_popup_banners,
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

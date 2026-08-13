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

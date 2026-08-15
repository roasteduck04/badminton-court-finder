"""On-court occluders: players, umpire chair, equipment."""

import bpy
import random
from src.tools.blender_mannequin import build_mannequin

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
        for col_ in list(obj.users_collection):
            col_.objects.unlink(obj)
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

    for col_ in list(obj.users_collection):
        col_.objects.unlink(obj)
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
        meta = build_mannequin(col, i)
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

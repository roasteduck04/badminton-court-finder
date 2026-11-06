"""On-court occluders: players, umpire chair, equipment."""

import math
import random

from src.tools.blender_common import (
    COURT_LENGTH, COURT_WIDTH, NET_POS,
    add_box, add_cone, add_cylinder, add_sphere, make_material, new_collection,
)
from src.tools.blender_mannequin import build_mannequin, sample_player_positions

DEFAULT_CONFIG = {
    "umpire_chance": 0.25,
    "equipment_chance": 0.3,
    "no_occluder_chance": 0.2,
}


def _add_umpire_chair(collection):
    """Raised umpire chair beside the net post, in the inter-court gap."""
    neg = random.random() < 0.5
    y = -random.uniform(0.8, 1.3) if neg else COURT_WIDTH + random.uniform(0.8, 1.3)
    seat_z = random.uniform(1.2, 1.7)
    mat = make_material("UmpireChairMat", random.choice([
        (0.3, 0.3, 0.35, 1.0), (0.1, 0.2, 0.5, 1.0), (0.6, 0.6, 0.6, 1.0)]), metallic=0.6)
    for i, (dx, dy) in enumerate([(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]):
        add_box(f"UmpireChair_Leg_{i}", (NET_POS + dx, y + dy, seat_z / 2), (0.04, 0.04, seat_z),
                collection, mat)
    add_box("UmpireChair_Seat", (NET_POS, y, seat_z), (0.6, 0.6, 0.06), collection, mat)
    add_box("UmpireChair_Back", (NET_POS, y + (-0.3 if neg else 0.3), seat_z + 0.3),
            (0.6, 0.04, 0.6), collection, mat)
    for k in range(3):
        add_box(f"UmpireChair_Step_{k}", (NET_POS, y + (-0.45 if neg else 0.45), (k + 1) * seat_z / 4),
                (0.5, 0.25, 0.03), collection, mat)
    if random.random() < 0.6:
        add_cylinder("Umpire_Torso", 0.16, 0.6, (NET_POS, y, seat_z + 0.33), collection,
                     make_material("UmpireMat", random.choice([(0.1, 0.1, 0.2, 1.0), (0.2, 0.2, 0.22, 1.0)])))
        add_sphere("Umpire_Head", 0.11, (NET_POS, y, seat_z + 0.78), collection,
                   make_material("UmpireSkin", (0.65, 0.48, 0.35, 1.0)))
    return {"type": "umpire_chair", "position": [NET_POS, y, 0]}


def _add_equipment(collection, index):
    """A shuttlecock or racket lying on/near the court."""
    x = random.uniform(-0.5, COURT_LENGTH + 0.5)
    y = random.uniform(-0.5, COURT_WIDTH + 0.5)
    item = random.choice(["shuttlecock", "racket"])
    if item == "shuttlecock":
        add_cone(f"Equipment_{index}_shuttlecock", 0.033, 0.085, (x, y, 0.04), collection,
                 make_material(f"ShuttleMat_{index}", (0.95, 0.95, 0.92, 1.0)))
    else:
        add_box(f"Equipment_{index}_racket", (x, y, 0.01), (0.67, 0.22, 0.02), collection,
                make_material(f"RacketMat_{index}", (0.15, 0.15, 0.2, 1.0), metallic=0.3),
                rot_z=random.uniform(0, math.pi))
    return {"type": item, "position": [x, y, 0]}


def add_occluders(config=None):
    """Add players (per half), umpire chair and loose equipment.

    Returns:
        dict with "occluders" list of metadata dicts
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if random.random() < cfg["no_occluder_chance"]:
        return {"occluders": []}

    col = new_collection("Occluders")
    occluders = []
    for i, (x, y, facing) in enumerate(sample_player_positions()):
        occluders.append(build_mannequin(col, i, {"position": (x, y), "facing": facing}))
    if random.random() < cfg["umpire_chance"]:
        occluders.append(_add_umpire_chair(col))
    if random.random() < cfg["equipment_chance"]:
        for i in range(random.randint(1, 3)):
            occluders.append(_add_equipment(col, i))
    return {"occluders": occluders}

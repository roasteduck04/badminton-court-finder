"""Articulated mannequin players for badminton court scenes.

Every limb's object origin sits at its proximal joint (hip, knee, shoulder,
elbow) and the mesh hangs from it, so rotating a limb swings it about the
joint and children stay attached. Hierarchy:

    root (empty, on the floor, rotation_z = facing)
      pelvis (empty, hip height)
        torso (origin at hips, extends up) -> head, upper arms -> forearms -> racket
        upper legs -> lower legs -> shoes

Local frame: +Y = forward, +X = the player's left... sign conventions below
only matter for pose presets: for a limb hanging along -Z, a positive
rotation about X swings it forward (+Y).
"""

import math
import random

import bpy
import bmesh
from mathutils import Matrix, Vector

from src.tools.blender_common import COURT_LENGTH, COURT_WIDTH, NET_POS, make_material

POSE_PRESETS = ["standing", "ready", "lunging", "serving", "walking"]

SHIRT_COLORS = [
    (0.05, 0.05, 0.25, 1.0), (0.6, 0.1, 0.08, 1.0), (0.08, 0.4, 0.12, 1.0),
    (0.3, 0.3, 0.3, 1.0), (0.9, 0.9, 0.9, 1.0), (0.85, 0.75, 0.1, 1.0),
    (0.1, 0.3, 0.7, 1.0), (0.8, 0.4, 0.1, 1.0), (0.8, 0.4, 0.6, 1.0),
    (0.1, 0.5, 0.5, 1.0), (0.08, 0.08, 0.08, 1.0),
]
SHORTS_COLORS = [(0.08, 0.08, 0.08, 1.0), (0.05, 0.05, 0.25, 1.0),
                 (0.9, 0.9, 0.9, 1.0), (0.3, 0.3, 0.3, 1.0)]
SKIN_TONES = [(0.95, 0.82, 0.70, 1.0), (0.80, 0.62, 0.47, 1.0), (0.65, 0.48, 0.35, 1.0),
              (0.50, 0.35, 0.25, 1.0), (0.38, 0.26, 0.18, 1.0), (0.28, 0.18, 0.12, 1.0)]

# Segment lengths (m)
UPPER_LEG, LOWER_LEG, SHOE_H = 0.45, 0.43, 0.05
HIP_H = UPPER_LEG + LOWER_LEG + SHOE_H
TORSO_LEN, UPPER_ARM, FOREARM = 0.55, 0.30, 0.27


def _mesh_obj(name, bm, collection, mat, parent, location):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    if mat is not None:
        mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.parent = parent
    obj.location = location
    return obj


def _limb(name, radius, length, collection, mat, parent, joint, up=False):
    """Cylinder whose origin is the joint; hangs down (or extends up)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=radius,
                          radius2=radius * 0.85, depth=length)
    off = length / 2 if up else -length / 2
    bmesh.ops.translate(bm, verts=bm.verts, vec=Vector((0, 0, off)))
    if not up:  # taper toward the distal end
        bmesh.ops.rotate(bm, verts=bm.verts, cent=Vector((0, 0, off)),
                         matrix=Matrix.Rotation(math.pi, 3, 'X'))
    return _mesh_obj(name, bm, collection, mat, parent, joint)


def _sphere(name, radius, collection, mat, parent, location):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=radius)
    return _mesh_obj(name, bm, collection, mat, parent, location)


def _box(name, dims, collection, mat, parent, location):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, verts=bm.verts, vec=Vector(dims))
    return _mesh_obj(name, bm, collection, mat, parent, location)


def _empty(name, collection, parent, location):
    obj = bpy.data.objects.new(name, None)
    collection.objects.link(obj)
    obj.parent = parent
    obj.location = location
    return obj


def _j(deg, amount=7):
    return math.radians(deg + random.uniform(-amount, amount))


def _apply_pose(p, preset, mirror=False):
    """Joint rotations. Hanging limbs: +x = swing forward. Torso: -x = lean forward."""
    R, L = ("r", "l") if not mirror else ("l", "r")
    side = -1 if not mirror else 1  # abduction sign for the racket arm
    if preset == "standing":
        for s in "lr":
            p[f"upper_arm_{s}"].rotation_euler.x = _j(5)
            p[f"forearm_{s}"].rotation_euler.x = _j(10)
    elif preset == "ready":
        p["torso"].rotation_euler.x = -_j(15)
        for s in "lr":
            p[f"upper_arm_{s}"].rotation_euler.x = _j(35)
            p[f"forearm_{s}"].rotation_euler.x = _j(70)
            p[f"upper_leg_{s}"].rotation_euler.x = _j(25)
            p[f"lower_leg_{s}"].rotation_euler.x = -_j(40)
        p["upper_leg_l"].rotation_euler.y = _j(-8, 3)
        p["upper_leg_r"].rotation_euler.y = _j(8, 3)
    elif preset == "lunging":
        p["torso"].rotation_euler.x = -_j(20)
        p[f"upper_leg_{R}"].rotation_euler.x = _j(60)
        p[f"lower_leg_{R}"].rotation_euler.x = -_j(60)
        p[f"upper_leg_{L}"].rotation_euler.x = -_j(30)
        p[f"lower_leg_{L}"].rotation_euler.x = -_j(20)
        p[f"upper_arm_{R}"].rotation_euler.x = _j(70)
        p[f"upper_arm_{R}"].rotation_euler.y = side * _j(25)
        p[f"forearm_{R}"].rotation_euler.x = _j(20)
        p[f"upper_arm_{L}"].rotation_euler.x = -_j(30)
        p[f"forearm_{L}"].rotation_euler.x = _j(30)
    elif preset == "serving":
        p["torso"].rotation_euler.x = -_j(5)
        p[f"upper_arm_{R}"].rotation_euler.x = _j(-40)
        p[f"forearm_{R}"].rotation_euler.x = _j(60)
        p[f"upper_arm_{L}"].rotation_euler.x = _j(60)
        p[f"forearm_{L}"].rotation_euler.x = _j(30)
        p[f"upper_leg_{R}"].rotation_euler.x = _j(-10)
        p[f"upper_leg_{L}"].rotation_euler.x = _j(15)
        p[f"lower_leg_{L}"].rotation_euler.x = -_j(10)
    elif preset == "walking":
        ph = random.choice([1, -1])
        p["upper_arm_l"].rotation_euler.x = _j(20 * ph)
        p["upper_arm_r"].rotation_euler.x = _j(-20 * ph)
        p["forearm_l"].rotation_euler.x = _j(20)
        p["forearm_r"].rotation_euler.x = _j(20)
        p["upper_leg_l"].rotation_euler.x = _j(-20 * ph)
        p["upper_leg_r"].rotation_euler.x = _j(20 * ph)
        p["lower_leg_l"].rotation_euler.x = -_j(25 if ph > 0 else 5)
        p["lower_leg_r"].rotation_euler.x = -_j(25 if ph < 0 else 5)


def _world_bbox(objs):
    pts = []
    for o in objs:
        if o.type != 'MESH':
            continue
        mw = o.matrix_world
        pts.extend(mw @ Vector(c) for c in o.bound_box)
    return pts


def sample_player_positions(num_per_side=None):
    """Positions/facings per half: 1-2 players per side, never near the net.

    Returns list of (x, y, facing_radians).
    """
    if num_per_side is None:
        mode = random.random()
        if mode < 0.15:
            num_per_side = (random.randint(0, 1), random.randint(0, 1))
        elif mode < 0.6:
            num_per_side = (1, 1)
        elif mode < 0.9:
            num_per_side = (2, 2)
        else:
            num_per_side = (random.randint(1, 2), random.randint(1, 2))
    out = []
    for half, n in zip((0, 1), num_per_side):
        placed = []
        for _ in range(n):
            for _try in range(20):
                if half == 0:
                    x = random.uniform(-1.0, NET_POS - 1.2)
                else:
                    x = random.uniform(NET_POS + 1.2, COURT_LENGTH + 1.0)
                y = random.uniform(-0.6, COURT_WIDTH + 0.6)
                if all(math.hypot(x - px, y - py) > 1.0 for px, py in placed):
                    break
            placed.append((x, y))
            toward_net = 0.0 if half == 0 else math.pi
            out.append((x, y, toward_net + random.uniform(-math.radians(50), math.radians(50))))
    return out


def build_mannequin(collection, index, config=None):
    """Build an articulated mannequin player.

    Args:
        collection: target collection
        index: player index for naming
        config: optional dict with "position" (x, y) and "facing" (radians,
            0 = facing +x). Sampled anywhere on court if absent.

    Returns:
        metadata dict (type, position, pose, has_racket, facing_angle)
    """
    cfg = config or {}
    if "position" in cfg:
        x, y = cfg["position"]
    else:
        x, y, facing = sample_player_positions((1, 0))[0]
        cfg = dict(cfg, facing=cfg.get("facing", facing))
    facing = cfg.get("facing", random.uniform(0, 2 * math.pi))

    skin = make_material(f"Skin_{index}", random.choice(SKIN_TONES), roughness=0.6)
    shirt = make_material(f"Shirt_{index}", random.choice(SHIRT_COLORS))
    shorts = make_material(f"Shorts_{index}", random.choice(SHORTS_COLORS))
    shoe = make_material(f"Shoe_{index}", random.choice([(0.9, 0.9, 0.9, 1.0), (0.1, 0.1, 0.1, 1.0)]), 0.7)

    scale = random.uniform(0.9, 1.1)
    root = _empty(f"Player_{index}_Root", collection, None, (x, y, 0.0))
    root.rotation_euler.z = facing - math.pi / 2  # local +Y -> facing direction
    root.scale = (scale, scale, scale)
    pelvis = _empty(f"Player_{index}_Pelvis", collection, root, (0, 0, HIP_H))

    torso = _limb(f"Player_{index}_Torso", 0.17, TORSO_LEN, collection, shirt, pelvis, (0, 0, 0), up=True)
    head = _sphere(f"Player_{index}_Head", 0.11, collection, skin, torso, (0, 0, TORSO_LEN + 0.14))
    parts = {"torso": torso, "head": head}
    for s, sx in (("l", -1), ("r", 1)):
        ua = _limb(f"Player_{index}_UpperArm_{s.upper()}", 0.05, UPPER_ARM, collection, shirt,
                   torso, (sx * 0.23, 0, TORSO_LEN - 0.05))
        fa = _limb(f"Player_{index}_Forearm_{s.upper()}", 0.04, FOREARM, collection, skin, ua, (0, 0, -UPPER_ARM))
        ul = _limb(f"Player_{index}_UpperLeg_{s.upper()}", 0.075, UPPER_LEG, collection, shorts,
                   pelvis, (sx * 0.1, 0, 0))
        ll = _limb(f"Player_{index}_LowerLeg_{s.upper()}", 0.055, LOWER_LEG, collection, skin, ul, (0, 0, -UPPER_LEG))
        _box(f"Player_{index}_Shoe_{s.upper()}", (0.1, 0.26, SHOE_H), collection, shoe, ll,
             (0, 0.05, -LOWER_LEG - SHOE_H / 2))
        parts.update({f"upper_arm_{s}": ua, f"forearm_{s}": fa,
                      f"upper_leg_{s}": ul, f"lower_leg_{s}": ll})

    if random.random() < 0.6:
        hair = make_material(f"Hair_{index}", random.choice([
            (0.05, 0.03, 0.02, 1.0), (0.15, 0.10, 0.05, 1.0), (0.02, 0.02, 0.02, 1.0)]))
        _sphere(f"Player_{index}_Hair", 0.1, collection, hair, head, (0, -0.01, 0.04))

    mirror = random.random() < 0.15  # left-handed
    has_racket = random.random() < 0.7
    if has_racket:
        rmat = make_material(f"Racket_{index}", (0.15, 0.15, 0.2, 1.0), metallic=0.3)
        hand = parts["forearm_l" if mirror else "forearm_r"]
        handle = _limb(f"Player_{index}_RacketHandle", 0.012, 0.4, collection, rmat, hand, (0, 0, -FOREARM))
        _box(f"Player_{index}_RacketHead", (0.01, 0.22, 0.27), collection, rmat, handle, (0, 0, -0.4 - 0.13))

    pose = random.choice(POSE_PRESETS)
    _apply_pose(parts, pose, mirror)

    # Put the lowest point on the floor and keep the body clear of the net.
    bpy.context.view_layer.update()
    objs = [o for o in collection.objects if o.name.startswith(f"Player_{index}_")]
    pts = _world_bbox(objs)
    root.location.z -= min(p.z for p in pts)
    if x < NET_POS:
        excess = max(p.x for p in pts) - (NET_POS - 0.3)
        if excess > 0:
            root.location.x -= excess + 0.02
    else:
        excess = (NET_POS + 0.3) - min(p.x for p in pts)
        if excess > 0:
            root.location.x += excess + 0.02
    bpy.context.view_layer.update()

    return {
        "type": "player",
        "position": [round(root.location.x, 3), round(root.location.y, 3), 0],
        "pose": pose,
        "has_racket": has_racket,
        "facing_angle": round(facing, 3),
    }

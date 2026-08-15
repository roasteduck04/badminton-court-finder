"""Articulated mannequin players for badminton court scenes."""

import bpy
import math
import random

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1

POSE_PRESETS = ["standing", "ready", "lunging", "serving", "walking"]

SHIRT_COLORS = [
    (0.05, 0.05, 0.25, 1.0),   # navy
    (0.6, 0.1, 0.08, 1.0),     # red
    (0.08, 0.4, 0.12, 1.0),    # green
    (0.3, 0.3, 0.3, 1.0),      # grey
    (0.9, 0.9, 0.9, 1.0),      # white
    (0.85, 0.75, 0.1, 1.0),    # yellow
    (0.1, 0.3, 0.7, 1.0),      # blue
    (0.8, 0.4, 0.1, 1.0),      # orange
    (0.8, 0.4, 0.6, 1.0),      # pink
    (0.1, 0.5, 0.5, 1.0),      # teal
    (0.08, 0.08, 0.08, 1.0),   # black
]

SHORTS_COLORS = [
    (0.08, 0.08, 0.08, 1.0),   # black
    (0.05, 0.05, 0.25, 1.0),   # navy
    (0.9, 0.9, 0.9, 1.0),      # white
    (0.3, 0.3, 0.3, 1.0),      # grey
]

SKIN_TONES = [
    (0.95, 0.82, 0.70, 1.0),
    (0.80, 0.62, 0.47, 1.0),
    (0.65, 0.48, 0.35, 1.0),
    (0.50, 0.35, 0.25, 1.0),
    (0.38, 0.26, 0.18, 1.0),
    (0.28, 0.18, 0.12, 1.0),
]


def _make_mat(name, color, roughness=0.8):
    """Create or replace a simple material."""
    if name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[name])
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _add_cylinder(name, r, h, location, parent, collection, mat):
    """Add a cylinder body part parented to another object."""
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    if parent:
        obj.parent = parent
        obj.location = (
            location[0] - parent.location[0],
            location[1] - parent.location[1],
            location[2] - parent.location[2],
        )
    for col_ in list(obj.users_collection):
        col_.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def _add_sphere(name, r, location, parent, collection, mat):
    """Add a sphere body part parented to another object."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    if parent:
        obj.parent = parent
        obj.location = (
            location[0] - parent.location[0],
            location[1] - parent.location[1],
            location[2] - parent.location[2],
        )
    for col_ in list(obj.users_collection):
        col_.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def _add_cube(name, dims, location, parent, collection, mat):
    """Add a cube body part parented to another object."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (dims[0], dims[1], dims[2])
    bpy.ops.object.transform_apply(scale=True)
    obj.data.materials.append(mat)
    if parent:
        obj.parent = parent
        obj.location = (
            location[0] - parent.location[0],
            location[1] - parent.location[1],
            location[2] - parent.location[2],
        )
    for col_ in list(obj.users_collection):
        col_.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def _jitter(deg, amount=7):
    """Add random jitter to an angle in degrees."""
    return deg + random.uniform(-amount, amount)


def _apply_pose(parts, preset, mirror=False):
    """Apply joint rotations for a pose preset.

    parts dict keys: torso, head, upper_arm_l, upper_arm_r,
    forearm_l, forearm_r, upper_leg_l, upper_leg_r,
    lower_leg_l, lower_leg_r
    """
    sign = -1 if mirror else 1

    if preset == "standing":
        parts["upper_arm_l"].rotation_euler.x = math.radians(_jitter(5))
        parts["upper_arm_r"].rotation_euler.x = math.radians(_jitter(5))
        parts["forearm_l"].rotation_euler.x = math.radians(_jitter(10))
        parts["forearm_r"].rotation_euler.x = math.radians(_jitter(10))

    elif preset == "ready":
        parts["torso"].rotation_euler.x = math.radians(_jitter(5))
        parts["upper_arm_l"].rotation_euler.x = math.radians(_jitter(40))
        parts["upper_arm_r"].rotation_euler.x = math.radians(_jitter(40))
        parts["forearm_l"].rotation_euler.x = math.radians(_jitter(90))
        parts["forearm_r"].rotation_euler.x = math.radians(_jitter(90))
        parts["upper_leg_l"].rotation_euler.x = math.radians(_jitter(20))
        parts["upper_leg_r"].rotation_euler.x = math.radians(_jitter(20))
        parts["lower_leg_l"].rotation_euler.x = math.radians(_jitter(-30))
        parts["lower_leg_r"].rotation_euler.x = math.radians(_jitter(-30))

    elif preset == "lunging":
        parts["torso"].rotation_euler.x = math.radians(_jitter(15))
        # Lead leg
        lead_leg = "upper_leg_r" if not mirror else "upper_leg_l"
        trail_leg = "upper_leg_l" if not mirror else "upper_leg_r"
        lead_lower = "lower_leg_r" if not mirror else "lower_leg_l"
        trail_lower = "lower_leg_l" if not mirror else "lower_leg_r"
        parts[lead_leg].rotation_euler.x = math.radians(_jitter(70))
        parts[lead_lower].rotation_euler.x = math.radians(_jitter(-90))
        parts[trail_leg].rotation_euler.x = math.radians(_jitter(-10))
        parts[trail_lower].rotation_euler.x = math.radians(_jitter(-5))
        # Racket arm
        racket_arm = "upper_arm_r" if not mirror else "upper_arm_l"
        off_arm = "upper_arm_l" if not mirror else "upper_arm_r"
        racket_fore = "forearm_r" if not mirror else "forearm_l"
        off_fore = "forearm_l" if not mirror else "forearm_r"
        parts[racket_arm].rotation_euler.x = math.radians(_jitter(60))
        parts[racket_arm].rotation_euler.y = math.radians(_jitter(30 * sign))
        parts[racket_fore].rotation_euler.x = math.radians(_jitter(120))
        parts[off_arm].rotation_euler.x = math.radians(_jitter(-30))
        parts[off_fore].rotation_euler.x = math.radians(_jitter(45))

    elif preset == "serving":
        parts["torso"].rotation_euler.x = math.radians(_jitter(-10))
        parts["upper_arm_r"].rotation_euler.x = math.radians(_jitter(160))
        parts["forearm_r"].rotation_euler.x = math.radians(_jitter(90))
        parts["upper_arm_l"].rotation_euler.x = math.radians(_jitter(70))
        parts["forearm_l"].rotation_euler.x = math.radians(_jitter(10))
        parts["upper_leg_l"].rotation_euler.x = math.radians(_jitter(5))
        parts["upper_leg_r"].rotation_euler.x = math.radians(_jitter(-5))
        parts["lower_leg_l"].rotation_euler.x = math.radians(_jitter(-10))
        parts["lower_leg_r"].rotation_euler.x = math.radians(_jitter(-10))

    elif preset == "walking":
        phase = random.choice([1, -1])
        parts["upper_arm_l"].rotation_euler.x = math.radians(_jitter(15 * phase))
        parts["upper_arm_r"].rotation_euler.x = math.radians(_jitter(-15 * phase))
        parts["forearm_l"].rotation_euler.x = math.radians(_jitter(20))
        parts["forearm_r"].rotation_euler.x = math.radians(_jitter(20))
        parts["upper_leg_l"].rotation_euler.x = math.radians(_jitter(-20 * phase))
        parts["upper_leg_r"].rotation_euler.x = math.radians(_jitter(20 * phase))
        parts["lower_leg_l"].rotation_euler.x = math.radians(_jitter(-10 if phase > 0 else 0))
        parts["lower_leg_r"].rotation_euler.x = math.radians(_jitter(-10 if phase < 0 else 0))


def build_mannequin(collection, index, config=None):
    """Build an articulated mannequin player figure.

    Args:
        collection: Blender collection to add objects to
        index: player index for naming
        config: optional dict (unused for now, reserved for future overrides)

    Returns:
        dict with type, position, pose, has_racket, facing_angle metadata
    """
    # Position on court (Gaussian bias toward center)
    x = random.gauss(COURT_LENGTH / 2, COURT_LENGTH / 4)
    x = max(0.5, min(COURT_LENGTH - 0.5, x))
    y = random.uniform(0.5, COURT_WIDTH - 0.5)

    # Materials
    skin_tone = random.choice(SKIN_TONES)
    skin_mat = _make_mat(f"Skin_{index}", skin_tone, roughness=0.6)
    shirt_mat = _make_mat(f"Shirt_{index}", random.choice(SHIRT_COLORS))
    shorts_mat = _make_mat(f"Shorts_{index}", random.choice(SHORTS_COLORS))
    shoe_color = random.choice([(0.9, 0.9, 0.9, 1.0), (0.1, 0.1, 0.1, 1.0)])
    shoe_mat = _make_mat(f"Shoe_{index}", shoe_color, roughness=0.7)

    # Torso (root)
    torso = _add_cylinder(f"Player_{index}_Torso", 0.18, 0.50,
                          (x, y, 1.05), None, collection, shirt_mat)

    # Head
    head = _add_sphere(f"Player_{index}_Head", 0.11,
                       (x, y, 1.50), torso, collection, skin_mat)

    # Upper arms
    upper_arm_l = _add_cylinder(f"Player_{index}_UpperArm_L", 0.05, 0.28,
                                (x - 0.25, y, 1.20), torso, collection, shirt_mat)
    upper_arm_r = _add_cylinder(f"Player_{index}_UpperArm_R", 0.05, 0.28,
                                (x + 0.25, y, 1.20), torso, collection, shirt_mat)

    # Forearms
    forearm_l = _add_cylinder(f"Player_{index}_Forearm_L", 0.04, 0.25,
                              (x - 0.25, y, 0.93), upper_arm_l, collection, skin_mat)
    forearm_r = _add_cylinder(f"Player_{index}_Forearm_R", 0.04, 0.25,
                              (x + 0.25, y, 0.93), upper_arm_r, collection, skin_mat)

    # Upper legs
    upper_leg_l = _add_cylinder(f"Player_{index}_UpperLeg_L", 0.07, 0.40,
                                (x - 0.10, y, 0.60), torso, collection, shorts_mat)
    upper_leg_r = _add_cylinder(f"Player_{index}_UpperLeg_R", 0.07, 0.40,
                                (x + 0.10, y, 0.60), torso, collection, shorts_mat)

    # Lower legs
    lower_leg_l = _add_cylinder(f"Player_{index}_LowerLeg_L", 0.05, 0.38,
                                (x - 0.10, y, 0.21), upper_leg_l, collection, skin_mat)
    lower_leg_r = _add_cylinder(f"Player_{index}_LowerLeg_R", 0.05, 0.38,
                                (x + 0.10, y, 0.21), upper_leg_r, collection, skin_mat)

    # Shoes
    _add_cube(f"Player_{index}_Shoe_L", (0.12, 0.08, 0.05),
              (x - 0.10, y, 0.025), lower_leg_l, collection, shoe_mat)
    _add_cube(f"Player_{index}_Shoe_R", (0.12, 0.08, 0.05),
              (x + 0.10, y, 0.025), lower_leg_r, collection, shoe_mat)

    # Hair (~60% chance)
    if random.random() < 0.6:
        hair_color = random.choice([
            (0.05, 0.03, 0.02, 1.0),
            (0.15, 0.10, 0.05, 1.0),
            (0.02, 0.02, 0.02, 1.0),
        ])
        hair_mat = _make_mat(f"Hair_{index}", hair_color)
        _add_sphere(f"Player_{index}_Hair", 0.09,
                    (x, y, 1.60), head, collection, hair_mat)

    # Racket (~30% chance)
    has_racket = random.random() < 0.3
    if has_racket:
        racket_mat = bpy.data.materials.new(f"Racket_{index}")
        racket_mat.use_nodes = True
        bsdf = racket_mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.15, 0.15, 0.2, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.3

        bpy.ops.mesh.primitive_plane_add(size=1, location=(x + 0.25, y, 0.70))
        racket = bpy.context.active_object
        racket.name = f"Player_{index}_Racket"
        racket.scale = (0.25, 0.10, 1)
        bpy.ops.object.transform_apply(scale=True)
        racket.data.materials.append(racket_mat)
        racket.parent = forearm_r
        racket.location = (0, 0, -0.18)
        for col_ in list(racket.users_collection):
            col_.objects.unlink(racket)
        collection.objects.link(racket)

    # Apply pose
    pose = random.choice(POSE_PRESETS)
    mirror = random.choice([True, False])
    parts = {
        "torso": torso, "head": head,
        "upper_arm_l": upper_arm_l, "upper_arm_r": upper_arm_r,
        "forearm_l": forearm_l, "forearm_r": forearm_r,
        "upper_leg_l": upper_leg_l, "upper_leg_r": upper_leg_r,
        "lower_leg_l": lower_leg_l, "lower_leg_r": lower_leg_r,
    }
    _apply_pose(parts, pose, mirror)

    # Random Y-axis rotation biased toward net
    net_dir = math.atan2(COURT_WIDTH / 2 - y, COURT_LENGTH / 2 - x)
    facing = net_dir + random.uniform(-math.radians(30), math.radians(30))
    torso.rotation_euler.z = facing

    return {
        "type": "player",
        "position": [x, y, 0],
        "pose": pose,
        "has_racket": has_racket,
        "facing_angle": round(facing, 3),
    }

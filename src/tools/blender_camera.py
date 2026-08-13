"""Blender camera placement with named strategies.

Each strategy defines a region of valid camera positions for
different viewing angles of the badminton court.
"""

import bpy
import random
import math
from mathutils import Vector

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
COURT_CENTER = Vector((COURT_LENGTH / 2, COURT_WIDTH / 2, 0))

STRATEGIES = {
    "broadcast": {
        "x_range": (-6.0, -3.0),
        "y_range": (COURT_WIDTH / 2 - 1.5, COURT_WIDTH / 2 + 1.5),
        "z_range": (6.0, 10.0),
        "elevation_deg": (30, 45),
    },
    "sideline": {
        "x_range": (2.0, 11.0),
        "y_range": (-5.0, -2.0),
        "z_range": (2.0, 5.0),
        "elevation_deg": (15, 30),
    },
    "corner": {
        "x_range": (-4.0, -1.0),
        "y_range": (-4.0, -1.0),
        "z_range": (3.0, 7.0),
        "elevation_deg": (20, 40),
    },
    "overhead": {
        "x_range": (4.0, 9.0),
        "y_range": (1.0, 5.0),
        "z_range": (10.0, 16.0),
        "elevation_deg": (60, 85),
    },
    "low": {
        "x_range": (-3.0, -1.0),
        "y_range": (0.0, COURT_WIDTH),
        "z_range": (0.5, 1.5),
        "elevation_deg": (0, 10),
    },
}

FOCAL_LENGTH_RANGE = (28.0, 85.0)
LOOK_AT_JITTER = 1.5

DEFAULT_CONFIG = {
    "strategy": "broadcast",
    "resolution": (640, 640),
}


def _project_keypoints(scene, camera, keypoints, resolution):
    """Project 3D keypoint empties to 2D pixel coordinates via the camera.

    Returns:
        keypoints_3d: list of [x, y, z] world positions
        keypoints_2d: list of [px, py] pixel positions
        visibility: list of 0/1 (1 if within frame bounds)
    """
    from bpy_extras.object_utils import world_to_camera_view

    res_x, res_y = resolution
    keypoints_3d = []
    keypoints_2d = []
    visibility = []

    for empty in keypoints:
        world_pos = empty.matrix_world.translation
        keypoints_3d.append([world_pos.x, world_pos.y, world_pos.z])

        co = world_to_camera_view(scene, camera, world_pos)
        px = co.x * res_x
        py = (1.0 - co.y) * res_y  # Blender's Y is bottom-up, image Y is top-down

        in_frame = (0 <= co.x <= 1) and (0 <= co.y <= 1) and (co.z > 0)
        keypoints_2d.append([px, py])
        visibility.append(1 if in_frame else 0)

    return keypoints_3d, keypoints_2d, visibility


def place_camera(keypoints, strategy="broadcast", config=None):
    """Place a camera using the named strategy with randomization.

    Args:
        keypoints: list of 30 bpy empty objects from build_court()
        strategy: one of "broadcast", "sideline", "corner", "overhead",
                  "low", "random"
        config: dict with optional "resolution" (tuple of int)

    Returns:
        (camera_obj, metadata_dict)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    resolution = cfg["resolution"]

    if strategy == "random":
        strategy = random.choice(list(STRATEGIES.keys()))

    strat = STRATEGIES[strategy]

    # Sample camera position within strategy bounds
    x = random.uniform(*strat["x_range"])
    y = random.uniform(*strat["y_range"])
    z = random.uniform(*strat["z_range"])

    # Also allow mirroring: sideline can be on either side
    if strategy == "sideline" and random.random() < 0.5:
        y = COURT_WIDTH - y  # flip to other sideline
    if strategy == "corner":
        # Randomly pick one of the 4 corners
        corner_choice = random.randint(0, 3)
        if corner_choice == 1:
            x = COURT_LENGTH - x + COURT_LENGTH  # far baseline
            x = random.uniform(COURT_LENGTH + 1.0, COURT_LENGTH + 4.0)
        elif corner_choice == 2:
            y = COURT_WIDTH - y + COURT_WIDTH
            y = random.uniform(COURT_WIDTH + 1.0, COURT_WIDTH + 4.0)
        elif corner_choice == 3:
            x = random.uniform(COURT_LENGTH + 1.0, COURT_LENGTH + 4.0)
            y = random.uniform(COURT_WIDTH + 1.0, COURT_WIDTH + 4.0)

    # Random focal length
    focal_length = random.uniform(*FOCAL_LENGTH_RANGE)

    # Look-at target: court center with jitter
    look_at = COURT_CENTER.copy()
    look_at.x += random.uniform(-LOOK_AT_JITTER, LOOK_AT_JITTER)
    look_at.y += random.uniform(-LOOK_AT_JITTER, LOOK_AT_JITTER)

    # Clear existing camera
    for obj in list(bpy.data.objects):
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)
    for cam in list(bpy.data.cameras):
        if cam.users == 0:
            bpy.data.cameras.remove(cam)

    # Create camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = focal_length
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_obj.location = (x, y, z)
    direction = look_at - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    # Set render resolution
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]

    # Update scene to ensure matrices are current
    bpy.context.view_layer.update()

    # Project keypoints
    kp_3d, kp_2d, vis = _project_keypoints(
        bpy.context.scene, cam_obj, keypoints, resolution,
    )

    metadata = {
        "strategy": strategy,
        "position": [x, y, z],
        "rotation": list(cam_obj.rotation_euler),
        "focal_length": focal_length,
        "keypoints_3d": kp_3d,
        "keypoints_2d": kp_2d,
        "visibility": vis,
    }

    return cam_obj, metadata

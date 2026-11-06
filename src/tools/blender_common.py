"""Shared constants and helpers for the Blender synthetic data pipeline.

Blender's bundled Python has numpy but not cv2, so ``src.court_geometry``
cannot be imported inside Blender. The court constants are mirrored here
(pure Python) and ``tests/test_blender_common.py`` asserts they equal the
values in ``src/court_geometry.py``.

``bpy`` is imported lazily so this module (constants, seeding, homography
support check) is importable from the project Python as well.
"""

import hashlib
import math

try:  # pragma: no cover - only available inside Blender
    import bpy
    import bmesh
except ImportError:  # project Python: constants/helpers only
    bpy = None
    bmesh = None

# --- BWF court dimensions (metres), mirrored from src/court_geometry.py ---
COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
SINGLES_OFFSET = 0.46
NET_POS = 6.7
SHORT_SERVICE = 1.98
LONG_SERVICE_DBL = 0.76
NUM_KEYPOINTS = 30

# Rows along the court length (x), columns across the width (y).
ROWS_X = [0.0, LONG_SERVICE_DBL, NET_POS - SHORT_SERVICE,
          NET_POS + SHORT_SERVICE, COURT_LENGTH - LONG_SERVICE_DBL, COURT_LENGTH]
COLS_Y = [0.0, SINGLES_OFFSET, COURT_WIDTH / 2,
          COURT_WIDTH - SINGLES_OFFSET, COURT_WIDTH]

# K = row * 5 + col, at template (x along length, y along width) - same order
# as court_geometry.COURT_KEYPOINTS_TEMPLATE. World (x, y) == template (x, y).
KEYPOINTS_TEMPLATE = [(x, y) for x in ROWS_X for y in COLS_Y]
CORNER_INDICES = [0, 4, 25, 29]

LINE_WIDTH = 0.04          # BWF: 40 mm
LINE_Z = 0.001             # painted lines / keypoints sit just above the floor
NET_HEIGHT_EDGE = 1.55     # top of net at the posts
NET_HEIGHT_CENTER = 1.524  # top of net at the centre
NET_DEPTH = 0.76           # top-to-bottom height of the net mesh
COURT_GAP = 2.0            # spacing between adjacent courts

# Object custom property: False = never counts as an occluder when ray casting
# keypoint visibility (floor, painted lines, decals, nets).
OCCLUDER_PROP = "cvn_occluder"


# --------------------------------------------------------------------------
# Pure-Python helpers (usable outside Blender)
# --------------------------------------------------------------------------

def image_seed(global_seed, image_index):
    """Per-image seed = hash(global_seed, image_index), 31-bit, stable."""
    digest = hashlib.sha256(f"cvn:{int(global_seed)}:{int(image_index)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def has_homography_support(visible):
    """True if the visible template points determine a homography.

    Same rule as court_geometry.has_homography_support: >= 4 points and the
    largest collinear subset leaves at least 2 points out.
    """
    pts = [KEYPOINTS_TEMPLATE[k] for k, v in enumerate(visible) if v]
    n = len(pts)
    if n < 4:
        return False
    best = 2
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]
            cnt = 0
            for k in range(n):
                ex, ey = pts[k][0] - pts[i][0], pts[k][1] - pts[i][1]
                if abs(dx * ey - dy * ex) < 1e-9:
                    cnt += 1
            best = max(best, cnt)
    return best <= n - 2


def log_uniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------
# Blender helpers
# --------------------------------------------------------------------------

def link_to_collection(obj, collection):
    """Move ``obj`` into ``collection`` only (users_collection pattern)."""
    for col in list(obj.users_collection):
        col.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def new_collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def reset_scene():
    """Remove every object, collection and data-block from the file.

    Called before each image so nothing leaks between frames (including the
    default startup cube/camera/light when not launched with
    --factory-startup).
    """
    scene = bpy.context.scene
    bpy.data.batch_remove(list(bpy.data.objects))
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.images, bpy.data.textures,
                 bpy.data.node_groups, bpy.data.curves):
        items = [x for x in coll if getattr(x, "type", None) not in ("RENDER_RESULT", "COMPOSITING")]
        if items:
            bpy.data.batch_remove(items)
    for w in list(bpy.data.worlds):
        if scene.world is not w:
            bpy.data.worlds.remove(w)
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def make_material(name, color, roughness=0.7, metallic=0.0, emission=0.0):
    """New Principled BSDF material (Blender makes the name unique)."""
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


def wood_planks_socket(nodes, links, base_rgba, rng):
    """Build a plank-floor colour (brick planks + stretched grain); return the socket."""
    tex = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, rng.choice([0.0, math.pi / 2]))
    links.new(tex.outputs["Object"], mapping.inputs["Vector"])

    def shade(k):
        return tuple(min(1.0, c * k) for c in base_rgba[:3]) + (1.0,)

    brick = nodes.new("ShaderNodeTexBrick")
    brick.offset = rng.uniform(0.3, 0.7)
    brick.inputs["Color1"].default_value = shade(rng.uniform(0.85, 1.0))
    brick.inputs["Color2"].default_value = shade(rng.uniform(1.0, 1.2))
    brick.inputs["Mortar"].default_value = shade(0.55)
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Mortar Size"].default_value = 0.002
    brick.inputs["Brick Width"].default_value = rng.uniform(1.0, 2.5)
    brick.inputs["Row Height"].default_value = rng.uniform(0.07, 0.15)
    links.new(mapping.outputs["Vector"], brick.inputs["Vector"])

    grain_map = nodes.new("ShaderNodeMapping")
    grain_map.inputs["Scale"].default_value = (2.0, 60.0, 2.0)
    links.new(mapping.outputs["Vector"], grain_map.inputs["Vector"])
    grain = nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = rng.uniform(1.0, 3.0)
    grain.inputs["Detail"].default_value = 6.0
    links.new(grain_map.outputs["Vector"], grain.inputs["Vector"])
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = 'OVERLAY'
    mix.inputs["Fac"].default_value = rng.uniform(0.1, 0.3)
    links.new(brick.outputs["Color"], mix.inputs["Color1"])
    links.new(grain.outputs["Color"], mix.inputs["Color2"])
    return mix.outputs[0]


def _mesh_object(name, verts, faces, collection, mat=None, occluder=True):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    if mat is not None:
        mesh.materials.append(mat)
    if not occluder:
        obj[OCCLUDER_PROP] = False
    return obj


def add_box(name, center, dims, collection, mat=None, rot_z=0.0, occluder=True):
    """Axis-aligned (then rotated about Z) box with real thickness.

    dims = (dx, dy, dz) full extents in metres. Use this instead of a scaled
    primitive plane for anything upright: a plane has no z extent, so
    scaling it in z collapses it.
    """
    hx, hy, hz = dims[0] / 2, dims[1] / 2, dims[2] / 2
    c, s = math.cos(rot_z), math.sin(rot_z)
    verts = []
    for z in (-hz, hz):
        for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            verts.append((center[0] + c * x - s * y, center[1] + s * x + c * y, center[2] + z))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return _mesh_object(name, verts, faces, collection, mat, occluder)


def add_floor_quad(name, p1, p2, width, z, collection, mat=None, occluder=False):
    """Flat strip on the floor from p1 to p2 (x, y) with the given width."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    nx, ny = -dy / length * width / 2, dx / length * width / 2
    verts = [(p1[0] + nx, p1[1] + ny, z), (p1[0] - nx, p1[1] - ny, z),
             (p2[0] - nx, p2[1] - ny, z), (p2[0] + nx, p2[1] + ny, z)]
    # Counter-clockwise seen from above -> normal +Z
    if (verts[1][0] - verts[0][0]) * (verts[2][1] - verts[0][1]) - \
            (verts[1][1] - verts[0][1]) * (verts[2][0] - verts[0][0]) < 0:
        verts = verts[::-1]
    return _mesh_object(name, verts, [(0, 1, 2, 3)], collection, mat, occluder)


def add_floor_rect(name, x0, y0, x1, y1, z, collection, mat=None, occluder=False):
    """Flat axis-aligned rectangle on the floor (normal +Z)."""
    verts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    return _mesh_object(name, verts, [(0, 1, 2, 3)], collection, mat, occluder)


def add_cylinder(name, radius, depth, location, collection, mat=None,
                 rotation=(0.0, 0.0, 0.0), segments=16, occluder=True):
    """Cylinder via bmesh (no operator context needed)."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments,
                          radius1=radius, radius2=radius, depth=depth)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.rotation_euler = rotation
    collection.objects.link(obj)
    if mat is not None:
        mesh.materials.append(mat)
    if not occluder:
        obj[OCCLUDER_PROP] = False
    return obj


def add_sphere(name, radius, location, collection, mat=None, occluder=True):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=radius)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    collection.objects.link(obj)
    if mat is not None:
        mesh.materials.append(mat)
    if not occluder:
        obj[OCCLUDER_PROP] = False
    return obj


def add_cone(name, radius, depth, location, collection, mat=None):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=12,
                          radius1=radius, radius2=0.0, depth=depth)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    collection.objects.link(obj)
    if mat is not None:
        mesh.materials.append(mat)
    return obj

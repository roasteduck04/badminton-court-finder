"""Blender lighting presets for indoor badminton venues."""

import math
import random

import bpy

from src.tools.blender_common import COURT_LENGTH, COURT_WIDTH, new_collection

PRESETS = ["fluorescent", "mixed", "dim", "harsh", "competition"]

DEFAULT_CONFIG = {
    "intensity_jitter": 0.2,
    "temp_jitter": 300,
    "court_offsets": [],   # adjacent courts also get overhead lights
    "ceiling_height": 12.0,
}


def _kelvin_to_rgb(temp):
    """Approximate colour temperature to RGB (simplified)."""
    t = temp / 100.0
    if t <= 66:
        r = 1.0
        g = max(0, min(1, 0.39 * math.log(t) - 0.63))
        b = max(0, min(1, 0.54 * math.log(t - 10) - 1.19)) if t > 19 else 0
    else:
        r = max(0, min(1, 1.29 * ((t - 60) ** -0.13)))
        g = max(0, min(1, 1.13 * ((t - 60) ** -0.08)))
        b = 1.0
    return (r, g, b)


class _Ctx:
    def __init__(self, collection, cfg):
        self.col = collection
        self.cfg = cfg
        self.lights = []
        self.max_z = cfg["ceiling_height"] - 0.4

    def area(self, name, pos, energy, size, color):
        pos = (pos[0], pos[1], min(pos[2], self.max_z))
        data = bpy.data.lights.new(name, type='AREA')
        data.energy = energy
        data.size = size
        data.color = color
        # Only lights over the main court cast shadows: keeps EEVEE's shadow
        # pool from overflowing in large multi-court halls.
        data.use_shadow = ("Ambient" not in name and -3.0 <= pos[0] <= COURT_LENGTH + 3.0
                           and -3.0 <= pos[1] <= COURT_WIDTH + 3.0)
        obj = bpy.data.objects.new(name, data)
        obj.location = pos
        self.col.objects.link(obj)
        self.lights.append({"name": name, "position": [round(p, 3) for p in pos],
                            "energy": round(energy, 1), "color": [round(c, 3) for c in color]})
        return obj

    def jit(self, base):
        j = self.cfg["intensity_jitter"]
        return base * (1 + random.uniform(-j, j))

    def courts(self):
        """Main court first, then adjacent courts (fewer lights on those)."""
        return [(0.0, 0.0, True)] + [(x, y, False) for x, y in self.cfg["court_offsets"]]


def _build_fluorescent(ctx):
    color = _kelvin_to_rgb(5200 + random.uniform(-ctx.cfg["temp_jitter"], ctx.cfg["temp_jitter"]))
    z = random.uniform(7.0, 10.0)
    for ci, (xo, yo, main) in enumerate(ctx.courts()):
        fracs = [0.25, 0.5, 0.75] if main else [0.3, 0.7]
        for i, fx in enumerate(fracs):
            ctx.area(f"Light_Fluoro_{ci}_{i}", (xo + COURT_LENGTH * fx + random.uniform(-0.5, 0.5),
                                                yo + COURT_WIDTH / 2 + random.uniform(-0.3, 0.3), z),
                     ctx.jit(450), 3.0, color)


def _build_mixed(ctx):
    warm = _kelvin_to_rgb(4500 + random.uniform(-200, 200))
    for ci, (xo, yo, main) in enumerate(ctx.courts()):
        for i, fx in enumerate([0.25, 0.5, 0.75] if main else [0.5]):
            ctx.area(f"Light_Mixed_{ci}_{i}", (xo + COURT_LENGTH * fx, yo + COURT_WIDTH / 2, 8.0),
                     ctx.jit(320), 3.0, warm)
    data = bpy.data.lights.new("Light_Mixed_Sun", type='SUN')
    data.energy = ctx.jit(random.uniform(1.0, 3.0))
    data.color = _kelvin_to_rgb(6500)
    sun = bpy.data.objects.new("Light_Mixed_Sun", data)
    sun.location = (0, 0, 10)
    sun.rotation_euler = (math.radians(random.uniform(35, 65)), 0, random.uniform(0, 2 * math.pi))
    ctx.col.objects.link(sun)
    ctx.lights.append({"name": sun.name, "position": [0, 0, 10],
                       "energy": round(data.energy, 2), "color": list(_kelvin_to_rgb(6500))})


def _build_dim(ctx):
    color = _kelvin_to_rgb(random.uniform(3200, 4200))
    for ci, (xo, yo, main) in enumerate(ctx.courts()):
        pts = [(0.33, 0.5), (0.67, 0.5)] if main else [(0.5, 0.5)]
        for i, (fx, fy) in enumerate(pts):
            ctx.area(f"Light_Dim_{ci}_{i}", (xo + COURT_LENGTH * fx, yo + COURT_WIDTH * fy, 7.0),
                     ctx.jit(300), 2.0, color)


def _build_harsh(ctx):
    color = _kelvin_to_rgb(5500)
    side = random.choice([0.15, 0.85])
    ctx.area("Light_Harsh_Key", (COURT_LENGTH / 2 + random.uniform(-3, 3), COURT_WIDTH * side, 9.0),
             ctx.jit(900), 1.0, color)
    ctx.area("Light_Harsh_Fill", (COURT_LENGTH / 2, COURT_WIDTH * (1 - side), 8.0),
             ctx.jit(150), 3.0, _kelvin_to_rgb(5000))
    for ci, (xo, yo, main) in enumerate(ctx.courts()):
        if not main:
            ctx.area(f"Light_Harsh_Adj_{ci}", (xo + COURT_LENGTH / 2, yo + COURT_WIDTH / 2, 9.0),
                     ctx.jit(400), 1.0, color)


def _build_competition(ctx):
    color = _kelvin_to_rgb(5600 + random.uniform(-150, 150))
    for ci, (xo, yo, main) in enumerate(ctx.courts()):
        pts = ([(fx, fy) for fx in (0.2, 0.5, 0.8) for fy in (0.2, 0.8)] if main
               else [(0.3, 0.5), (0.7, 0.5)])
        for i, (fx, fy) in enumerate(pts):
            ctx.area(f"Light_Comp_{ci}_{i}", (xo + COURT_LENGTH * fx + random.uniform(-0.3, 0.3),
                                              yo + COURT_WIDTH * fy + random.uniform(-0.3, 0.3), 10.0),
                     ctx.jit(550), 4.0, color)


PRESET_BUILDERS = {
    "fluorescent": _build_fluorescent,
    "mixed": _build_mixed,
    "dim": _build_dim,
    "harsh": _build_harsh,
    "competition": _build_competition,
}

# Ambient fill per preset (relative): keeps frames from going black without
# flattening the character of dim/harsh presets.
AMBIENT_SCALE = {"fluorescent": 1.0, "mixed": 0.8, "dim": 0.25, "harsh": 0.3, "competition": 0.6}


def setup_lighting(preset="fluorescent", config=None):
    """Set up venue lighting using a named preset (or "random").

    Returns:
        dict with "preset", "lights", "ambient_energy", "world_strength".
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if preset == "random":
        preset = random.choice(PRESETS)

    ctx = _Ctx(new_collection("Lighting"), cfg)
    PRESET_BUILDERS[preset](ctx)

    ambient_energy = random.uniform(60, 180) * AMBIENT_SCALE[preset]
    ctx.area("Light_Ambient", (COURT_LENGTH / 2, COURT_WIDTH / 2, cfg["ceiling_height"] - 0.5),
             ambient_energy, 10.0, _kelvin_to_rgb(5000))

    # World background = weak omnidirectional bounce light.
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    strength = random.uniform(0.15, 0.6) * (0.5 if preset in ("dim", "harsh") else 1.0)
    if bg:
        g = random.uniform(0.08, 0.16)
        bg.inputs["Color"].default_value = (g, g, g * random.uniform(0.9, 1.1), 1.0)
        bg.inputs["Strength"].default_value = strength

    return {"preset": preset, "lights": ctx.lights,
            "ambient_energy": round(ambient_energy, 1), "world_strength": round(strength, 3)}

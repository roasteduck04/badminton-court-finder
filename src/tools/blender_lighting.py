"""Blender lighting presets for indoor badminton venues."""

import bpy
import random
import math

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1

PRESETS = ["fluorescent", "mixed", "dim", "harsh", "competition"]

DEFAULT_CONFIG = {
    "preset": "fluorescent",
    "intensity_jitter": 0.2,
    "temp_jitter": 300,
}


def _clear_lights():
    """Remove all existing lights and their collection."""
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    for light in list(bpy.data.lights):
        if light.users == 0:
            bpy.data.lights.remove(light)
    if "Lighting" in bpy.data.collections:
        col = bpy.data.collections["Lighting"]
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)


def _kelvin_to_rgb(temp):
    """Approximate color temperature to RGB (simplified)."""
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


def _add_area_light(name, location, energy, size, color, collection):
    """Add an area light to the scene."""
    bpy.ops.object.light_add(type='AREA', location=location)
    light = bpy.context.active_object
    light.name = name
    light.data.energy = energy
    light.data.size = size
    light.data.color = color

    if light.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(light)
    collection.objects.link(light)

    return light


def _build_fluorescent(collection, config):
    """Multiple evenly-spaced rectangular area lights, high-mounted."""
    base_temp = 5200 + random.uniform(-config["temp_jitter"], config["temp_jitter"])
    color = _kelvin_to_rgb(base_temp)
    jitter = config["intensity_jitter"]
    lights = []

    positions = [
        (COURT_LENGTH * 0.25, COURT_WIDTH / 2, 9.0),
        (COURT_LENGTH * 0.5, COURT_WIDTH / 2, 9.0),
        (COURT_LENGTH * 0.75, COURT_WIDTH / 2, 9.0),
        (COURT_LENGTH * 0.25, COURT_WIDTH * 0.15, 8.5),
        (COURT_LENGTH * 0.75, COURT_WIDTH * 0.85, 8.5),
    ]
    for i, pos in enumerate(positions):
        jittered_pos = (
            pos[0] + random.uniform(-0.5, 0.5),
            pos[1] + random.uniform(-0.3, 0.3),
            pos[2] + random.uniform(-0.3, 0.3),
        )
        energy = 500 * (1 + random.uniform(-jitter, jitter))
        light = _add_area_light(f"Light_Fluoro_{i}", jittered_pos, energy, 4.0, color, collection)
        lights.append({"name": light.name, "position": list(jittered_pos), "energy": energy, "color": list(color)})

    return lights


def _build_mixed(collection, config):
    """Overhead lights plus angled sun for window light."""
    jitter = config["intensity_jitter"]
    lights = []

    # Overhead
    warm = _kelvin_to_rgb(4500 + random.uniform(-200, 200))
    for i in range(3):
        x = COURT_LENGTH * (0.25 + 0.25 * i)
        pos = (x, COURT_WIDTH / 2, 8.0)
        energy = 350 * (1 + random.uniform(-jitter, jitter))
        light = _add_area_light(f"Light_Mixed_Over_{i}", pos, energy, 3.0, warm, collection)
        lights.append({"name": light.name, "position": list(pos), "energy": energy, "color": list(warm)})

    # Sun lamp (window light)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.name = "Light_Mixed_Sun"
    sun.data.energy = 2.0 * (1 + random.uniform(-jitter, jitter))
    sun.data.color = _kelvin_to_rgb(6500)
    sun.rotation_euler = (math.radians(50), math.radians(random.uniform(-30, 30)), 0)
    if sun.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(sun)
    collection.objects.link(sun)
    sun_color = list(_kelvin_to_rgb(6500))
    lights.append({"name": sun.name, "position": [0, 0, 10], "energy": sun.data.energy, "color": sun_color})

    return lights


def _build_dim(collection, config):
    """Fewer, weaker lights with visible falloff."""
    jitter = config["intensity_jitter"]
    color = _kelvin_to_rgb(4000 + random.uniform(-200, 200))
    lights = []

    positions = [
        (COURT_LENGTH * 0.33, COURT_WIDTH / 2, 7.0),
        (COURT_LENGTH * 0.67, COURT_WIDTH / 2, 7.0),
    ]
    for i, pos in enumerate(positions):
        energy = 200 * (1 + random.uniform(-jitter, jitter))
        light = _add_area_light(f"Light_Dim_{i}", pos, energy, 2.5, color, collection)
        lights.append({"name": light.name, "position": list(pos), "energy": energy, "color": list(color)})

    return lights


def _build_harsh(collection, config):
    """Single strong directional light creating sharp shadows."""
    jitter = config["intensity_jitter"]
    color = _kelvin_to_rgb(5500)
    lights = []

    side = random.choice(["left", "right"])
    y = -3.0 if side == "left" else COURT_WIDTH + 3.0
    pos = (COURT_LENGTH / 2 + random.uniform(-2, 2), y, 8.0)
    energy = 1200 * (1 + random.uniform(-jitter, jitter))
    light = _add_area_light("Light_Harsh", pos, energy, 1.0, color, collection)
    lights.append({"name": light.name, "position": list(pos), "energy": energy, "color": list(color)})

    return lights


def _build_competition(collection, config):
    """Bright, even, multi-position lighting with minimal shadows."""
    jitter = config["intensity_jitter"]
    color = _kelvin_to_rgb(5500 + random.uniform(-100, 100))
    lights = []

    positions = [
        (COURT_LENGTH * 0.2, COURT_WIDTH * 0.2, 10.0),
        (COURT_LENGTH * 0.2, COURT_WIDTH * 0.8, 10.0),
        (COURT_LENGTH * 0.5, COURT_WIDTH * 0.2, 10.0),
        (COURT_LENGTH * 0.5, COURT_WIDTH * 0.8, 10.0),
        (COURT_LENGTH * 0.8, COURT_WIDTH * 0.2, 10.0),
        (COURT_LENGTH * 0.8, COURT_WIDTH * 0.8, 10.0),
    ]
    for i, pos in enumerate(positions):
        jittered_pos = (
            pos[0] + random.uniform(-0.3, 0.3),
            pos[1] + random.uniform(-0.3, 0.3),
            pos[2],
        )
        energy = 600 * (1 + random.uniform(-jitter, jitter))
        light = _add_area_light(f"Light_Comp_{i}", jittered_pos, energy, 5.0, color, collection)
        lights.append({"name": light.name, "position": list(jittered_pos), "energy": energy, "color": list(color)})

    return lights


PRESET_BUILDERS = {
    "fluorescent": _build_fluorescent,
    "mixed": _build_mixed,
    "dim": _build_dim,
    "harsh": _build_harsh,
    "competition": _build_competition,
}


def setup_lighting(preset="fluorescent", config=None):
    """Set up venue lighting using a named preset.

    Args:
        preset: one of "fluorescent", "mixed", "dim", "harsh",
                "competition", or "random"
        config: dict with optional intensity_jitter, temp_jitter

    Returns:
        dict with "preset" and "lights" metadata
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    if preset == "random":
        preset = random.choice(PRESETS)

    _clear_lights()

    col = bpy.data.collections.new("Lighting")
    bpy.context.scene.collection.children.link(col)

    builder = PRESET_BUILDERS[preset]
    lights = builder(col, cfg)

    return {"preset": preset, "lights": lights}

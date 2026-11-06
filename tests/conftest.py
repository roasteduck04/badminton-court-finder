"""Shared pytest configuration.

Makes ``python -m pytest tests/`` work from the repo root without Blender installed:
``test_blender_smoke.py`` imports ``bpy`` at module level and is meant to be run inside Blender
(``blender --background --python tests/test_blender_smoke.py``), so it is skipped at collection
time when ``bpy`` is unavailable.
"""

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

collect_ignore = []
if importlib.util.find_spec("bpy") is None:
    collect_ignore.append("test_blender_smoke.py")

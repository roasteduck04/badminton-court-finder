"""Generate synthetic badminton court data or convert to CVN format.

Usage:
    python scripts/generate.py render [OPTIONS]
    python scripts/generate.py convert [OPTIONS]

Render options:
    --count N       Number of images (default: 500)
    --seed N        Random seed, -1 for random (default: 42)
    --engine STR    BLENDER_EEVEE or CYCLES (default: BLENDER_EEVEE)
    --samples N     Render samples (default: 32)
    --res-min N     Min resolution (default: 800)
    --res-max N     Max resolution (default: 1280)
    --start N       Starting image index (default: 1)
    --blender PATH  Path to Blender executable

Convert options:
    --min-visible N  Min visible keypoints (default: 4)
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"


def find_blender(user_path=None):
    if user_path:
        if os.path.isfile(user_path):
            return user_path
        print(f"Error: Blender not found at {user_path}")
        sys.exit(1)

    if os.path.isfile(DEFAULT_BLENDER):
        return DEFAULT_BLENDER

    for name in ["blender", "blender.exe"]:
        for d in os.environ.get("PATH", "").split(os.pathsep):
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate):
                return candidate

    print("Error: Blender not found. Install it or pass --blender PATH")
    sys.exit(1)


def cmd_render(args):
    blender = find_blender(args.blender)
    render_script = os.path.join(ROOT, "src", "tools", "blender_render.py")

    cmd = [
        blender,
        "--background",
        "--python", render_script,
        "--",
        "--count", str(args.count),
        "--seed", str(args.seed),
        "--engine", args.engine,
        "--samples", str(args.samples),
        "--res-min", str(args.res_min),
        "--res-max", str(args.res_max),
        "--start", str(args.start),
    ]

    print(f"Blender:  {blender}")
    print(f"Count:    {args.count}")
    print(f"Seed:     {args.seed}")
    print(f"Engine:   {args.engine}")
    print(f"Samples:  {args.samples}")
    print(f"Res:      {args.res_min}-{args.res_max}")
    print(f"Start:    {args.start}")
    print()

    return subprocess.call(cmd)


def cmd_convert(args):
    convert_script = os.path.join(ROOT, "src", "tools", "blender_to_cvn.py")

    cmd = [sys.executable, convert_script, "--min-visible", str(args.min_visible)]

    print(f"Min visible: {args.min_visible}")
    print()

    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic badminton court data"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="Render images with Blender")
    render.add_argument("--count", type=int, default=500)
    render.add_argument("--seed", type=int, default=42)
    render.add_argument("--engine", default="BLENDER_EEVEE")
    render.add_argument("--samples", type=int, default=32)
    render.add_argument("--res-min", type=int, default=800)
    render.add_argument("--res-max", type=int, default=1280)
    render.add_argument("--start", type=int, default=1)
    render.add_argument("--blender", help="Path to Blender executable")

    convert = sub.add_parser("convert", help="Convert raw output to CVN format")
    convert.add_argument("--min-visible", type=int, default=4)

    args = parser.parse_args()

    if args.command == "render":
        sys.exit(cmd_render(args))
    elif args.command == "convert":
        sys.exit(cmd_convert(args))


if __name__ == "__main__":
    main()

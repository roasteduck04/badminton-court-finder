"""Generate synthetic badminton court data, convert it to CVN, or augment it.

Usage:
    python scripts/generate.py render [OPTIONS]
    python scripts/generate.py convert [OPTIONS]
    python scripts/generate.py augment [OPTIONS]

Render options (defaults match src/tools/blender_render.py):
    --count N         Number of images (default: 500)
    --seed N          Global seed, -1 for random (default: 42). Each image's
                      seed = hash(seed, image index), so --start ranges never collide.
    --engine STR      BLENDER_EEVEE or CYCLES (default: BLENDER_EEVEE)
    --samples N       Render samples (EEVEE TAA or Cycles) (default: 32)
    --res-min N       Min SHORT image side in px, >= 480 (default: 480)
    --res-max N       Max SHORT image side in px (default: 720)
    --aspects LIST    Comma list of 16:9,9:16,4:3,3:4,1:1 (default: 16:9,9:16,4:3,1:1)
    --strategies LIST Restrict camera strategies (default: all)
    --min-usable N    Min in-frame & unoccluded keypoints per image (default: 6)
    --start N         Index of the first image (default: 1)
    --output DIR      Output dir (default: data/blender/raw)
    --blender PATH    Path to Blender executable

Convert options:
    --min-visible N   Min visible keypoints (default: 4)

Augment options (lens distortion, blur, noise, JPEG; see src/tools/synthetic_augment.py):
    --images DIR --annotations DIR --out DIR [--seed N] [--copies N]
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


def build_render_command(args, blender):
    render_script = os.path.join(ROOT, "src", "tools", "blender_render.py")
    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python", render_script,
        "--",
        "--count", str(args.count),
        "--seed", str(args.seed),
        "--engine", args.engine,
        "--samples", str(args.samples),
        "--res-min", str(args.res_min),
        "--res-max", str(args.res_max),
        "--aspects", args.aspects,
        "--min-usable", str(args.min_usable),
        "--start", str(args.start),
        "--output", args.output,
    ]
    if args.strategies:
        cmd += ["--strategies", args.strategies]
    return cmd


def cmd_render(args):
    blender = find_blender(args.blender)
    cmd = build_render_command(args, blender)
    print(f"Blender:  {blender}")
    print(f"Count:    {args.count} (start {args.start})")
    print(f"Seed:     {args.seed}")
    print(f"Engine:   {args.engine}, samples {args.samples}")
    print(f"Res:      short side {args.res_min}-{args.res_max}, aspects {args.aspects}")
    print(f"Output:   {args.output}")
    print()
    return subprocess.call(cmd)


def cmd_convert(args):
    convert_script = os.path.join(ROOT, "src", "tools", "blender_to_cvn.py")
    cmd = [sys.executable, convert_script, "--min-visible", str(args.min_visible)]
    print(f"Min visible: {args.min_visible}")
    print()
    return subprocess.call(cmd)


def cmd_augment(args):
    script = os.path.join(ROOT, "src", "tools", "synthetic_augment.py")
    cmd = [sys.executable, script, "--images", args.images, "--annotations", args.annotations,
           "--out", args.out, "--seed", str(args.seed), "--copies", str(args.copies)]
    return subprocess.call(cmd)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate synthetic badminton court data")
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="Render images with Blender")
    render.add_argument("--count", type=int, default=500)
    render.add_argument("--seed", type=int, default=42)
    render.add_argument("--engine", default="BLENDER_EEVEE")
    render.add_argument("--samples", type=int, default=32)
    render.add_argument("--res-min", type=int, default=480)
    render.add_argument("--res-max", type=int, default=720)
    render.add_argument("--aspects", default="16:9,9:16,4:3,1:1")
    render.add_argument("--strategies", default="")
    render.add_argument("--min-usable", type=int, default=6)
    render.add_argument("--start", type=int, default=1)
    render.add_argument("--output", default=os.path.join(ROOT, "data", "blender", "raw"))
    render.add_argument("--blender", help="Path to Blender executable")

    convert = sub.add_parser("convert", help="Convert raw output to CVN format")
    convert.add_argument("--min-visible", type=int, default=4)

    aug = sub.add_parser("augment", help="Lens/blur/noise/JPEG augmentation of a CVN dataset")
    aug.add_argument("--images", default=os.path.join(ROOT, "data", "blender", "images"))
    aug.add_argument("--annotations", default=os.path.join(ROOT, "data", "blender", "annotations"))
    aug.add_argument("--out", required=True)
    aug.add_argument("--seed", type=int, default=0)
    aug.add_argument("--copies", type=int, default=1)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "render":
        sys.exit(cmd_render(args))
    elif args.command == "convert":
        sys.exit(cmd_convert(args))
    elif args.command == "augment":
        sys.exit(cmd_augment(args))


if __name__ == "__main__":
    main()

"""Rewrite a directory of CVN annotations into the canonical labeling.

Writes to --out (never touches the input) unless --in-place is given, in
which case every changed file is first copied to <file>.bak. --dry-run only
reports what would change.

Usage:
    python scripts/canonicalize_annotations.py data/annotations --dry-run
    python scripts/canonicalize_annotations.py data/annotations --out data/annotations_canonical
    python scripts/canonicalize_annotations.py data/annotations --in-place   # keeps .bak copies
"""

import argparse
import collections
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_manifest import canonicalize_annotation, read_json, write_json  # noqa: E402


def canonicalize_directory(in_dir, out_dir=None, in_place=False, dry_run=False):
    """Canonicalize every *.json in ``in_dir``.

    Returns a dict with per-file ``results`` [(file, applied_perm)] and
    ``counts`` (Counter of applied perms; None = undeterminable).
    """
    if not dry_run and not in_place and not out_dir:
        raise ValueError("need out_dir, in_place=True or dry_run=True")
    if out_dir and os.path.abspath(out_dir) == os.path.abspath(in_dir) and not in_place:
        raise ValueError("--out equals the input directory; use --in-place for that")
    results, counts = [], collections.Counter()
    for f in sorted(os.listdir(in_dir)):
        if not f.endswith(".json"):
            continue
        src = os.path.join(in_dir, f)
        ann = read_json(src)
        if "keypoints" not in ann or "visibility" not in ann:
            continue
        new, applied = canonicalize_annotation(ann)
        counts[applied] += 1
        results.append((f, applied))
        if dry_run:
            continue
        changed = applied not in (None, "identity")
        if in_place:
            if changed:
                shutil.copy2(src, src + ".bak")
                write_json(src, new)
        else:
            write_json(os.path.join(out_dir, f), new)
    return {"results": results, "counts": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Canonicalize CVN annotation labels")
    parser.add_argument("in_dir")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", help="write all annotations (canonical) to this directory")
    mode.add_argument("--in-place", action="store_true", help="rewrite changed files, keeping <file>.bak")
    mode.add_argument("--dry-run", action="store_true", help="only report what would change")
    parser.add_argument("--list", action="store_true", help="list every file that would change")
    args = parser.parse_args(argv)

    rep = canonicalize_directory(args.in_dir, args.out, args.in_place, args.dry_run)
    c = rep["counts"]
    total = sum(c.values())
    changing = sum(n for k, n in c.items() if k not in (None, "identity"))
    print(f"{total} annotation(s) in {args.in_dir}")
    for k in ("identity", "rot180", "col_flip", "row_flip", None):
        if c.get(k):
            label = {"identity": "already canonical", None: "undeterminable (unchanged)"}.get(k, f"relabel {k}")
            print(f"  {label:<28} {c[k]}")
    verb = "would change" if args.dry_run else "changed"
    print(f"{changing} file(s) {verb}")
    if args.list:
        for f, applied in rep["results"]:
            if applied not in (None, "identity"):
                print(f"  {applied:<9} {f}")
    return rep


if __name__ == "__main__":
    main()

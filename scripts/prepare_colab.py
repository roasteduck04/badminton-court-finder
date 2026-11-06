"""Build a leak-free, reproducible training/evaluation dataset for Colab.

Sources (all pooled, then split by GROUP so no video/session spans two splits):
  1. Hand labels      data/annotations + data/images        group = capture session
  2. Roboflow export  data/cvn_dataset/{train,valid,test}   group = source video
     (Roboflow's own random split is ignored: its test frames are +-1 frame
     from train frames of the same video)
  3. Blender renders  data/blender/raw (converted on the fly; falls back to
     data/blender/annotations)                                TRAIN ONLY
  4. Optional extra synthetic CVN dirs (--extra-synthetic DIR, e.g. the output
     of src/tools/synthetic_augment.py: DIR/annotations + DIR/images)  TRAIN ONLY

Every annotation is re-canonicalized (src.court_geometry.canonicalize) and
QA'd: image_size is checked against the real image, the homography
reprojection residual is computed at original resolution and outliers above
--max-residual px are dropped. A 32x32 grayscale NCC check reports
near-duplicate images that ended up in different splits.

Output (default data/colab_data_v2; data/colab_data is never touched):
    <out>/train/{annotations,images}/          hand + Roboflow + Blender
    <out>/val/{annotations,images}/            hand + Roboflow, for model selection
    <out>/test_real/{annotations,images}/      held-out hand-labeled sessions
    <out>/test_roboflow/{annotations,images}/  held-out Roboflow videos
    <out>/manifest.json                        frozen split definition
    <out>/summary.txt                          split table

Usage:
    python scripts/prepare_colab.py                       # build data/colab_data_v2
    python scripts/prepare_colab.py --seed 1 --out data/colab_data_v2_seed1
    python scripts/prepare_colab.py --manifest data/colab_data_v2/manifest.json --out DIR
    python scripts/prepare_colab.py --no-synthetic --max-residual 20
    python scripts/prepare_colab.py --zip                 # also write <out>.zip
"""

import argparse
import collections
import datetime as _dt
import os
import shutil
import sys
import zipfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.court_geometry import court_orientation  # noqa: E402
from src.data_manifest import (  # noqa: E402
    assign_groups_to_splits,
    canonicalize_annotation,
    cross_split_near_duplicates,
    hand_label_groups,
    homography_residual,
    image_shape,
    path_basename,
    read_json,
    roboflow_group,
    sha256_file,
    size_status,
    thumb_vector,
    write_json,
)
from src.tools.blender_to_cvn import convert_metadata  # noqa: E402

REAL_ANN_DIR = os.path.join(ROOT, "data", "annotations")
REAL_IMG_DIR = os.path.join(ROOT, "data", "images")
DATASET_DIR = os.path.join(ROOT, "data", "cvn_dataset")
BLENDER_DIR = os.path.join(ROOT, "data", "blender")
LEGACY_OUTPUT_DIR = os.path.join(ROOT, "data", "colab_data")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "data", "colab_data_v2")

SPLITS = ("train", "val", "test_real", "test_roboflow")
PREFIX = {"manual": "mine_", "roboflow": "ds_", "blender": "bl_"}
MANIFEST_VERSION = 1


def _rel(path):
    """Repo-relative POSIX path (absolute if outside the repo)."""
    ap = os.path.abspath(path)
    try:
        rel = os.path.relpath(ap, ROOT)
    except ValueError:
        return ap.replace("\\", "/")
    return ap.replace("\\", "/") if rel.startswith("..") else rel.replace("\\", "/")


def _abs(rel):
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


# ---------------------------------------------------------------------------
# Collect candidate samples
# ---------------------------------------------------------------------------

def _json_files(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []


def collect_manual(ann_dir, img_dir):
    out = []
    for f in _json_files(ann_dir):
        ann = read_json(os.path.join(ann_dir, f))
        img_name = path_basename(ann.get("image_path", ""))
        out.append({
            "source": "manual",
            "src_annotation": os.path.join(ann_dir, f),
            "src_image": os.path.join(img_dir, img_name),
            "ann": ann,
        })
    return out


def collect_roboflow(dataset_dir):
    out = []
    for split in ("train", "valid", "test"):
        ann_dir = os.path.join(dataset_dir, split, "annotations")
        img_dir = os.path.join(dataset_dir, split, "images")
        for f in _json_files(ann_dir):
            ann = read_json(os.path.join(ann_dir, f))
            out.append({
                "source": "roboflow",
                "src_annotation": os.path.join(ann_dir, f),
                "src_image": os.path.join(img_dir, path_basename(ann.get("image_path", ""))),
                "ann": ann,
                "group": ann.get("group") or roboflow_group(f),
            })
    return out


def blender_annotation_from_source(src_annotation, min_visible=0):
    """Load a Blender sample from raw metadata (converted) or a CVN annotation."""
    data = read_json(src_annotation)
    if "camera" in data and "resolution" in data:
        return convert_metadata(data, min_visible=min_visible)
    return data


def collect_blender(blender_dir, min_visible):
    raw_meta = os.path.join(blender_dir, "raw", "metadata")
    raw_img = os.path.join(blender_dir, "raw", "images")
    out, below = [], []
    if os.path.isdir(raw_meta):
        for f in _json_files(raw_meta):
            path = os.path.join(raw_meta, f)
            ann = blender_annotation_from_source(path, min_visible=min_visible)
            if ann is None:
                below.append(path)
                continue
            out.append({"source": "blender", "src_annotation": path,
                        "src_image": os.path.join(raw_img, ann["image_path"]),
                        "ann": ann, "group": ann.get("group") or "blender"})
        return out, below, "raw"
    ann_dir = os.path.join(blender_dir, "annotations")
    img_dir = os.path.join(blender_dir, "images")
    for f in _json_files(ann_dir):
        ann = read_json(os.path.join(ann_dir, f))
        out.append({"source": "blender", "src_annotation": os.path.join(ann_dir, f),
                    "src_image": os.path.join(img_dir, path_basename(ann["image_path"])),
                    "ann": ann, "group": ann.get("group") or "blender"})
    return out, below, "annotations"


def collect_extra_synthetic(syn_dir, index):
    """Extra synthetic CVN set (DIR/annotations + DIR/images), train only."""
    ann_dir, img_dir = os.path.join(syn_dir, "annotations"), os.path.join(syn_dir, "images")
    out = []
    for f in _json_files(ann_dir):
        ann = read_json(os.path.join(ann_dir, f))
        out.append({"source": "blender", "src_annotation": os.path.join(ann_dir, f),
                    "src_image": os.path.join(img_dir, path_basename(ann.get("image_path", ""))),
                    "ann": ann, "group": ann.get("group") or "blender",
                    "prefix": f"blx{index}_"})
    return out


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_sample(s, max_residual=None):
    """Canonicalize + QA one sample in place. Returns a drop reason or None."""
    ann = s["ann"]
    if not os.path.isfile(s["src_image"]):
        return "missing_image"
    actual = image_shape(s["src_image"])
    status = size_status(ann.get("image_size"), actual)
    s["size_status"] = status
    if status == "rescaled":
        ann = dict(ann, image_size=[int(actual[0]), int(actual[1])])
    elif status != "ok":
        return f"image_size_{status}"

    n_vis = int(sum(1 for v in ann["visibility"] if v))
    if n_vis == 0:
        return "no_keypoints"

    canon, applied = canonicalize_annotation(ann)
    if applied is None:
        # Blender's converter canonicalized with amodal (off-frame) points.
        if s["source"] == "blender" and ann.get("canonical_perm"):
            canon = dict(ann)
            s["canon_status"] = "converter"
        else:
            return "orientation_undeterminable"
    else:
        orient = court_orientation(canon["keypoints"], canon["visibility"], canon["image_size"])
        assert orient is not None and orient[0] > 0 and orient[1] >= 0, (
            f"canonicalization failed for {s['src_annotation']}: {orient}")
        s["canon_status"] = "applied"
    s["applied_perm"] = applied

    res = homography_residual(canon["keypoints"], canon["visibility"], canon["image_size"])
    s["residual_max"] = None if res is None else round(res["max"], 3)
    if max_residual is not None and res is not None and res["max"] > max_residual:
        return "residual_outlier"

    s["ann"] = canon
    s["n_visible"] = n_vis
    return None


def output_names(s):
    prefix = s.get("prefix") or PREFIX[s["source"]]
    stem = os.path.splitext(os.path.basename(s["src_annotation"]))[0]
    img_name = os.path.basename(s["src_image"])
    return f"{prefix}{stem}.json", f"{prefix}{img_name}"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def assign_splits(samples, hand_ratios, rf_ratios, seed):
    """Set s['split'] for every sample (whole groups per split)."""
    by_source = collections.defaultdict(list)
    for s in samples:
        by_source[s["source"]].append(s)

    plans = {
        "manual": dict(zip(("train", "val", "test_real"), hand_ratios)),
        "roboflow": dict(zip(("train", "val", "test_roboflow"), rf_ratios)),
    }
    assignment = {}
    for source, ratios in plans.items():
        sizes = collections.Counter(s["group"] for s in by_source.get(source, []))
        if sizes:
            assignment[source] = assign_groups_to_splits(dict(sizes), ratios, seed)
    for s in samples:
        s["split"] = "train" if s["source"] == "blender" else assignment[s["source"]][s["group"]]
    return assignment


def load_thumb(s, size=32):
    if "thumb" not in s:
        img = cv2.imread(s["src_image"], cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if img is None:
            img = cv2.imread(s["src_image"], cv2.IMREAD_GRAYSCALE)
        s["thumb"] = thumb_vector(img, size)
    return s["thumb"]


def merge_similar_groups(samples, threshold):
    """Union groups (within a source) that share near-identical images.

    Different Roboflow "videos" can be cuts of the same broadcast (same
    venue, same camera); different hand sessions can be the same scene. Any
    pair of images from different groups of the same source with thumbnail
    NCC > threshold merges their groups, so they always land in one split.
    Blender (train only) is skipped. Returns a list of merge records.
    """
    idx = [i for i, s in enumerate(samples) if s["source"] != "blender"]
    if not idx:
        return []
    V = np.array([load_thumb(samples[i]) for i in idx])
    S = V @ V.T
    parent = {}

    def find(g):
        parent.setdefault(g, g)
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    best = {}
    for r, c in zip(*np.nonzero(np.triu(S, 1) > threshold)):
        a, b = samples[idx[r]], samples[idx[c]]
        if a["source"] != b["source"] or a["group"] == b["group"]:
            continue
        ka, kb = (a["source"], a["group"]), (b["source"], b["group"])
        key = tuple(sorted([ka, kb]))
        best[key] = max(best.get(key, -1.0), float(S[r, c]))
        ra, rb = find(ka), find(kb)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    members = collections.defaultdict(set)
    for s in samples:
        if s["source"] == "blender":
            continue
        k = (s["source"], s["group"])
        members[find(k)].add(s["group"])
    rename = {}
    for root, gs in members.items():
        if len(gs) > 1:
            name = "+".join(sorted(gs))
            for g in gs:
                rename[(root[0], g)] = name
    for s in samples:
        k = (s["source"], s["group"])
        if k in rename:
            s["orig_group"] = s["group"]
            s["group"] = rename[k]
    return [{"source": a[0], "groups": [a[1], b[1]], "max_ncc": round(v, 4)}
            for (a, b), v in sorted(best.items())]


def near_duplicate_report(samples, threshold):
    vecs = np.array([load_thumb(s) for s in samples])
    pairs = cross_split_near_duplicates(vecs, [s["split"] for s in samples], threshold)
    return [{"a": _rel(samples[i]["src_image"]), "a_split": samples[i]["split"],
             "b": _rel(samples[j]["src_image"]), "b_split": samples[j]["split"],
             "ncc": round(c, 4)} for i, j, c in pairs]


def write_dataset(samples, out_dir):
    entries = {sp: [] for sp in SPLITS}
    for s in sorted(samples, key=lambda s: (s["split"], s["source"], s["src_annotation"])):
        ann_name, img_name = output_names(s)
        split_dir = os.path.join(out_dir, s["split"])
        ann = dict(s["ann"])
        ann["image_path"] = img_name
        ann["source"] = s["source"]
        ann["group"] = s["group"]
        ann.setdefault("canonical_perm", None)
        ann["orig_annotation"] = _rel(s["src_annotation"])
        write_json(os.path.join(split_dir, "annotations", ann_name), ann)
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        shutil.copy2(s["src_image"], os.path.join(split_dir, "images", img_name))
        entries[s["split"]].append({
            "file": ann_name,
            "image": img_name,
            "prefix": s.get("prefix") or PREFIX[s["source"]],
            "source": s["source"],
            "group": s["group"],
            "src_annotation": _rel(s["src_annotation"]),
            "src_image": _rel(s["src_image"]),
            "annotation_sha256": s["annotation_sha256"],
            "image_sha256": s["image_sha256"],
            "canonical_perm": ann["canonical_perm"],
            "n_visible": s.get("n_visible"),
            "residual_max_px": s.get("residual_max"),
        })
    return entries


def summary_table(entries, dropped, near_dups, merges):
    lines = []
    sources = ("manual", "roboflow", "blender")
    lines.append(f"{'split':<15}" + "".join(f"{src:>10}" for src in sources) + f"{'total':>10}{'groups':>8}")
    for sp in SPLITS:
        c = collections.Counter(e["source"] for e in entries[sp])
        ng = len({(e["source"], e["group"]) for e in entries[sp]})
        lines.append(f"{sp:<15}" + "".join(f"{c.get(src, 0):>10}" for src in sources)
                     + f"{len(entries[sp]):>10}{ng:>8}")
    lines.append("")
    lines.append("Group assignment:")
    for sp in SPLITS:
        groups = collections.Counter((e["source"], e["group"]) for e in entries[sp])
        for (src, g), n in sorted(groups.items()):
            if src == "blender":
                continue
            lines.append(f"  {sp:<14} {src:<9} {g:<32} {n:>4}")
        if any(e["source"] == "blender" for e in entries[sp]):
            lines.append(f"  {sp:<14} {'blender':<9} {'blender':<32} "
                         f"{sum(e['source'] == 'blender' for e in entries[sp]):>4}")
    lines.append("")
    lines.append("Merged groups (shared near-identical images):")
    if not merges:
        lines.append("  none")
    for m in merges:
        lines.append(f"  {m['source']:<9} {m['groups'][0]} + {m['groups'][1]}  (max NCC {m['max_ncc']})")
    lines.append("")
    lines.append("Dropped:")
    dc = collections.Counter((d["source"], d["reason"]) for d in dropped)
    if not dc:
        lines.append("  none")
    for (src, reason), n in sorted(dc.items()):
        lines.append(f"  {src:<9} {reason:<32} {n:>4}")
    lines.append("")
    lines.append(f"Cross-split near duplicates (NCC > threshold): {len(near_dups)}")
    for d in near_dups[:20]:
        lines.append(f"  {d['ncc']:.4f}  {d['a_split']}:{d['a']}  <->  {d['b_split']}:{d['b']}")
    return "\n".join(lines)


def prepare_out_dir(out_dir, overwrite):
    out_abs = os.path.realpath(out_dir)
    protected = [LEGACY_OUTPUT_DIR, REAL_ANN_DIR, REAL_IMG_DIR, DATASET_DIR, BLENDER_DIR,
                 os.path.join(ROOT, "data")]
    for p in protected:
        if out_abs == os.path.realpath(p):
            raise SystemExit(f"Refusing to write into {p}; choose a new --out directory.")
    if os.path.exists(out_dir) and os.listdir(out_dir):
        if not overwrite:
            raise SystemExit(f"{out_dir} exists and is not empty; pass --overwrite to rebuild it.")
        if not os.path.isfile(os.path.join(out_dir, "manifest.json")):
            raise SystemExit(f"{out_dir} has no manifest.json (not built by this script); "
                             "refusing to delete it.")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def zip_directory(dir_path, zip_path):
    print(f"\nZipping to {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(dir_path):
            for file in files:
                abs_path = os.path.join(root, file)
                zf.write(abs_path, os.path.relpath(abs_path, os.path.dirname(dir_path)))
    print(f"  {os.path.getsize(zip_path) / (1024 * 1024):.1f} MB")


def build(args):
    samples, dropped = [], []

    manual = collect_manual(args.annotations, args.images)
    groups = hand_label_groups([(i, s["src_image"]) for i, s in enumerate(manual)],
                               gap_minutes=args.session_gap_min)
    for i, s in enumerate(manual):
        s["group"] = groups[i]
    samples += manual
    if not args.no_dataset:
        samples += collect_roboflow(args.dataset)
    blender_mode = None
    if not args.no_synthetic:
        bl, below, blender_mode = collect_blender(args.blender, args.min_visible)
        samples += bl
        dropped += [{"source": "blender", "reason": "below_min_visible_train", "file": _rel(p),
                     "group": "blender"} for p in below]
    for i, d in enumerate(args.extra_synthetic or []):
        samples += collect_extra_synthetic(d, i)

    kept = []
    for s in samples:
        reason = process_sample(s, args.max_residual)
        if reason:
            dropped.append({"source": s["source"], "reason": reason,
                            "file": _rel(s["src_annotation"]), "group": s.get("group"),
                            "residual_max_px": s.get("residual_max")})
        else:
            kept.append(s)

    print("Merging groups that share near-identical images ...")
    merges = merge_similar_groups(kept, args.group_merge_threshold)
    for m in merges:
        print(f"  merged {m['source']} groups {m['groups'][0]} + {m['groups'][1]} (max NCC {m['max_ncc']})")

    assign_splits(kept, args.hand_split, args.roboflow_split, args.seed)  # sets s["split"]

    final = []
    for s in kept:
        if s["split"] == "train" and s["n_visible"] < args.min_visible:
            dropped.append({"source": s["source"], "reason": "below_min_visible_train",
                            "file": _rel(s["src_annotation"]), "group": s["group"]})
            continue
        final.append(s)
    return final, dropped, merges, blender_mode


def rebuild_from_manifest(manifest, allow_changed=False):
    """Recreate the sample list exactly as recorded in a manifest."""
    samples, problems = [], []
    for split, items in manifest["splits"].items():
        for e in items:
            s = {"source": e["source"], "group": e["group"], "split": split, "prefix": e.get("prefix"),
                 "src_annotation": _abs(e["src_annotation"]), "src_image": _abs(e["src_image"])}
            if not (os.path.isfile(s["src_annotation"]) and os.path.isfile(s["src_image"])):
                problems.append(f"missing source for {e['file']}")
                continue
            for key, path in (("annotation_sha256", s["src_annotation"]), ("image_sha256", s["src_image"])):
                if sha256_file(path) != e[key]:
                    problems.append(f"{key} changed for {e['file']} ({_rel(path)})")
            if s["source"] == "blender":
                s["ann"] = blender_annotation_from_source(s["src_annotation"], 0)
            else:
                s["ann"] = read_json(s["src_annotation"])
            reason = process_sample(s, max_residual=None)
            if reason:
                problems.append(f"{e['file']}: {reason}")
                continue
            if s["ann"].get("canonical_perm") != e["canonical_perm"]:
                problems.append(f"{e['file']}: canonical_perm {s['ann'].get('canonical_perm')} "
                                f"!= manifest {e['canonical_perm']}")
            samples.append(s)
    if problems:
        print(f"WARNING: {len(problems)} problem(s) rebuilding from manifest:")
        for p in problems[:30]:
            print("  " + p)
        if not allow_changed:
            raise SystemExit("Sources differ from the manifest; pass --allow-changed to build anyway.")
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build leak-free group-split dataset for Colab")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="output dir (default data/colab_data_v2)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out built by this script")
    parser.add_argument("--manifest", default=None, help="rebuild exactly from an existing manifest.json")
    parser.add_argument("--allow-changed", action="store_true",
                        help="with --manifest: build even if source hashes changed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hand-split", nargs=3, type=float, default=[50, 15, 35],
                        metavar=("TRAIN", "VAL", "TEST"), help="hand-label group ratios (default 50 15 35)")
    parser.add_argument("--roboflow-split", nargs=3, type=float, default=[80, 8, 12],
                        metavar=("TRAIN", "VAL", "TEST"), help="Roboflow video ratios (default 80 8 12)")
    parser.add_argument("--min-visible", type=int, default=4, help="min visible keypoints, TRAIN only")
    parser.add_argument("--max-residual", type=float, default=15.0,
                        help="drop labels whose homography residual exceeds this many px (default 15)")
    parser.add_argument("--near-dup-threshold", type=float, default=0.98,
                        help="report cross-split image pairs with 32x32 NCC above this")
    parser.add_argument("--group-merge-threshold", type=float, default=0.95,
                        help="merge groups of the same source sharing images with NCC above this")
    parser.add_argument("--session-gap-min", type=float, default=10.0,
                        help="hand-label photos further apart than this start a new session")
    parser.add_argument("--no-synthetic", action="store_true", help="exclude Blender renders")
    parser.add_argument("--no-dataset", action="store_true", help="exclude the Roboflow export")
    parser.add_argument("--annotations", default=REAL_ANN_DIR)
    parser.add_argument("--images", default=REAL_IMG_DIR)
    parser.add_argument("--dataset", default=DATASET_DIR)
    parser.add_argument("--blender", default=BLENDER_DIR)
    parser.add_argument("--extra-synthetic", action="append", default=[], metavar="DIR",
                        help="extra synthetic CVN dir (DIR/annotations + DIR/images), train only; repeatable")
    parser.add_argument("--zip", action="store_true", help="also write <out>.zip")
    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    prepare_out_dir(out_dir, args.overwrite)

    blender_mode = None
    if args.manifest:
        src_manifest = read_json(args.manifest)
        print(f"Rebuilding from manifest {args.manifest}")
        samples = rebuild_from_manifest(src_manifest, args.allow_changed)
        dropped = src_manifest.get("dropped", [])
        merges = src_manifest.get("group_merges", [])
        params = dict(src_manifest.get("params", {}), rebuilt_from=_rel(args.manifest))
        seed = src_manifest.get("seed")
    else:
        samples, dropped, merges, blender_mode = build(args)
        params = {k: v for k, v in vars(args).items()
                  if k not in ("out", "overwrite", "manifest", "allow_changed", "zip")}
        params = {k: (_rel(v) if k in ("annotations", "images", "dataset", "blender") else v)
                  for k, v in params.items()}
        params["extra_synthetic"] = [_rel(d) for d in args.extra_synthetic]
        params["blender_input"] = blender_mode
        seed = args.seed

    for s in samples:
        s["annotation_sha256"] = sha256_file(s["src_annotation"])
        s["image_sha256"] = sha256_file(s["src_image"])

    print("Checking cross-split near duplicates ...")
    near_dups = near_duplicate_report(samples, args.near_dup_threshold)

    entries = write_dataset(samples, out_dir)
    manifest = {
        "format_version": MANIFEST_VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "created_by": "scripts/prepare_colab.py",
        "seed": seed,
        "params": params,
        "splits": entries,
        "dropped": dropped,
        "group_merges": merges,
        "near_duplicates": near_dups,
        "counts": {sp: len(v) for sp, v in entries.items()},
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)
    table = summary_table(entries, dropped, near_dups, merges)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")

    print("\n" + table)
    print(f"\nOutput:   {out_dir}")
    print(f"Manifest: {os.path.join(out_dir, 'manifest.json')}")
    if near_dups:
        print(f"\nWARNING: {len(near_dups)} cross-split near-duplicate pair(s); see summary.txt")
    if args.zip:
        zip_directory(out_dir, out_dir.rstrip("\\/") + ".zip")
    return manifest


if __name__ == "__main__":
    main()

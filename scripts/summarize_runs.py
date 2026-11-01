"""Aggregate evaluation results over seeds: mean +- std per config and split.

Reads <runs>/<name>_s<seed>/eval_<split>.json (written by
scripts/runpod/run_experiments.sh via src.evaluation.evaluate) and prints a
Markdown table; --out also writes it to a file.

Usage:
    python scripts/summarize_runs.py --runs /workspace/runs --out results/summary.md
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

METRICS = ["pck@5px", "pck@10px", "pck_norm@0.02", "auc_norm", "err_ref_median", "pcr_norm@0.02"]


def collect(runs_dir):
    """{(config, split): {metric: [values over seeds]}} and seed counts."""
    table = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(runs_dir, "*_s*", "eval_*.json"))):
        run = os.path.basename(os.path.dirname(path))
        m = re.match(r"(.+)_s(\d+)$", run)
        if not m:
            continue
        split = os.path.splitext(os.path.basename(path))[0][len("eval_"):]
        with open(path, encoding="utf-8") as f:
            head = json.load(f).get("headline", {})
        for k in METRICS:
            v = head.get(k)
            if isinstance(v, (int, float)) and not math.isnan(v):
                table[(m.group(1), split)][k].append(float(v))
    return table


def fmt(vals, pct):
    if not vals:
        return "-"
    n = len(vals)
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    s = 100.0 if pct else 1.0
    return f"{mean * s:.1f} ± {std * s:.1f}" if pct else f"{mean:.2f} ± {std:.2f}"


def render(table):
    lines = ["| config | split | seeds | " + " | ".join(METRICS) + " |",
             "|---" * (len(METRICS) + 3) + "|"]
    for (cfg, split) in sorted(table, key=lambda k: (k[1], k[0])):
        row = table[(cfg, split)]
        n = max((len(v) for v in row.values()), default=0)
        cells = [fmt(row.get(k, []), pct=not k.startswith("err")) for k in METRICS]
        lines.append(f"| {cfg} | {split} | {n} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("PCK/AUC/PCR in %, errors in px (original image). mean ± sample std over seeds.")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", required=True, help="directory holding <name>_s<seed>/ run folders")
    p.add_argument("--out", help="also write the Markdown table here")
    a = p.parse_args(argv)
    text = render(collect(a.runs))
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()

"""Nearest-neighbour label-copy baseline.

For each evaluation image, find the most similar TRAINING image and copy its
keypoint labels (in letterbox-normalized coordinates, so differing aspect
ratios are handled the same way the model sees them), then score the copied
labels with exactly the same metric code as the model
(``src.evaluation.evaluate.run_evaluation``). A model that does not clearly
beat this baseline has memorised scenes rather than learned court geometry.

Similarity:
  ncc     (default) 32x32 letterboxed grayscale thumbnails, zero-mean,
          unit-norm; similarity = normalized cross-correlation (dot product).
  resnet  global-average-pooled ImageNet ResNet-18 features of the 224 px
          letterbox, cosine similarity (needs torchvision weights; they are
          downloaded on first use if not cached).

Copied labels: GT-visible keypoints of the neighbour are copied; its
non-visible keypoints are filled by projecting the template through the
neighbour's GT homography (when it has one) or set to the image centre.
Predicted visibility = the neighbour's GT visibility (0/1).

Usage:
  python scripts/nn_baseline.py \
      --train-annotations data/x/train/annotations --train-images data/x/train/images \
      --eval-annotations data/x/test/annotations --eval-images data/x/test/images \
      [--out results/nn_baseline.json]
  python scripts/nn_baseline.py --manifest splits.json --train-split train \
      --eval-split test_real [--embedding resnet]
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.court_geometry import COURT_KEYPOINTS_TEMPLATE, NUM_KEYPOINTS, project_points  # noqa: E402
from src.evaluation.evaluate import (  # noqa: E402
    letterbox_info,
    load_items,
    print_summary,
    run_evaluation,
    write_outputs,
)
from src.evaluation.metrics import fit_gt_homography  # noqa: E402
from src.preprocessing.letterbox import (  # noqa: E402
    keypoints_from_letterbox,
    keypoints_to_letterbox,
    letterbox,
)

THUMB = 32
CANVAS = 640


def ncc_descriptor(bgr, size=THUMB):
    """Zero-mean, unit-norm letterboxed grayscale thumbnail (flattened)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    canvas, _ = letterbox(gray, size)
    v = canvas.astype(np.float64).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class ResNetEmbedder:
    """ImageNet ResNet-18 global-pool embedding (unit norm)."""

    def __init__(self, device="cpu"):
        import torch
        import torchvision

        self.torch = torch
        self.device = device
        m = torchvision.models.resnet18(weights="DEFAULT")
        m.fc = torch.nn.Identity()
        self.model = m.eval().to(device)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __call__(self, bgr):
        canvas, _ = letterbox(bgr, 224)
        rgb = canvas[..., ::-1].astype(np.float64) / 255.0
        x = ((rgb - self.mean) / self.std).transpose(2, 0, 1)[None].astype(np.float32)
        with self.torch.no_grad():
            f = self.model(self.torch.from_numpy(x).to(self.device)).cpu().numpy()[0]
        return f / max(np.linalg.norm(f), 1e-12)


def canvas_labels(ann, h, w, canvas=CANVAS):
    """Neighbour labels as (30,2) normalized letterbox-canvas coords, with
    non-visible keypoints filled from its GT homography or the centre."""
    kps = np.asarray(ann["keypoints"], dtype=np.float64)
    vis = np.asarray(ann["visibility"], dtype=np.float64) > 0.5
    info = letterbox_info(h, w, canvas)
    out = np.full((NUM_KEYPOINTS, 2), 0.5)
    H = fit_gt_homography(kps * np.array([w, h]), vis)
    if H is not None:
        filled = project_points(H, COURT_KEYPOINTS_TEMPLATE) / np.array([w, h])
        if np.all(np.isfinite(filled)):
            out = keypoints_to_letterbox(filled, info)
    if vis.any():
        out[vis] = keypoints_to_letterbox(kps[vis], info)
    return out, vis.astype(np.float64)


class NearestNeighbourPredictor:
    """``predict(bgr)`` -> dict(keypoints, visibility, neighbour, similarity)."""

    def __init__(self, train_items, embed_fn, progress=True):
        self.embed = embed_fn
        self.names, feats, self.labels = [], [], []
        for i, it in enumerate(train_items):
            img = cv2.imread(it["image_path"]) if it["image_path"] else None
            if img is None:
                continue
            h, w = img.shape[:2]
            feats.append(embed_fn(img))
            self.labels.append(canvas_labels(it["ann"], h, w))
            self.names.append(it["name"])
            if progress and (i + 1) % 200 == 0:
                print(f"  indexed {i + 1}/{len(train_items)} training images", file=sys.stderr)
        if not feats:
            raise ValueError("no readable training images")
        self.feats = np.stack(feats)
        self.last = None

    def predict(self, bgr):
        sims = self.feats @ self.embed(bgr)
        j = int(np.argmax(sims))
        kps_canvas, vis = self.labels[j]
        h, w = bgr.shape[:2]
        kps = keypoints_from_letterbox(kps_canvas, letterbox_info(h, w, CANVAS))
        self.last = (self.names[j], float(sims[j]))
        return {"keypoints": kps, "visibility": vis,
                "neighbour": self.names[j], "similarity": float(sims[j])}


def main(argv=None):
    p = argparse.ArgumentParser(description="Nearest-neighbour label-copy baseline")
    p.add_argument("--train-annotations")
    p.add_argument("--train-images")
    p.add_argument("--eval-annotations")
    p.add_argument("--eval-images")
    p.add_argument("--manifest")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test_real")
    p.add_argument("--embedding", choices=["ncc", "resnet"], default="ncc")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.manifest:
        train = load_items(manifest=args.manifest, split=args.train_split, images_dir=args.train_images)
        evals = load_items(manifest=args.manifest, split=args.eval_split, images_dir=args.eval_images)
    else:
        if not (args.train_annotations and args.eval_annotations):
            p.error("give --manifest or both --train-annotations and --eval-annotations")
        train = load_items(args.train_annotations, args.train_images)
        evals = load_items(args.eval_annotations, args.eval_images)

    overlap = {it["name"] for it in train} & {it["name"] for it in evals}
    if overlap:
        print(f"warning: {len(overlap)} eval images also appear in train (by filename)", file=sys.stderr)

    embed = ResNetEmbedder(args.device) if args.embedding == "resnet" else ncc_descriptor
    nn = NearestNeighbourPredictor(train, embed)
    neighbours = []

    def predict_fn(img):
        det = nn.predict(img)
        neighbours.append(nn.last)
        return det

    rows, summary = run_evaluation(evals, predict_fn, n_boot=args.n_boot, seed=args.seed)
    for r, (nb, sim) in zip(rows, neighbours):
        r["neighbour"], r["similarity"] = nb, sim
    sims = np.array([s for _, s in neighbours]) if neighbours else np.zeros(0)
    summary["meta"] = {
        "baseline": f"nearest-neighbour label copy ({args.embedding})",
        "n_train": len(nn.names),
        "n_eval": len(evals),
        "train_eval_filename_overlap": len(overlap),
        "near_duplicates_sim_gt_0.99": int((sims > 0.99).sum()),
        "manifest": args.manifest,
        "train_split": args.train_split if args.manifest else args.train_annotations,
        "eval_split": args.eval_split if args.manifest else args.eval_annotations,
    }
    print_summary(summary)
    if args.out:
        write_outputs(args.out, rows, summary)
        print(f"wrote {os.path.splitext(args.out)[0]}.json/_rows.jsonl/.md")


if __name__ == "__main__":
    main()

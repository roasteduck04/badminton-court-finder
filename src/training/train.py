"""Training loop for CourtVisionNet.

CLI::

    python -m src.training.train --config configs/baseline.json --set seed=1
    python -m src.training.train --config configs/baseline.json \\
        --resume-from checkpoints/baseline_s1/last.pt

Provides:
    train_one_epoch(model, dataloader, loss_fn, optimizer, device, ...) -> float
    validate(model, dataloader, loss_fn, device, image_size) -> (loss, components, metrics)
    train(config: TrainConfig) -> dict

Model selection and early stopping use ``config.select_metric`` (default
pooled PCK@10 px in the letterbox canvas on the real validation split).
Checkpoints follow format_version 2 (see ``save_checkpoint``).
"""

import argparse
import json
import math
import os
import random
import subprocess
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from src.training.config import SELECT_METRICS, TrainConfig

CHECKPOINT_FORMAT_VERSION = 2
_TARGET_KEYS = ("mask", "heatmaps", "keypoints", "visibility", "valid")
NUM_KPTS = 30
_ROT180 = list(range(29, -1, -1))


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------
def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def git_sha():
    """Current commit (with '-dirty' if the tree has changes), or 'unknown'."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def build_datasets(cfg):
    """(train_ds, val_ds) from the manifest, or the directory fallback."""
    from src.preprocessing.augmentation import get_train_transforms
    from src.training.dataset import CourtDataset

    common = dict(
        image_size=cfg.image_size, heatmap_size=cfg.heatmap_size,
        sigma=cfg.sigma, line_thickness=cfg.line_thickness,
        in_channels=cfg.in_channels, canonicalize_labels=cfg.canonicalize_labels,
    )
    train_tf = get_train_transforms(cfg.image_size, seed=cfg.seed)
    if cfg.manifest and os.path.isfile(cfg.manifest):
        images = cfg.images_dir or None
        train_ds = CourtDataset(
            None, images, transform=train_tf, manifest=cfg.manifest,
            split=cfg.train_split, exclude_sources=cfg.exclude_sources, **common)
        val_ds = CourtDataset(
            None, images, transform=None, manifest=cfg.manifest,
            split=cfg.val_split, exclude_sources=cfg.val_exclude_sources, **common)
    else:
        if cfg.manifest:
            warnings.warn(
                f"Manifest {cfg.manifest!r} not found; using annotation directories "
                f"{cfg.train_annotations!r} / {cfg.val_annotations!r}"
            )
        train_ds = CourtDataset(
            cfg.train_annotations, cfg.train_images, transform=train_tf,
            exclude_sources=cfg.exclude_sources, **common)
        val_ds = CourtDataset(
            cfg.val_annotations, cfg.val_images, transform=None,
            exclude_sources=cfg.val_exclude_sources, **common)
    if len(train_ds) == 0:
        raise RuntimeError("Training set is empty")
    if len(val_ds) == 0:
        raise RuntimeError("Validation set is empty")
    return train_ds, val_ds


def build_model(cfg):
    from src.models.courtvisionnet import CourtVisionNet

    return CourtVisionNet(
        in_channels=cfg.in_channels,
        num_keypoints=cfg.num_keypoints,
        image_size=cfg.image_size,
        heatmap_size=cfg.heatmap_size,
        pretrained=cfg.pretrained,
    )


def build_loss(cfg):
    from src.models.losses import CourtVisionLoss

    return CourtVisionLoss(
        seg_weight=cfg.seg_weight,
        heatmap_weight=cfg.heatmap_weight,
        offset_weight=cfg.offset_weight,
        vis_weight=cfg.vis_weight,
        reproj_weight=cfg.reproj_weight,
        heatmap_loss=cfg.heatmap_loss,
        symmetric=cfg.symmetric_loss,
    )


def build_optimizer(model, cfg):
    """AdamW with discriminative LR: backbone at learning_rate, the rest
    (FPN + heads) at learning_rate * head_lr_mult.
    """
    groups = model.parameter_groups()
    no_wd_fn = getattr(model, "no_weight_decay_params", None)
    no_wd = {id(p) for p in (no_wd_fn() if callable(no_wd_fn) else [])}
    head_lr = cfg.learning_rate * cfg.head_lr_mult
    param_groups = [
        {"params": [p for p in groups["backbone"] if id(p) not in no_wd],
         "lr": cfg.learning_rate, "name": "backbone"},
        {"params": [p for p in groups["other"] if id(p) not in no_wd],
         "lr": head_lr, "name": "other"},
        {"params": [p for p in model.parameters() if id(p) in no_wd],
         "lr": head_lr, "weight_decay": 0.0, "name": "no_decay"},
    ]
    return torch.optim.AdamW(
        [g for g in param_groups if g["params"]], weight_decay=cfg.weight_decay
    )


class WarmupCosine:
    """Per-step LR factor: linear warmup from ``start`` then cosine decay to
    ``min_ratio``. ``begin`` delays the warmup (used to re-warm the backbone
    when it is unfrozen); the cosine decay always runs over the whole run.
    """

    def __init__(self, total_steps, warmup_steps, begin=0, min_ratio=0.01, start=0.1):
        self.total = max(1, int(total_steps))
        self.warmup = max(0, int(warmup_steps))
        self.begin = int(begin)
        self.min_ratio = float(min_ratio)
        self.start = float(start)

    def __call__(self, step):
        progress = min(step / self.total, 1.0)
        cosine = self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        local = step - self.begin
        if local < 0:
            return self.start * cosine
        if self.warmup and local < self.warmup:
            return (self.start + (1 - self.start) * local / self.warmup) * cosine
        return cosine


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = cfg.num_epochs * steps_per_epoch
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))
    if cfg.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in optimizer.param_groups],
            total_steps=total,
            pct_start=min(max(warmup / max(total, 1), 0.01), 0.5),
        )
    lambdas = []
    for g in optimizer.param_groups:
        begin = cfg.freeze_backbone_epochs * steps_per_epoch if g.get("name") == "backbone" else 0
        lambdas.append(WarmupCosine(total, warmup, begin=begin, min_ratio=cfg.min_lr_ratio))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


def build_ema(model, cfg):
    if not cfg.ema:
        return None
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay),
                         use_buffers=True)


def _to_device(batch, device):
    targets = {k: batch[k].to(device, non_blocking=True) for k in _TARGET_KEYS if k in batch}
    return batch["image"].to(device, non_blocking=True), targets


# ----------------------------------------------------------------------
# Train / validate
# ----------------------------------------------------------------------
def train_one_epoch(model, dataloader, loss_fn, optimizer, device="cuda",
                    scaler=None, scheduler=None, grad_clip=1.0, amp=False, ema=None):
    """Run one training epoch. Returns the mean loss over all samples."""
    model.train()
    use_amp = bool(amp) and str(device).startswith("cuda")
    total_loss = 0.0
    count = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images, targets = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(images)
            loss, _ = loss_fn(pred, targets)

        if not torch.isfinite(loss):
            warnings.warn(f"Non-finite loss {loss.item()}; skipping batch")
            continue

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update_parameters(model)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size

    return total_loss / max(count, 1)


def summarize_val_errors(errors, prefix=""):
    """Pooled stats of a (N, K) canvas-px error array (NaN = invisible GT).

    Uses ``src.evaluation.metrics.summarize_errors`` so validation and
    evaluation numbers agree, and adds the short aliases used for model
    selection: pck5, pck10 (pooled, err <= t px), mean_px, median_px.
    """
    from src.evaluation.metrics import summarize_errors

    errors = np.asarray(errors, dtype=np.float64)
    stats = summarize_errors(errors if errors.ndim == 2 else np.atleast_2d(errors))
    stats.update({
        "pck5": stats["pck@5"], "pck10": stats["pck@10"],
        "mean_px": stats["err_mean"], "median_px": stats["err_median"],
    })
    return {f"{prefix}{k}": v for k, v in stats.items()}


@torch.no_grad()
def validate(model, dataloader, loss_fn, device="cuda", image_size=640, amp=False):
    """Validation loss + pooled keypoint metrics in letterbox-canvas pixels.

    Returns (avg_loss, avg_components, metrics). ``metrics`` holds the
    ``src.evaluation.metrics.summarize_errors`` statistics (pck@5/10/20,
    err_mean, img_* ...) plus aliases pck5, pck10, mean_px, median_px,
    pooled over all visible GT keypoints; ``sym_*`` variants where each image
    may match the rot180 relabeling of the prediction (symmetry-agnostic);
    and per-source ``<source>/...`` entries.
    """
    from src.evaluation.metrics import keypoint_errors

    model.eval()
    use_amp = bool(amp) and str(device).startswith("cuda")
    total_loss = 0.0
    total_components = defaultdict(float)
    count = 0
    errs, sym_errs, sources = [], [], []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images, targets = _to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(images)
            loss, components = loss_fn(pred, targets)

        batch_size = images.size(0)
        total_loss += float(loss) * batch_size
        count += batch_size
        for k, v in components.items():
            total_components[k] += float(v) * batch_size

        pred_kps = pred["offsets"].float().cpu().numpy()
        gt_kps = batch["keypoints"].numpy()
        gt_vis = batch["visibility"].numpy()
        size_hw = (image_size, image_size)
        e = keypoint_errors(pred_kps, gt_kps, gt_vis, size_hw)
        e_rot = keypoint_errors(pred_kps[:, _ROT180], gt_kps, gt_vis, size_hw)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            use_rot = np.nanmean(e_rot, axis=1) < np.nanmean(e, axis=1)
        errs.append(e)
        sym_errs.append(np.where(use_rot[:, None], e_rot, e))
        sources.extend(batch.get("source", ["unknown"] * batch_size))

    empty = np.zeros((0, NUM_KPTS))
    all_errs = np.concatenate(errs) if errs else empty
    metrics = summarize_val_errors(all_errs)
    metrics.update(summarize_val_errors(np.concatenate(sym_errs) if sym_errs else empty,
                                        prefix="sym_"))
    sources = np.asarray(sources)
    for source in sorted(set(sources.tolist())):
        metrics.update(summarize_val_errors(all_errs[sources == source], prefix=f"{source}/"))
    avg_loss = total_loss / max(count, 1)
    avg_components = {k: v / max(count, 1) for k, v in total_components.items()}
    return avg_loss, avg_components, metrics


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------
def _rng_state():
    """RNG states as plain types/tensors (loadable with weights_only=True)."""
    np_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]),
                  int(np_state[3]), float(np_state[4])],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    if not state:
        return
    if "python" in state:
        py = state["python"]
        random.setstate((py[0], tuple(py[1]), py[2]))
    if "numpy" in state:
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception as e:  # different GPU count
            warnings.warn(f"Could not restore CUDA RNG state: {e}")


def save_checkpoint(path, *, model, optimizer, scheduler, scaler, ema, epoch,
                    best_metric, best_metric_name, config, history,
                    patience_counter, steps_per_epoch, sha):
    """Write a format_version=2 checkpoint (atomically).

    ``model_state_dict`` holds the weights to evaluate (the EMA weights when
    EMA is enabled); ``train_model_state_dict`` then holds the raw weights.
    """
    model_sd = model.state_dict()
    ckpt = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model_sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "best_metric_name": best_metric_name,
        "config": config.to_dict(),
        "git_sha": sha,
        "history": history,
        "rng_state": _rng_state(),
        "patience_counter": int(patience_counter),
        "steps_per_epoch": int(steps_per_epoch),
    }
    if ema is not None:
        ckpt["model_state_dict"] = ema.module.state_dict()
        ckpt["train_model_state_dict"] = model_sd
        ckpt["ema_state_dict"] = ema.state_dict()
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=float)
    os.replace(tmp, path)


def _is_better(value, best, higher_is_better):
    if value is None or not isinstance(value, (int, float)) or math.isnan(value):
        return False
    return value > best if higher_is_better else value < best


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def train(config, force_resume=False):
    """Full training run driven by a TrainConfig.

    Returns a dict with best_metric, best_metric_name, best_val_loss,
    final_epoch, history and checkpoint_dir.
    """
    from torch.utils.data import DataLoader

    from src.preprocessing.augmentation import seed_worker

    cfg = config
    set_seed(cfg.seed, cfg.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg.amp and device == "cuda"
    if cfg.amp and not use_amp:
        warnings.warn("amp=True ignored: CUDA not available")
    ckpt_dir = cfg.resolved_checkpoint_dir()
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Using device: {device}; checkpoints -> {ckpt_dir}")

    train_ds, val_ds = build_datasets(cfg)
    print(f"Train: {len(train_ds)} samples; Val: {len(val_ds)} samples")

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader_kwargs = dict(
        num_workers=cfg.num_workers, pin_memory=(device == "cuda"),
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, generator=generator,
        drop_last=len(train_ds) > cfg.batch_size, **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    steps_per_epoch = len(train_loader)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    ema = build_ema(model, cfg)

    higher_is_better = SELECT_METRICS[cfg.select_metric]
    best_metric = -math.inf if higher_is_better else math.inf
    best_val_loss = math.inf
    patience_counter = 0
    history = []
    start_epoch = 0
    sha = git_sha()

    if cfg.resume_from:
        if not os.path.isfile(cfg.resume_from):
            raise FileNotFoundError(f"resume_from not found: {cfg.resume_from}")
        print(f"Resuming from {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        if ckpt.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            if not force_resume:
                raise RuntimeError(
                    "Checkpoint is not format_version 2 (legacy run); pass --force to "
                    "warm-start from its weights with a fresh optimizer/schedule"
                )
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            warnings.warn(
                f"Warm start from legacy checkpoint: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys; optimizer/schedule reset"
            )
        else:
            old = ckpt.get("config", {})
            changed = [
                f"{k}: {old.get(k)} -> {v}" for k, v in
                (("num_epochs", cfg.num_epochs), ("batch_size", cfg.batch_size),
                 ("scheduler", cfg.scheduler))
                if old.get(k) != v
            ]
            if ckpt.get("steps_per_epoch") != steps_per_epoch:
                changed.append(f"steps_per_epoch: {ckpt.get('steps_per_epoch')} -> {steps_per_epoch}")
            if changed and not force_resume:
                raise RuntimeError(
                    "LR schedule length changed since the checkpoint (" + "; ".join(changed)
                    + "). Pass --force to resume anyway."
                )
            model.load_state_dict(ckpt.get("train_model_state_dict", ckpt["model_state_dict"]))
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") and not changed:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            elif changed:
                # Fast-forward the new schedule to the resumed position.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    for _ in range(ckpt["epoch"] * steps_per_epoch):
                        scheduler.step()
            if scaler is not None and ckpt.get("scaler_state_dict"):
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            if ema is not None and ckpt.get("ema_state_dict"):
                ema.load_state_dict(ckpt["ema_state_dict"])
            start_epoch = int(ckpt["epoch"])
            if ckpt.get("best_metric_name") == cfg.select_metric:
                best_metric = float(ckpt["best_metric"])
            history = list(ckpt.get("history") or [])
            best_val_loss = min([h["val_loss"] for h in history] or [math.inf])
            patience_counter = int(ckpt.get("patience_counter", 0))
            _set_rng_state(ckpt.get("rng_state"))
            print(f"  Resumed at epoch {start_epoch}, best {ckpt.get('best_metric_name')}="
                  f"{ckpt.get('best_metric')}")

    cfg.save(os.path.join(ckpt_dir, "config.json"))

    final_epoch = start_epoch
    for epoch in range(start_epoch, cfg.num_epochs):
        final_epoch = epoch + 1
        t0 = time.time()

        # Per-epoch reseeding makes shuffling and augmentation a function of
        # (seed, epoch) only, so a resumed run continues identically.
        generator.manual_seed(cfg.seed * 1000 + epoch)
        if cfg.num_workers == 0:
            train_ds.reseed(cfg.seed * 1000 + epoch)

        frozen = epoch < cfg.freeze_backbone_epochs
        if frozen:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()
        print(f"\nEpoch {final_epoch}/{cfg.num_epochs}{' (backbone frozen)' if frozen else ''}")

        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            scaler=scaler, scheduler=scheduler, grad_clip=cfg.grad_clip,
            amp=use_amp, ema=ema,
        )
        eval_model = ema.module if ema is not None else model
        val_loss, val_components, val_metrics = validate(
            eval_model, val_loader, loss_fn, device, image_size=cfg.image_size, amp=use_amp,
        )

        entry = {
            "epoch": final_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_components.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": {g.get("name", str(i)): g["lr"] for i, g in enumerate(optimizer.param_groups)},
            "frozen": frozen,
            "time_s": time.time() - t0,
        }
        history.append(entry)
        best_val_loss = min(best_val_loss, val_loss)

        print(f"  Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | "
              f"PCK@5 {val_metrics['pck5']:.3f} PCK@10 {val_metrics['pck10']:.3f} "
              f"mean {val_metrics['mean_px']:.2f}px (sym PCK@10 {val_metrics['sym_pck10']:.3f})")
        for k, v in val_components.items():
            print(f"    {k}: {v:.4f}")
        for k in sorted(val_metrics):
            if "/" in k and k.endswith("pck10"):
                print(f"    {k}: {val_metrics[k]:.3f}")

        value = entry.get(cfg.select_metric)
        improved = _is_better(value, best_metric, higher_is_better)
        if improved:
            best_metric = float(value)
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt_args = dict(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, ema=ema,
            epoch=final_epoch, best_metric=best_metric, best_metric_name=cfg.select_metric,
            config=cfg, history=history, patience_counter=patience_counter,
            steps_per_epoch=steps_per_epoch, sha=sha,
        )
        if improved:
            save_checkpoint(os.path.join(ckpt_dir, "best_model.pt"), **ckpt_args)
            print(f"  Saved best model ({cfg.select_metric}={best_metric:.4f})")
        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), **ckpt_args)
        if cfg.save_every and final_epoch % cfg.save_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"checkpoint_epoch_{final_epoch}.pt"),
                            **ckpt_args)
        _write_json(os.path.join(ckpt_dir, "history.json"), history)

        if patience_counter >= cfg.patience:
            print(f"Early stopping at epoch {final_epoch} "
                  f"(no {cfg.select_metric} improvement for {patience_counter} epochs)")
            break

    return {
        "best_metric": best_metric,
        "best_metric_name": cfg.select_metric,
        "best_val_loss": best_val_loss,
        "final_epoch": final_epoch,
        "history": history,
        "checkpoint_dir": ckpt_dir,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train CourtVisionNet")
    parser.add_argument("--config", default=None, help="JSON/YAML config file")
    parser.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[],
                        metavar="KEY=VALUE", help="override config fields")
    parser.add_argument("--resume-from", default=None, help="checkpoint to resume from")
    parser.add_argument("--force", action="store_true",
                        help="resume even if the schedule length changed / legacy checkpoint")
    return parser.parse_args(argv)


def config_from_args(args):
    cfg = TrainConfig.load(args.config) if args.config else TrainConfig()
    overrides = list(args.overrides)
    if args.resume_from:
        overrides.append(f"resume_from={args.resume_from}")
    return cfg.with_overrides(overrides)


def main(argv=None):
    args = parse_args(argv)
    cfg = config_from_args(args)
    result = train(cfg, force_resume=args.force)
    print(f"\nBest {result['best_metric_name']}: {result['best_metric']:.4f} "
          f"(final epoch {result['final_epoch']}) -> {result['checkpoint_dir']}")
    return result


if __name__ == "__main__":
    main()

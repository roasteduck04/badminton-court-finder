"""Training configuration for CourtVisionNet.

A ``TrainConfig`` can be built in code, loaded from a JSON (or YAML, if
pyyaml is installed) file, and tweaked with ``key=value`` overrides::

    cfg = TrainConfig.load("configs/baseline.json")
    cfg = cfg.with_overrides(["seed=1", "exclude_sources=blender"])

Config files may contain ``"base": "other.json"`` (resolved relative to the
file) to inherit from another config, and keys starting with ``_`` (e.g.
``"_comment"``) are ignored.
"""

import json
import os
import warnings
from dataclasses import asdict, dataclass, field, fields, replace


@dataclass
class TrainConfig:
    """All hyperparameters for a CourtVisionNet training run."""

    # Run identity. ``checkpoint_dir`` may contain ``{name}`` / ``{seed}``.
    name: str = "baseline"
    seed: int = 42
    deterministic: bool = False  # cudnn.deterministic (slower, bit-reproducible)

    # Data: a manifest (JSON: split -> list of {file, source, group, ...})
    # is preferred. If the manifest file does not exist, the directory
    # fallback below is used.
    manifest: str = "data/colab_data_v2/manifest.json"
    train_split: str = "train"
    val_split: str = "val"
    images_dir: str = ""  # optional extra directory for image basename lookup
    train_annotations: str = "data/colab_data/train/annotations"
    val_annotations: str = "data/colab_data/valid/annotations"
    train_images: str = "data/colab_data/train/images"
    val_images: str = "data/colab_data/valid/images"
    # Sources ("roboflow" | "blender" | "manual") dropped from each split.
    # Validation is restricted to real images so model selection never
    # looks at synthetic data.
    exclude_sources: list = field(default_factory=list)
    val_exclude_sources: list = field(default_factory=lambda: ["blender"])
    canonicalize_labels: bool = True

    # Targets
    sigma: float = 2.0          # heatmap Gaussian sigma in heatmap pixels
    line_thickness: int = 5     # seg-mask line width in canvas pixels

    # Model
    in_channels: int = 7        # 7 = full input, 3 = RGB only
    num_keypoints: int = 30
    image_size: int = 640       # square letterbox canvas
    heatmap_size: int = 160
    pretrained: bool = True

    # Optimization
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4   # backbone peak LR
    head_lr_mult: float = 10.0    # FPN + heads LR = learning_rate * head_lr_mult
    weight_decay: float = 1e-4
    scheduler: str = "cosine"     # "cosine" (linear warmup + cosine) | "onecycle"
    warmup_epochs: float = 1.0    # also re-applied to the backbone when it unfreezes
    min_lr_ratio: float = 0.01    # final LR = peak * min_lr_ratio (cosine)
    freeze_backbone_epochs: int = 5
    grad_clip: float = 1.0        # 0 disables clipping
    amp: bool = False             # autocast + GradScaler (CUDA only)
    ema: bool = False
    ema_decay: float = 0.999

    # Loss (see src.models.losses.CourtVisionLoss for the term scales)
    seg_weight: float = 1.0
    heatmap_weight: float = 1.0
    offset_weight: float = 5.0    # use ~1.0 with heatmap_loss="mse"
    vis_weight: float = 0.5
    reproj_weight: float = 0.01   # reprojection term is in canvas px
    heatmap_loss: str = "js"      # "js" | "mse"
    symmetric_loss: bool = True   # min over {identity, rot180} relabelings

    # Model selection / early stopping
    select_metric: str = "val_pck10"  # val_pck10 | val_pck5 | val_mean_px | val_loss
    patience: int = 15

    # Resume from checkpoint
    resume_from: str = ""

    # Output
    checkpoint_dir: str = "checkpoints/{name}_s{seed}"
    save_every: int = 10
    log_interval: int = 10

    # DataLoader. Default 0 for Windows (worker processes need a
    # ``__main__`` guard there); use 2-4 on Colab/Kaggle.
    num_workers: int = 0

    # ------------------------------------------------------------------
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key.startswith("_") or key == "base":
                continue
            if key not in known:
                warnings.warn(f"TrainConfig: ignoring unknown key {key!r}")
                continue
            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path):
        return cls.from_dict(_load_config_dict(path))

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def with_overrides(self, overrides):
        """Return a copy with ``["key=value", ...]`` overrides applied."""
        types = {f.name: f for f in fields(self)}
        updates = {}
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"Override must be key=value, got {item!r}")
            key, raw = item.split("=", 1)
            key = key.strip()
            if key not in types:
                raise KeyError(f"Unknown config key {key!r}")
            updates[key] = _coerce(raw.strip(), getattr(self, key))
        return replace(self, **updates)

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.in_channels not in (3, 7):
            raise ValueError(f"in_channels must be 3 or 7, got {self.in_channels}")
        if self.heatmap_loss not in ("js", "mse"):
            raise ValueError(f"heatmap_loss must be 'js' or 'mse', got {self.heatmap_loss!r}")
        if self.scheduler not in ("cosine", "onecycle"):
            raise ValueError(f"scheduler must be 'cosine' or 'onecycle', got {self.scheduler!r}")
        if self.select_metric not in SELECT_METRICS:
            raise ValueError(
                f"select_metric must be one of {sorted(SELECT_METRICS)}, got {self.select_metric!r}"
            )
        for name in ("exclude_sources", "val_exclude_sources"):
            value = getattr(self, name)
            if isinstance(value, str):
                setattr(self, name, [s for s in value.split(",") if s])

    def resolved_checkpoint_dir(self):
        return self.checkpoint_dir.format(name=self.name, seed=self.seed)


# metric name -> True if larger is better
SELECT_METRICS = {
    "val_pck10": True,
    "val_pck5": True,
    "val_mean_px": False,
    "val_loss": False,
}


def _load_config_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as e:
                raise ImportError("pyyaml is required for YAML configs") from e
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)
    base = data.get("base")
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        merged = _load_config_dict(base_path)
        merged.update({k: v for k, v in data.items() if k != "base"})
        return merged
    return data


def _coerce(raw, current):
    """Parse an override string using the type of the current value."""
    if isinstance(current, bool):
        low = raw.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"Cannot parse {raw!r} as bool")
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError:
            return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        if raw.startswith("["):
            return list(json.loads(raw))
        return [s.strip() for s in raw.split(",") if s.strip()]
    return raw

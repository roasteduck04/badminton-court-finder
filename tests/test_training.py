"""Tests for src.training.config and src.training.train.

The training-loop tests use a tiny stub model/loss that satisfy the shared
contract (forward keys, freeze/unfreeze, parameter_groups) so they do not
depend on the real ResNet-50 model; one slow test runs the real model.
"""

import glob
import json
import math
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.training.train as train_mod
from src.training.config import TrainConfig
from tests.test_dataset import END_ON_VIEW, SIDE_VIEW, make_court_annotation, write_sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------
# Stubs satisfying the model/loss contract
# ----------------------------------------------------------------------
class StubModel(nn.Module):
    def __init__(self, in_channels=7, num_keypoints=30, heatmap_size=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU())
        self.neck = nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.hm = nn.Conv2d(8, num_keypoints, 1)
        self.seg = nn.Conv2d(8, 1, 1)
        self.vis = nn.Linear(8, num_keypoints)
        self.heatmap_size = heatmap_size
        self._frozen = False

    def forward(self, x):
        f = F.relu(self.neck(self.backbone(x)))
        f = F.interpolate(f, size=(self.heatmap_size,) * 2, mode="bilinear", align_corners=False)
        hm = self.hm(f)
        B, K, h, w = hm.shape
        p = F.softmax(hm.reshape(B, K, -1) * 10, dim=-1).reshape(B, K, h, w)
        gx = (torch.arange(w, dtype=hm.dtype) + 0.5) / w
        gy = (torch.arange(h, dtype=hm.dtype) + 0.5) / h
        offsets = torch.stack([(p.sum(2) * gx).sum(-1), (p.sum(3) * gy).sum(-1)], dim=-1)
        seg = F.interpolate(self.seg(f), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"seg_logits": seg, "heatmaps": hm, "offsets": offsets,
                "visibility": self.vis(f.mean((2, 3)))}

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._frozen = False

    def parameter_groups(self):
        ids = {id(p) for p in self.backbone.parameters()}
        return {"backbone": [p for p in self.parameters() if id(p) in ids],
                "other": [p for p in self.parameters() if id(p) not in ids]}


class StubLoss(nn.Module):
    def forward(self, pred, targets):
        vis = targets["visibility"]
        seg = F.binary_cross_entropy_with_logits(pred["seg_logits"], targets["mask"])
        hm = ((torch.sigmoid(pred["heatmaps"]) - targets["heatmaps"]) ** 2).mean()
        off = ((pred["offsets"] - targets["keypoints"]).abs().sum(-1) * vis).sum() / vis.sum().clamp_min(1)
        v = F.binary_cross_entropy_with_logits(pred["visibility"], vis)
        total = seg + hm + off + v
        return total, {"seg_loss": seg.item(), "heatmap_loss": hm.item(),
                       "offset_loss": off.item(), "visibility_loss": v.item()}


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(train_mod, "build_model", lambda cfg: StubModel(
        cfg.in_channels, cfg.num_keypoints, cfg.heatmap_size))
    monkeypatch.setattr(train_mod, "build_loss", lambda cfg: StubLoss())


@pytest.fixture
def mock_data(tmp_path):
    img_dir, ann_dir = tmp_path / "images", tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()
    sources = ["blender", "roboflow", "manual", "roboflow"]
    for i, view in enumerate([END_ON_VIEW, SIDE_VIEW, END_ON_VIEW, SIDE_VIEW]):
        image, ann = make_court_annotation(view)
        write_sample(img_dir, ann_dir, f"f{i}", image, ann, source=sources[i], group=f"g{i}")
    return str(img_dir), str(ann_dir)


def _cfg(mock_data, tmp_path, **kw):
    img_dir, ann_dir = mock_data
    base = dict(
        manifest="", train_annotations=ann_dir, val_annotations=ann_dir,
        train_images=img_dir, val_images=img_dir, val_exclude_sources=[],
        image_size=64, heatmap_size=16, pretrained=False, batch_size=2,
        num_epochs=2, freeze_backbone_epochs=1, warmup_epochs=0.5, patience=10,
        checkpoint_dir=str(tmp_path / "ckpt"), save_every=1, seed=3,
    )
    base.update(kw)
    return TrainConfig(**base)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
class TestTrainConfig:
    def test_defaults(self):
        c = TrainConfig()
        assert (c.in_channels, c.num_keypoints, c.image_size, c.heatmap_size) == (7, 30, 640, 160)
        assert c.heatmap_loss == "js" and c.symmetric_loss is True
        assert c.select_metric == "val_pck10" and c.sigma == 2.0 and c.line_thickness == 5
        assert c.manifest == "data/colab_data_v2/manifest.json"
        assert not hasattr(c, "collinear_weight")

    def test_round_trip(self, tmp_path):
        c = TrainConfig(seed=7, exclude_sources=["blender"], amp=True)
        assert TrainConfig.from_dict(c.to_dict()) == c
        path = str(tmp_path / "c.json")
        c.save(path)
        assert TrainConfig.load(path) == c

    def test_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        path = tmp_path / "c.yaml"
        path.write_text("seed: 9\nsigma: 3.0\n", encoding="utf-8")
        c = TrainConfig.load(str(path))
        assert c.seed == 9 and c.sigma == 3.0

    def test_base_inheritance_and_comments(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps({"seed": 1, "sigma": 4.0}), encoding="utf-8")
        (tmp_path / "b.json").write_text(json.dumps(
            {"base": "a.json", "_comment": "x", "sigma": 5.0}), encoding="utf-8")
        c = TrainConfig.load(str(tmp_path / "b.json"))
        assert c.seed == 1 and c.sigma == 5.0

    def test_unknown_keys_warn(self):
        with pytest.warns(UserWarning):
            c = TrainConfig.from_dict({"collinear_weight": 0.1, "seed": 2})
        assert c.seed == 2

    def test_overrides(self):
        c = TrainConfig().with_overrides([
            "seed=5", "amp=true", "learning_rate=2e-4", "exclude_sources=blender,manual",
            "heatmap_loss=mse", "val_exclude_sources=[]",
        ])
        assert c.seed == 5 and c.amp is True and c.learning_rate == 2e-4
        assert c.exclude_sources == ["blender", "manual"] and c.heatmap_loss == "mse"
        assert c.val_exclude_sources == []
        with pytest.raises(KeyError):
            TrainConfig().with_overrides(["nope=1"])
        with pytest.raises(ValueError):
            TrainConfig().with_overrides(["heatmap_loss=l1"])

    def test_checkpoint_dir_template(self):
        assert TrainConfig(name="x", seed=2).resolved_checkpoint_dir() == "checkpoints/x_s2"

    @pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "configs", "*.json"))))
    def test_shipped_configs_load(self, path):
        c = TrainConfig.load(path)
        assert c.with_overrides(["seed=1"]).seed == 1


def test_ablation_configs_differ_from_baseline_in_one_factor():
    load = lambda n: TrainConfig.load(os.path.join(ROOT, "configs", f"{n}.json")).to_dict()  # noqa: E731
    base = load("baseline")
    expected = {
        "rgb_only": {"in_channels"}, "no_synthetic": {"exclude_sources"},
        "no_reproj": {"reproj_weight"}, "mse_heatmap": {"heatmap_loss", "offset_weight"},
        "no_symmetric": {"symmetric_loss"}, "sigma6": {"sigma"},
    }
    for name, keys in expected.items():
        cfg = load(name)
        diff = {k for k in base if base[k] != cfg[k]} - {"name"}
        assert diff == keys, (name, diff)


# ----------------------------------------------------------------------
# Optimizer / schedule
# ----------------------------------------------------------------------
def test_param_groups_and_head_lr():
    cfg = TrainConfig(learning_rate=1e-4, head_lr_mult=10)
    opt = train_mod.build_optimizer(StubModel(), cfg)
    lrs = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lrs == {"backbone": 1e-4, "other": pytest.approx(1e-3)}


def test_schedule_not_tiny_during_freeze():
    cfg = TrainConfig(num_epochs=10, freeze_backbone_epochs=2, warmup_epochs=1,
                      learning_rate=1e-4, head_lr_mult=10)
    opt = train_mod.build_optimizer(StubModel(), cfg)
    sched = train_mod.build_scheduler(opt, cfg, steps_per_epoch=10)
    lrs = []
    for _ in range(100):
        lrs.append({g["name"]: g["lr"] for g in opt.param_groups})
        opt.step()
        sched.step()
    # Heads reach ~peak LR after the 1-epoch warmup, during the freeze.
    assert lrs[10]["other"] > 0.9e-3
    # Backbone re-warms when it unfreezes at epoch 2, then follows the cosine.
    assert lrs[20]["backbone"] < 0.2e-4 and lrs[30]["backbone"] > 0.7e-4
    assert lrs[99]["other"] < 0.05e-3


def test_onecycle_option():
    cfg = TrainConfig(num_epochs=4, scheduler="onecycle")
    opt = train_mod.build_optimizer(StubModel(), cfg)
    sched = train_mod.build_scheduler(opt, cfg, steps_per_epoch=5)
    assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def _val_loader(mock_data, size=64, hm=16):
    from torch.utils.data import DataLoader

    from src.training.dataset import CourtDataset

    img_dir, ann_dir = mock_data
    ds = CourtDataset(ann_dir, img_dir, image_size=size, heatmap_size=hm)
    return DataLoader(ds, batch_size=3, shuffle=False)


class PerfectModel(StubModel):
    """Returns the GT keypoints (optionally rot180-relabeled) as offsets."""

    def __init__(self, loader, rot=False):
        super().__init__(heatmap_size=16)
        self.gt = torch.cat([b["keypoints"] for b in loader])
        self.rot = rot
        self.i = 0

    def forward(self, x):
        out = super().forward(x)
        gt = self.gt[self.i:self.i + x.shape[0]]
        self.i += x.shape[0]
        out["offsets"] = gt[:, list(range(29, -1, -1))] if self.rot else gt
        return out


def test_validate_metrics_and_sources(mock_data):
    loader = _val_loader(mock_data)
    loss, comps, metrics = train_mod.validate(
        PerfectModel(loader), loader, StubLoss(), device="cpu", image_size=64)
    assert math.isfinite(loss) and "seg_loss" in comps
    assert metrics["pck10"] == 1.0 and metrics["pck5"] == 1.0 and metrics["mean_px"] == 0.0
    assert metrics["pck@10"] == metrics["pck10"]  # evaluation-module keys present
    for src in ("blender", "roboflow", "manual"):
        assert metrics[f"{src}/pck10"] == 1.0
    n_vis = sum(int(b["visibility"].sum()) for b in loader)
    assert metrics["n_points"] == n_vis


def test_validate_symmetric_metric(mock_data):
    loader = _val_loader(mock_data)
    _, _, metrics = train_mod.validate(
        PerfectModel(loader, rot=True), loader, StubLoss(), device="cpu", image_size=64)
    assert metrics["pck10"] < 0.5
    assert metrics["sym_pck10"] == 1.0


def test_validate_does_not_update_weights(mock_data):
    loader = _val_loader(mock_data)
    model = StubModel(heatmap_size=16)
    before = [p.clone() for p in model.parameters()]
    train_mod.validate(model, loader, StubLoss(), device="cpu", image_size=64)
    for a, b in zip(before, model.parameters()):
        assert torch.equal(a, b)


def test_is_better_guards_nan():
    assert not train_mod._is_better(float("nan"), -math.inf, True)
    assert train_mod._is_better(0.5, -math.inf, True)
    assert train_mod._is_better(1.0, 2.0, False)


# ----------------------------------------------------------------------
# Training loop, checkpoints, resume
# ----------------------------------------------------------------------
def test_train_one_epoch_updates_weights(mock_data):
    from torch.utils.data import DataLoader

    from src.training.dataset import CourtDataset

    img_dir, ann_dir = mock_data
    loader = DataLoader(CourtDataset(ann_dir, img_dir, image_size=64, heatmap_size=16),
                        batch_size=2, shuffle=True)
    model = StubModel(heatmap_size=16)
    before = [p.clone() for p in model.parameters()]
    loss = train_mod.train_one_epoch(model, loader, StubLoss(),
                                     torch.optim.Adam(model.parameters(), lr=1e-2), device="cpu")
    assert isinstance(loss, float) and loss > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_train_writes_v2_checkpoint(stubbed, mock_data, tmp_path):
    cfg = _cfg(mock_data, tmp_path, num_epochs=1)
    result = train_mod.train(cfg)
    ckpt_dir = result["checkpoint_dir"]
    assert result["final_epoch"] == 1 and len(result["history"]) == 1
    for name in ("best_model.pt", "last.pt", "checkpoint_epoch_1.pt", "config.json", "history.json"):
        assert os.path.isfile(os.path.join(ckpt_dir, name)), name
    # Loadable without unpickling arbitrary objects.
    ckpt = torch.load(os.path.join(ckpt_dir, "last.pt"), map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch",
                "best_metric", "best_metric_name", "config", "git_sha", "history", "rng_state"):
        assert key in ckpt, key
    assert ckpt["format_version"] == 2 and ckpt["epoch"] == 1
    assert ckpt["best_metric_name"] == "val_pck10"
    assert TrainConfig.from_dict(ckpt["config"]).line_thickness == cfg.line_thickness
    assert ckpt["history"][0]["val_pck10"] == pytest.approx(result["best_metric"])
    with open(os.path.join(ckpt_dir, "history.json"), encoding="utf-8") as f:
        assert json.load(f)[0]["epoch"] == 1
    entry = ckpt["history"][0]
    assert entry["frozen"] is True and "val_roboflow/pck10" in entry


def test_resume_matches_uninterrupted_run(stubbed, mock_data, tmp_path):
    full = train_mod.train(_cfg(mock_data, tmp_path, checkpoint_dir=str(tmp_path / "full")))
    full_sd = torch.load(os.path.join(full["checkpoint_dir"], "last.pt"),
                         weights_only=True)["model_state_dict"]

    resumed_cfg = _cfg(mock_data, tmp_path, checkpoint_dir=str(tmp_path / "resumed"),
                       resume_from=os.path.join(full["checkpoint_dir"], "checkpoint_epoch_1.pt"))
    resumed = train_mod.train(resumed_cfg)
    assert resumed["final_epoch"] == 2
    assert [h["epoch"] for h in resumed["history"]] == [1, 2]
    res_sd = torch.load(os.path.join(resumed["checkpoint_dir"], "last.pt"),
                        weights_only=True)["model_state_dict"]
    for k in full_sd:
        torch.testing.assert_close(res_sd[k], full_sd[k], rtol=0, atol=1e-6)


def test_resume_refuses_changed_schedule(stubbed, mock_data, tmp_path):
    first = train_mod.train(_cfg(mock_data, tmp_path, num_epochs=1))
    last = os.path.join(first["checkpoint_dir"], "last.pt")
    cfg = _cfg(mock_data, tmp_path, num_epochs=3, resume_from=last)
    with pytest.raises(RuntimeError, match="schedule"):
        train_mod.train(cfg)
    result = train_mod.train(cfg, force_resume=True)
    assert result["final_epoch"] == 3


def test_early_stopping(stubbed, mock_data, tmp_path, monkeypatch):
    calls = iter(range(100))
    monkeypatch.setattr(train_mod, "validate", lambda *a, **k: (
        1.0, {}, {"pck5": 0.0, "pck10": 0.5 if next(calls) == 0 else 0.1,
                  "mean_px": 1.0, "sym_pck10": 0.0}))
    result = train_mod.train(_cfg(mock_data, tmp_path, num_epochs=20, patience=2))
    assert result["final_epoch"] == 3
    assert result["best_metric"] == 0.5


def test_cli_with_config_and_overrides(stubbed, mock_data, tmp_path):
    cfg = _cfg(mock_data, tmp_path)
    path = str(tmp_path / "cfg.json")
    cfg.save(path)
    result = train_mod.main(["--config", path, "--set", "num_epochs=1", "seed=4",
                             f"checkpoint_dir={tmp_path / 'cli'}_s{{seed}}"])
    assert result["final_epoch"] == 1
    assert result["checkpoint_dir"].endswith("cli_s4")


@pytest.mark.slow
def test_real_model_smoke(mock_data, tmp_path):
    """Two training iterations with the real CourtVisionNet + CourtVisionLoss."""
    try:
        cfg = _cfg(mock_data, tmp_path, num_epochs=1, image_size=128, heatmap_size=32,
                   freeze_backbone_epochs=1)
        model = train_mod.build_model(cfg)
        train_mod.build_loss(cfg)
        model.parameter_groups()
    except Exception as e:  # model worker's code not in place yet
        pytest.skip(f"real model/loss not available per contract: {e}")
    result = train_mod.train(cfg)
    assert result["final_epoch"] == 1
    assert math.isfinite(result["history"][0]["train_loss"])
    ckpt = torch.load(os.path.join(result["checkpoint_dir"], "last.pt"), weights_only=True)
    assert ckpt["format_version"] == 2
    np.testing.assert_equal(ckpt["config"]["image_size"], 128)

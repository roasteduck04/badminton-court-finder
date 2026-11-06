import pytest
import torch
import torch.nn as nn

from src.models.courtvisionnet import CourtVisionNet, resolve_heatmap_stride


def test_model_forward_default_640():
    model = CourtVisionNet(in_channels=7, pretrained=False)
    out = model(torch.rand(1, 7, 640, 640))
    assert out["seg_logits"].shape == (1, 1, 640, 640)
    assert out["heatmaps"].shape == (1, 30, 160, 160)
    assert out["offsets"].shape == (1, 30, 2)
    assert out["visibility"].shape == (1, 30)


@pytest.mark.parametrize("in_channels", [3, 7])
@pytest.mark.parametrize("heatmap_size,stride", [(32, 4), (64, 2)])
def test_model_forward_strides(in_channels, heatmap_size, stride):
    model = CourtVisionNet(in_channels=in_channels, image_size=128, heatmap_size=heatmap_size,
                           pretrained=False)
    assert model.heatmap_stride == stride
    out = model(torch.rand(2, in_channels, 128, 128))
    assert out["seg_logits"].shape == (2, 1, 128, 128)
    assert out["heatmaps"].shape == (2, 30, heatmap_size, heatmap_size)
    assert out["offsets"].shape == (2, 30, 2)
    assert out["visibility"].shape == (2, 30)
    assert torch.all((out["offsets"] >= 0) & (out["offsets"] <= 1))


def test_resolve_heatmap_stride():
    assert resolve_heatmap_stride(640, 160) == 4
    assert resolve_heatmap_stride(640, 320) == 2
    assert resolve_heatmap_stride(640, 320, 2) == 2
    with pytest.raises(ValueError):
        resolve_heatmap_stride(640, 160, 2)


def test_model_parameter_count():
    model = CourtVisionNet(in_channels=7, pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    assert 20_000_000 < total < 40_000_000


def test_model_gradient_flow():
    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32, pretrained=False)
    out = model(torch.rand(1, 7, 128, 128))
    loss = (out["seg_logits"].sum() + out["heatmaps"].sum()
            + out["offsets"].sum() + out["visibility"].sum())
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def _bn_layers(modules):
    return [m for mod in modules for m in mod.modules() if isinstance(m, nn.BatchNorm2d)]


def test_freeze_backbone_only_freezes_pretrained_and_keeps_bn_eval():
    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32, pretrained=False)
    model.freeze_backbone()
    pretrained = model.backbone.pretrained_modules()
    assert all(not p.requires_grad for m in pretrained for p in m.parameters())
    assert all(p.requires_grad for p in model.backbone.fpn.parameters())
    assert all(p.requires_grad for p in model.seg_head.parameters())
    assert all(p.requires_grad for p in model.kpt_head.parameters())

    model.train()
    frozen_bns = _bn_layers(pretrained)
    assert frozen_bns and all(not bn.training for bn in frozen_bns)
    assert all(bn.training for bn in _bn_layers([model.seg_head, model.kpt_head]))

    # running stats of frozen BN don't move during a training forward
    before = frozen_bns[0].running_mean.clone()
    model(torch.rand(2, 7, 128, 128))
    torch.testing.assert_close(frozen_bns[0].running_mean, before)

    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.parameters())
    assert all(bn.training for bn in frozen_bns)
    model.eval()
    assert all(not bn.training for bn in frozen_bns)


def test_parameter_groups_cover_all_params_once():
    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=64, pretrained=False)
    groups = model.parameter_groups()
    assert set(groups) == {"backbone", "other"}
    ids_b = [id(p) for p in groups["backbone"]]
    ids_o = [id(p) for p in groups["other"]]
    all_ids = [id(p) for p in model.parameters()]
    assert len(ids_b) + len(ids_o) == len(all_ids)
    assert set(ids_b) | set(ids_o) == set(all_ids)
    assert not set(ids_b) & set(ids_o)
    fpn_ids = {id(p) for p in model.backbone.fpn.parameters()}
    assert fpn_ids <= set(ids_o)
    assert id(model.backbone.stem[0].weight) in set(ids_b)


def test_no_weight_decay_params():
    model = CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32, pretrained=False)
    nwd = model.no_weight_decay_params()
    assert len(nwd) == 1 and nwd[0] is model.kpt_head.beta_raw
    assert any(p is nwd[0] for p in model.parameter_groups()["other"])


def test_state_dict_keys_backward_compatible():
    """Backbone/seg/heatmap keys keep their old names (legacy checkpoints map)."""
    keys = set(CourtVisionNet(in_channels=7, image_size=128, heatmap_size=32,
                              pretrained=False).state_dict())
    for k in ("backbone.stem.0.weight", "backbone.fpn.inner_blocks.0.0.weight",
              "seg_head.fuse.3.weight", "kpt_head.heatmap_conv.6.weight", "kpt_head.beta_raw"):
        assert k in keys

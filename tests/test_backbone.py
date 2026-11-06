import torch
import torch.nn.functional as F

from src.models.backbone import (
    EXTRA_CHANNEL_MEAN, EXTRA_CHANNEL_STD, IMAGENET_MEAN, IMAGENET_STD,
    CourtBackbone, adapt_input_conv, input_normalization,
)


def test_backbone_output_shapes():
    model = CourtBackbone(in_channels=7, pretrained=False)
    x = torch.rand(2, 7, 640, 640)
    features = model(x)

    assert features["s2"].shape == (2, 64, 320, 320)
    assert features["p2"].shape == (2, 256, 160, 160)
    assert features["p3"].shape == (2, 256, 80, 80)
    assert features["p4"].shape == (2, 256, 40, 40)
    assert features["p5"].shape == (2, 256, 20, 20)


def test_backbone_3_channel_input():
    model = CourtBackbone(in_channels=3, pretrained=False)
    features = model(torch.rand(2, 3, 256, 256))
    assert features["p2"].shape == (2, 256, 64, 64)


def test_backbone_gradient_flow():
    model = CourtBackbone(in_channels=7, pretrained=False)
    x = torch.rand(1, 7, 128, 128, requires_grad=True)
    features = model(x)
    sum(f.sum() for f in features.values()).backward()
    assert x.grad is not None


def test_input_normalization_values():
    mean, std = input_normalization(7)
    assert mean[:3] == list(IMAGENET_MEAN) and std[:3] == list(IMAGENET_STD)
    assert mean[3:] == [EXTRA_CHANNEL_MEAN] * 4 and std[3:] == [EXTRA_CHANNEL_STD] * 4
    mean3, std3 = input_normalization(3)
    assert len(mean3) == 3 and len(std3) == 3


def test_backbone_normalizes_internally():
    model = CourtBackbone(in_channels=7, pretrained=False)
    x = torch.rand(1, 7, 8, 8)
    y = model.normalize(x)
    expected_r = (x[:, 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    expected_extra = (x[:, 5] - EXTRA_CHANNEL_MEAN) / EXTRA_CHANNEL_STD
    torch.testing.assert_close(y[:, 0], expected_r)
    torch.testing.assert_close(y[:, 5], expected_extra)

    raw = CourtBackbone(in_channels=7, pretrained=False, normalize_input=False)
    torch.testing.assert_close(raw.normalize(x), x)


def test_normalization_buffers_not_in_state_dict():
    model = CourtBackbone(in_channels=7, pretrained=False)
    assert not any("input_mean" in k or "input_std" in k for k in model.state_dict())


def test_adapt_input_conv_preserves_grey_response():
    """For inputs whose channels all carry the same signal, the adapted
    7-channel stem gives exactly the pretrained 3-channel response."""
    torch.manual_seed(0)
    rgb_w = torch.randn(64, 3, 7, 7)
    w7 = adapt_input_conv(rgb_w, 7)
    assert w7.shape == (64, 7, 7, 7)
    # RGB filters are the pretrained ones (scaled), extra = scaled RGB mean
    torch.testing.assert_close(w7[:, :3], rgb_w * 3 / 7)
    torch.testing.assert_close(w7[:, 3], rgb_w.mean(1) * 3 / 7)

    g = torch.rand(1, 1, 32, 32)
    y3 = F.conv2d(g.repeat(1, 3, 1, 1), rgb_w, padding=3)
    y7 = F.conv2d(g.repeat(1, 7, 1, 1), w7, padding=3)
    torch.testing.assert_close(y7, y3, rtol=1e-4, atol=1e-4)

    torch.testing.assert_close(adapt_input_conv(rgb_w, 3), rgb_w)

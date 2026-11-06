import pytest
import torch

from src.models.keypoint_head import KeypointHead, soft_argmax_2d
from src.models.segmentation_head import SegmentationHead


def make_fake_features(batch_size=2, size=640):
    s = size
    return {
        "s2": torch.randn(batch_size, 64, s // 2, s // 2),
        "p2": torch.randn(batch_size, 256, s // 4, s // 4),
        "p3": torch.randn(batch_size, 256, s // 8, s // 8),
        "p4": torch.randn(batch_size, 256, s // 16, s // 16),
        "p5": torch.randn(batch_size, 256, s // 32, s // 32),
    }


def test_segmentation_head_output_shape():
    head = SegmentationHead(in_channels=256, image_size=640)
    out = head(make_fake_features())
    assert out.shape == (2, 1, 640, 640)


def test_segmentation_head_out_size():
    head = SegmentationHead(in_channels=256, image_size=640)
    out = head(make_fake_features(size=128), out_size=(128, 128))
    assert out.shape == (2, 1, 128, 128)


@pytest.mark.parametrize("stride,hm", [(4, 160), (2, 320)])
def test_keypoint_head_output_shapes(stride, hm):
    head = KeypointHead(in_channels=256, num_keypoints=30, heatmap_size=hm, heatmap_stride=stride)
    out = head(make_fake_features())
    assert out["heatmaps"].shape == (2, 30, hm, hm)
    assert out["offsets"].shape == (2, 30, 2)
    assert out["visibility"].shape == (2, 30)
    assert out["beta"].dim() == 0


def test_keypoint_head_rejects_bad_stride():
    with pytest.raises(ValueError):
        KeypointHead(heatmap_stride=8)


def test_segmentation_head_gradient():
    head = SegmentationHead(in_channels=256, image_size=128)
    features = make_fake_features(size=128)
    features["p2"].requires_grad_(True)
    head(features).sum().backward()
    assert features["p2"].grad is not None


@pytest.mark.parametrize("stride,hm", [(4, 32), (2, 64)])
def test_keypoint_head_gradient(stride, hm):
    head = KeypointHead(in_channels=256, num_keypoints=30, heatmap_size=hm, heatmap_stride=stride)
    features = make_fake_features(size=128)
    for k in ("p2", "s2"):
        features[k].requires_grad_(True)
    out = head(features)
    loss = out["heatmaps"].sum() + out["offsets"].sum() + out["visibility"].sum()
    loss.backward()
    assert features["p2"].grad is not None
    if stride == 2:
        assert features["s2"].grad is not None
    assert head.beta_raw.grad is not None


def test_visibility_does_not_backprop_into_heatmaps():
    """Sampling location and heatmap statistics are detached."""
    head = KeypointHead(in_channels=256, num_keypoints=30, heatmap_size=32)
    features = make_fake_features(size=128)
    out = head(features)
    out["visibility"].sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in head.heatmap_conv.parameters())
    assert head.vis_mlp[0].weight.grad is not None


def test_soft_argmax_pixel_centre_grid():
    W = 16
    hm = torch.full((1, 2, W, W), -50.0)
    hm[0, 0, 3, 7] = 50.0  # row 3, col 7
    hm[0, 1, 0, W - 1] = 50.0
    coords, _ = soft_argmax_2d(hm, beta=1.0)
    torch.testing.assert_close(coords[0, 0], torch.tensor([(7 + 0.5) / W, (3 + 0.5) / W]))
    torch.testing.assert_close(coords[0, 1], torch.tensor([(W - 0.5) / W, 0.5 / W]))


def test_beta_is_always_positive():
    head = KeypointHead(in_channels=256, num_keypoints=30, heatmap_size=32, soft_argmax_beta=10.0)
    assert head.beta.item() == pytest.approx(10.0, rel=1e-4)
    with torch.no_grad():
        head.beta_raw.fill_(-100.0)
    assert head.beta.item() > 0

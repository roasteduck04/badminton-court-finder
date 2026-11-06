"""Tests for src.inference.decode."""

import numpy as np
import pytest
import torch

from src.inference.decode import decode_heatmaps, gaussian_heatmap


def _random_centres(n, W, seed=0, margin=8):
    rng = np.random.default_rng(seed)
    return rng.uniform(margin, W - margin, (n, 2))


def _stack(centres, W, sigma):
    return np.stack([gaussian_heatmap(c, (W, W), sigma) for c in centres])


@pytest.mark.parametrize("sigma", [1.5, 2.0, 3.0, 6.0])
def test_local_dark_decodes_gaussian_within_0_1_px(sigma):
    W = 160
    centres = _random_centres(20, W)
    coords, conf = decode_heatmaps(_stack(centres, W, sigma))  # defaults: local + dark
    err_px = np.abs(coords * W - 0.5 - centres).max()
    assert err_px < 0.1
    assert np.all(conf > 0.8)


@pytest.mark.parametrize("sigma", [1.5, 2.0])
def test_local_without_dark_within_0_1_px(sigma):
    W = 160
    centres = _random_centres(20, W, seed=1)
    coords, _ = decode_heatmaps(_stack(centres, W, sigma), refine=None)
    assert np.abs(coords * W - 0.5 - centres).max() < 0.1


def test_global_soft_argmax_beta10_is_biased():
    """The bug that motivated the local decoder: global soft-argmax with
    beta ~10 pulls predictions towards the heatmap centre by many pixels."""
    W, sigma = 160, 2.0
    centres = _random_centres(20, W, seed=2)
    coords, _ = decode_heatmaps(_stack(centres, W, sigma), method="soft_argmax", beta=10.0)
    err_px = np.abs(coords * W - 0.5 - centres).max()
    assert err_px > 1.0
    # and the error points towards the centre
    d_pred = np.abs(coords * W - 0.5 - W / 2).sum()
    d_true = np.abs(centres - W / 2).sum()
    assert d_pred < d_true


def test_logits_kind_matches_softmax_target():
    """For JS-trained heatmaps, beta * logits = log of the target distribution."""
    W, sigma, beta = 64, 2.0, 3.0
    centres = _random_centres(10, W, seed=3)
    hm = _stack(centres, W, sigma)
    logits = np.log(hm / hm.sum(axis=(1, 2), keepdims=True) + 1e-30) / beta
    coords, conf = decode_heatmaps(logits, kind="logits", beta=beta)
    assert np.abs(coords * W - 0.5 - centres).max() < 0.1
    assert np.all((conf > 0.5) & (conf <= 1.0))


def test_torch_batched_input_and_border_peaks():
    W = 32
    hm = torch.zeros(2, 3, W, W)
    hm[0, 0, 0, 0] = 1.0           # corner
    hm[0, 1, 5, W - 1] = 1.0       # right border
    hm[1, 2] = torch.from_numpy(gaussian_heatmap((10.3, 20.6), (W, W), 1.5))
    coords, conf = decode_heatmaps(hm)
    assert isinstance(coords, torch.Tensor) and coords.shape == (2, 3, 2) and conf.shape == (2, 3)
    torch.testing.assert_close(coords[0, 0], torch.tensor([0.5 / W, 0.5 / W]))
    torch.testing.assert_close(coords[0, 1], torch.tensor([(W - 0.5) / W, 5.5 / W]))
    assert torch.abs(coords[1, 2] * W - 0.5 - torch.tensor([10.3, 20.6])).max() < 0.1
    assert torch.isfinite(coords).all()


def test_flat_heatmap_is_finite():
    coords, conf = decode_heatmaps(np.zeros((2, 16, 16), np.float32))
    assert np.all(np.isfinite(coords)) and np.all(conf == 0)

"""Heatmap -> keypoint coordinate decoding.

Global soft-argmax over the whole heatmap (what the model's ``offsets``
output is) is biased towards the heatmap centre: with beta ~10 and a
Gaussian heatmap of peak 1, the ~25k background pixels at value ~0 carry
about as much softmax mass as the peak, pulling the expectation towards the
middle. ``decode_heatmaps`` instead takes the argmax and refines it locally:

    method="local":   centroid of a (2r+1)^2 window around the argmax, then a
                      few mean-shift steps that re-centre the window on the
                      estimate (bilinear sampling), removing most of the
                      truncation bias of a window fixed at the integer argmax
    refine="dark":    DARK-style (Zhang et al., CVPR 2020) second-order Taylor
                      step on the log-heatmap at the argmax, -H^-1 g. Exact for
                      a Gaussian peak, so it removes the window-truncation bias
                      of the local centroid. Falls back to the local centroid
                      where the Hessian is not negative definite, the step
                      leaves the argmax pixel's neighbourhood, or the argmax is
                      on the border.
    method="soft_argmax": global soft-argmax (for comparison / legacy).

Coordinates follow the pixel-centre convention: heatmap index i <-> normalized
(i + 0.5) / W.

``kind`` says what the heatmap values are:
    "gaussian": regressed Gaussians with peak ~1 (heatmap_loss="mse"); window
                weights are max(h, 0), the DARK log-heatmap is log(h), and the
                confidence is the peak value clipped to [0, 1].
    "logits":   logits of a spatial softmax(beta * h) (heatmap_loss="js");
                window weights are exp(beta * (h - max)), the log-heatmap is
                beta * h, and the confidence is the softmax mass inside the
                window.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.models.keypoint_head import soft_argmax_2d


def _to_tensor(heatmaps):
    if isinstance(heatmaps, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(heatmaps)), True
    return heatmaps, False


@torch.no_grad()
def decode_heatmaps(heatmaps, method="local", radius=3, refine="dark", kind="gaussian", beta=1.0,
                    iterations=3):
    """Decode heatmaps to normalized coordinates and a peak confidence.

    Args:
        heatmaps: (B, K, H, W) or (K, H, W), torch tensor or numpy array
        method: "local" or "soft_argmax"
        radius: window radius r for "local" (window is (2r+1)^2)
        iterations: mean-shift re-centring steps of the local window
        refine: "dark" or None
        kind: "gaussian" or "logits" (see module docstring)
        beta: softmax sharpness for kind="logits" / method="soft_argmax"

    Returns:
        (coords (…, K, 2) normalized (x, y), confidence (…, K)), as numpy
        arrays if the input was numpy, else float32 tensors.
    """
    hm, was_numpy = _to_tensor(heatmaps)
    hm = hm.float()
    squeeze = hm.dim() == 3
    if squeeze:
        hm = hm.unsqueeze(0)
    B, K, H, W = hm.shape
    beta = float(beta)
    flat = hm.reshape(B, K, H * W)
    peak, idx = flat.max(dim=-1)
    iy = torch.div(idx, W, rounding_mode="floor")
    ix = idx % W

    if method == "soft_argmax":
        coords, probs = soft_argmax_2d(hm, beta)
        conf = probs.max(dim=-1).values * (H * W)
        conf = 1.0 - torch.exp(-conf / 50.0)  # monotone squash to [0, 1)
        return _finish(coords, conf, squeeze, was_numpy)
    if method != "local":
        raise ValueError(f"unknown method {method!r}")
    if kind not in ("gaussian", "logits"):
        raise ValueError(f"unknown kind {kind!r}")

    # Local window around the argmax
    d = torch.arange(-radius, radius + 1, device=hm.device)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    yy = iy.unsqueeze(-1) + dy.reshape(-1)  # (B, K, M)
    xx = ix.unsqueeze(-1) + dx.reshape(-1)
    inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    vals = flat.gather(-1, yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1))
    if kind == "gaussian":
        w = vals.clamp_min(0.0)
    else:
        w = torch.exp(beta * (vals - peak.unsqueeze(-1)))
    w = w * inb
    wsum = w.sum(-1)
    safe = wsum > 1e-12
    cx = torch.where(safe, (w * xx).sum(-1) / wsum.clamp_min(1e-12), ix.float())
    cy = torch.where(safe, (w * yy).sum(-1) / wsum.clamp_min(1e-12), iy.float())
    # Mean-shift: re-centre the window on the estimate (bilinear sampling) so
    # the window is symmetric about the peak and truncation bias vanishes.
    for _ in range(iterations):
        cx, cy = _shifted_centroid(hm, cx, cy, dx.reshape(-1), dy.reshape(-1), kind, beta, peak)

    if refine == "dark":
        log_hm = torch.log(hm.clamp_min(1e-10)) if kind == "gaussian" else beta * hm
        ox, oy, ok = _dark_offset(log_hm, ix, iy)
        cx = torch.where(ok, ix.float() + ox, cx)
        cy = torch.where(ok, iy.float() + oy, cy)
    elif refine is not None:
        raise ValueError(f"unknown refine {refine!r}")

    coords = torch.stack([(cx + 0.5) / W, (cy + 0.5) / H], dim=-1)
    if kind == "gaussian":
        conf = peak.clamp(0.0, 1.0)
    else:
        total = torch.exp(beta * (flat - peak.unsqueeze(-1))).sum(-1)
        conf = (wsum / total).clamp(0.0, 1.0)
    return _finish(coords, conf, squeeze, was_numpy)


def _shifted_centroid(hm, cx, cy, dx, dy, kind, beta, peak):
    """Weighted centroid of a window of bilinear samples centred at (cx, cy)."""
    B, K, H, W = hm.shape
    sx = cx.unsqueeze(-1) + dx.to(cx.dtype)  # (B, K, M)
    sy = cy.unsqueeze(-1) + dy.to(cy.dtype)
    inb = (sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1)
    grid = torch.stack([(sx + 0.5) / W * 2 - 1, (sy + 0.5) / H * 2 - 1], dim=-1)
    vals = F.grid_sample(
        hm.reshape(B * K, 1, H, W), grid.reshape(B * K, 1, -1, 2),
        mode="bilinear", padding_mode="border", align_corners=False,
    ).reshape(B, K, -1)
    if kind == "gaussian":
        w = vals.clamp_min(0.0)
    else:
        w = torch.exp(beta * (vals - peak.unsqueeze(-1)))
    w = w * inb
    wsum = w.sum(-1)
    safe = wsum > 1e-12
    nx = torch.where(safe, (w * sx).sum(-1) / wsum.clamp_min(1e-12), cx)
    ny = torch.where(safe, (w * sy).sum(-1) / wsum.clamp_min(1e-12), cy)
    return nx, ny


def _dark_offset(log_hm, ix, iy):
    """Newton step -H^-1 g of the log-heatmap at integer (ix, iy)."""
    B, K, H, W = log_hm.shape
    interior = (ix >= 1) & (ix <= W - 2) & (iy >= 1) & (iy <= H - 2)
    x = ix.clamp(1, max(1, W - 2))
    y = iy.clamp(1, max(1, H - 2))
    flat = log_hm.reshape(B, K, H * W)

    def at(oy, ox):
        yy = (y + oy).clamp(0, H - 1)
        xx = (x + ox).clamp(0, W - 1)
        return flat.gather(-1, (yy * W + xx).unsqueeze(-1)).squeeze(-1)

    c = at(0, 0)
    gx = 0.5 * (at(0, 1) - at(0, -1))
    gy = 0.5 * (at(1, 0) - at(-1, 0))
    hxx = at(0, 1) - 2 * c + at(0, -1)
    hyy = at(1, 0) - 2 * c + at(-1, 0)
    hxy = 0.25 * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1))
    det = hxx * hyy - hxy ** 2
    ok = interior & (hxx < 0) & (det > 1e-12)
    det_safe = torch.where(ok, det, torch.ones_like(det))
    ox = -(hyy * gx - hxy * gy) / det_safe
    oy = -(-hxy * gx + hxx * gy) / det_safe
    ok = ok & (ox.abs() <= 1.0) & (oy.abs() <= 1.0)
    return ox, oy, ok


def _finish(coords, conf, squeeze, was_numpy):
    if squeeze:
        coords, conf = coords[0], conf[0]
    if was_numpy:
        return coords.cpu().numpy(), conf.cpu().numpy()
    return coords, conf


def gaussian_heatmap(center_xy, size, sigma):
    """Reference Gaussian heatmap (peak 1) centred at heatmap-index coords.

    ``center_xy`` is in heatmap index space, i.e. normalized kp * W - 0.5.
    ``size`` is (H, W).
    """
    H, W = size
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = center_xy
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


__all__ = ["decode_heatmaps", "gaussian_heatmap"]

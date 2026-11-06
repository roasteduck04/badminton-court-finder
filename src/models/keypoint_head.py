import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BETA_EPS = 1e-3


def _inverse_softplus(y):
    return math.log(math.expm1(y))


def soft_argmax_2d(heatmaps, beta=1.0):
    """Global spatial soft-argmax with the pixel-centre grid.

    Args:
        heatmaps: (B, K, H, W) logits
        beta: scalar (tensor or float) sharpness; softmax(beta * heatmaps)

    Returns:
        coords (B, K, 2) normalized (x, y) where pixel index i <-> (i + 0.5) / W,
        probs (B, K, H*W) the softmax distribution (float32).
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.float().reshape(B, K, H * W)
    probs = F.softmax(flat * beta, dim=-1)
    gx = (torch.arange(W, device=heatmaps.device, dtype=torch.float32) + 0.5) / W
    gy = (torch.arange(H, device=heatmaps.device, dtype=torch.float32) + 0.5) / H
    p = probs.view(B, K, H, W)
    x = (p.sum(dim=2) * gx).sum(dim=-1)  # marginal over rows, then E[x]
    y = (p.sum(dim=3) * gy).sum(dim=-1)
    return torch.stack([x, y], dim=-1), probs


def _conv_bn_relu(cin, cout, k=3):
    return [
        nn.Conv2d(cin, cout, k, padding=k // 2),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ]


class KeypointHead(nn.Module):
    """Keypoint head: heatmaps, soft-argmax coordinates and per-keypoint visibility.

    Produces:
        heatmaps: (B, K, hm, hm) raw logits
        offsets: (B, K, 2) normalized [0,1] coordinates via soft-argmax over
            softmax(beta * heatmaps) on the pixel-centre grid (i + 0.5) / W
        visibility: (B, K) logits
        beta: () the current soft-argmax sharpness, softplus(beta_raw) + eps

    ``heatmap_stride`` 4 predicts from P2 (H/4). ``heatmap_stride`` 2 adds a
    learned 2x upsampling (transposed conv) fused with the stride-2 stem
    feature ``s2`` (H/2).

    Visibility is predicted per keypoint from (a) heatmap statistics (peak
    logit, log peak-to-uniform probability ratio, spread of the soft-argmax
    distribution), (b) a reduced P2 feature bilinearly sampled at the
    soft-argmax location and (c) a global context vector from P5, plus a
    learned keypoint embedding, through one MLP shared across keypoints.
    The sampling location and heatmap statistics are detached so the
    visibility loss does not move the coordinates.

    ``beta_raw`` should be excluded from weight decay (decay would pull beta
    towards softplus(0) = 0.69); see CourtVisionNet.no_weight_decay_params().
    """

    def __init__(self, in_channels=256, num_keypoints=30, heatmap_size=160,
                 soft_argmax_beta=10.0, heatmap_stride=4, stem_channels=64):
        super().__init__()
        if heatmap_stride not in (2, 4):
            raise ValueError(f"heatmap_stride must be 2 or 4, got {heatmap_stride}")
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.heatmap_stride = heatmap_stride
        self.beta_raw = nn.Parameter(torch.tensor(_inverse_softplus(soft_argmax_beta - BETA_EPS)))

        if heatmap_stride == 4:
            # Same layout (and state-dict keys) as earlier versions.
            self.heatmap_conv = nn.Sequential(
                *_conv_bn_relu(in_channels, 256),
                *_conv_bn_relu(256, 256),
                nn.Conv2d(256, num_keypoints, 1),
            )
            final = self.heatmap_conv[-1]
        else:
            self.heatmap_conv = nn.Sequential(
                *_conv_bn_relu(in_channels, 256),
                *_conv_bn_relu(256, 256),
            )
            self.heatmap_up = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.heatmap_skip = nn.Sequential(*_conv_bn_relu(stem_channels, 64, k=1))
            self.heatmap_out = nn.Sequential(
                *_conv_bn_relu(128 + 64, 128),
                nn.Conv2d(128, num_keypoints, 1),
            )
            final = self.heatmap_out[-1]
        # Near-zero initial logits: flat soft-argmax distribution and a
        # heatmap that starts at ~0 (HRNet-style init), so neither the JS nor
        # the per-heatmap MSE loss starts saturated.
        nn.init.normal_(final.weight, std=1e-3)
        nn.init.zeros_(final.bias)

        # Visibility branch
        vis_dim = 64
        self.vis_reduce = nn.Sequential(*_conv_bn_relu(in_channels, vis_dim, k=1))
        self.vis_global = nn.Sequential(nn.Linear(in_channels, vis_dim), nn.ReLU(inplace=True))
        self.vis_embed = nn.Embedding(num_keypoints, 16)
        self.vis_mlp = nn.Sequential(
            nn.Linear(vis_dim + 3 + vis_dim + 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    @property
    def beta(self):
        return F.softplus(self.beta_raw) + BETA_EPS

    def _heatmaps(self, features):
        p2 = features["p2"]
        if self.heatmap_stride == 4:
            return self.heatmap_conv(p2)
        x = self.heatmap_up(self.heatmap_conv(p2))
        s2 = features["s2"]
        skip = self.heatmap_skip(s2)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.heatmap_out(torch.cat([x, skip], dim=1))

    def forward(self, features):
        heatmaps = self._heatmaps(features)
        if heatmaps.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            heatmaps = F.interpolate(
                heatmaps, size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear", align_corners=False,
            )

        beta = self.beta
        offsets, probs = soft_argmax_2d(heatmaps, beta)
        visibility = self._visibility(features, heatmaps, offsets, probs)

        return {
            "heatmaps": heatmaps,
            "offsets": offsets,
            "visibility": visibility,
            "beta": beta,
        }

    def _visibility(self, features, heatmaps, offsets, probs):
        B, K, H, W = heatmaps.shape
        with torch.no_grad():
            flat = heatmaps.detach().float().reshape(B, K, H * W)
            peak_logit = flat.max(dim=-1).values
            log_ratio = torch.log(probs.detach().max(dim=-1).values * (H * W) + 1e-6)
            gx = (torch.arange(W, device=flat.device, dtype=torch.float32) + 0.5) / W
            gy = (torch.arange(H, device=flat.device, dtype=torch.float32) + 0.5) / H
            p = probs.detach().view(B, K, H, W)
            mu = offsets.detach().float()
            var_x = (p.sum(dim=2) * (gx - mu[..., 0:1]) ** 2).sum(-1)
            var_y = (p.sum(dim=3) * (gy - mu[..., 1:2]) ** 2).sum(-1)
            spread = torch.sqrt(var_x + var_y + 1e-12)
            stats = torch.stack([peak_logit, log_ratio, spread], dim=-1)  # (B, K, 3)
            grid = (mu * 2.0 - 1.0).unsqueeze(2)  # (B, K, 1, 2) for align_corners=False

        local_map = self.vis_reduce(features["p2"])  # (B, 64, h, w)
        local = F.grid_sample(
            local_map, grid.to(local_map.dtype), mode="bilinear",
            padding_mode="border", align_corners=False,
        )  # (B, 64, K, 1)
        local = local.squeeze(-1).transpose(1, 2)  # (B, K, 64)

        glob = self.vis_global(features["p5"].mean(dim=(2, 3)))  # (B, 64)
        glob = glob.unsqueeze(1).expand(-1, K, -1)
        emb = self.vis_embed.weight.unsqueeze(0).expand(B, -1, -1)

        x = torch.cat([local, stats.to(local.dtype), glob, emb.to(local.dtype)], dim=-1)
        return self.vis_mlp(x).squeeze(-1)  # (B, K)

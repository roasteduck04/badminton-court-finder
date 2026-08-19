import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointHead(nn.Module):
    """Keypoint detection head with heatmaps, soft-argmax coordinates, and visibility.

    Produces:
        heatmaps: (B, num_keypoints, heatmap_size, heatmap_size)
        offsets: (B, num_keypoints, 2) — normalized [0,1] coordinates via soft-argmax
        visibility: (B, num_keypoints) — probability each keypoint is visible
    """

    def __init__(self, in_channels=256, num_keypoints=30, heatmap_size=160,
                 soft_argmax_beta=10.0):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size
        self.beta = nn.Parameter(torch.tensor(soft_argmax_beta))

        # Heatmap branch — operates on P2 (highest resolution)
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1),
        )

        # Visibility branch — fuse P2-P5 features for richer global context
        self.vis_reduce = nn.ModuleDict({
            f"p{i}": nn.Sequential(
                nn.Conv2d(in_channels, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
            )
            for i in range(2, 6)
        })
        # 4 scales * 64 channels * 4x4 spatial = 4096
        fused_dim = 4 * 64 * 4 * 4
        self.vis_fc = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_keypoints),
        )

    def forward(self, features):
        p2 = features["p2"]  # (B, 256, H/4, W/4)

        heatmaps = self.heatmap_conv(p2)  # (B, K, H/4, W/4)

        # Ensure heatmap is at target resolution
        if heatmaps.shape[2] != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )

        offsets = self._soft_argmax(heatmaps)  # (B, K, 2) in [0,1]

        # Visibility — fuse multi-scale FPN features
        vis_parts = []
        for key in ["p2", "p3", "p4", "p5"]:
            reduced = self.vis_reduce[key](features[key])  # (B, 64, 4, 4)
            vis_parts.append(reduced.flatten(1))  # (B, 1024)
        vis_fused = torch.cat(vis_parts, dim=1)  # (B, 4096)
        visibility = self.vis_fc(vis_fused)  # (B, K)

        return {
            "heatmaps": heatmaps,
            "offsets": offsets,
            "visibility": visibility,
        }

    def _soft_argmax(self, heatmaps):
        """Extract normalized [0,1] coordinates from heatmaps via spatial soft-argmax.

        Applies softmax over the flattened spatial dimensions to get a probability
        distribution, then computes the expected x and y as weighted sums over a
        normalized coordinate grid. The learnable beta parameter controls sharpness:
        higher beta concentrates weight on the peak, lower beta spreads it out.
        """
        B, K, H, W = heatmaps.shape

        flat = heatmaps.view(B, K, -1)  # (B, K, H*W)
        weights = F.softmax(flat * self.beta, dim=-1)  # (B, K, H*W)

        grid_x = torch.linspace(0, 1, W, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_y = torch.linspace(0, 1, H, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_yy, grid_xx = torch.meshgrid(grid_y, grid_x, indexing="ij")

        grid_xx = grid_xx.reshape(1, 1, -1)  # (1, 1, H*W)
        grid_yy = grid_yy.reshape(1, 1, -1)  # (1, 1, H*W)

        x = (weights * grid_xx).sum(dim=-1)  # (B, K)
        y = (weights * grid_yy).sum(dim=-1)  # (B, K)

        return torch.stack([x, y], dim=-1)  # (B, K, 2)

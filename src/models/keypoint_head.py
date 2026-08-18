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

        # Visibility branch — global features to per-keypoint visibility
        self.visibility_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_keypoints),
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

        visibility = self.visibility_conv(p2)  # (B, K)

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

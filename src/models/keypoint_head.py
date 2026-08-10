import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointHead(nn.Module):
    """Keypoint detection head with heatmaps, offset regression, and visibility.

    Produces:
        heatmaps: (B, num_keypoints, heatmap_size, heatmap_size)
        offsets: (B, num_keypoints, 2) — sub-pixel refinement
        visibility: (B, num_keypoints) — probability each keypoint is visible
    """

    def __init__(self, in_channels=256, num_keypoints=31, heatmap_size=160):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

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

        # Offset regression branch — global average pool then predict
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_keypoints * 2),
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

        heatmaps = self.heatmap_conv(p2)  # (B, 14, H/4, W/4)

        # Ensure heatmap is at target resolution
        if heatmaps.shape[2] != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )

        offsets = self.offset_conv(p2)  # (B, 28)
        offsets = offsets.view(-1, self.num_keypoints, 2)  # (B, 14, 2)

        visibility = self.visibility_conv(p2)  # (B, 14)

        return {
            "heatmaps": heatmaps,
            "offsets": offsets,
            "visibility": visibility,
        }

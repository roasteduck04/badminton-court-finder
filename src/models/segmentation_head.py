import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """Lightweight FPN-based decoder for court line segmentation.

    Fuses multi-scale features and produces a full-resolution binary mask.
    """

    def __init__(self, in_channels=256, image_size=640):
        super().__init__()
        self.image_size = image_size

        # Per-level refinement
        self.refine_p2 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.refine_p3 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.refine_p4 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.refine_p5 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Fusion after concatenation (128 * 4 = 512 channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, features):
        p2 = self.refine_p2(features["p2"])  # (B, 128, H/4, W/4)

        target_size = p2.shape[2:]
        p3 = F.interpolate(self.refine_p3(features["p3"]), size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.refine_p4(features["p4"]), size=target_size, mode="bilinear", align_corners=False)
        p5 = F.interpolate(self.refine_p5(features["p5"]), size=target_size, mode="bilinear", align_corners=False)

        fused = torch.cat([p2, p3, p4, p5], dim=1)  # (B, 512, H/4, W/4)
        out = self.fuse(fused)  # (B, 1, H/4, W/4)

        out = F.interpolate(out, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return out  # (B, 1, H, W) logits

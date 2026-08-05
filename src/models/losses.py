import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation logits."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = torch.sigmoid(pred).reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class CourtVisionLoss(nn.Module):
    """Combined multi-task loss for CourtVisionNet.

    Components:
        seg_loss: BCE + Dice for segmentation
        heatmap_loss: MSE for keypoint heatmaps (visible keypoints only)
        offset_loss: L1 for keypoint offset regression (visible keypoints only)
        visibility_loss: BCE for visibility classification

    Expects:
        pred: dict with "seg_logits" (B,1,H,W), "heatmaps" (B,K,h,w),
              "offsets" (B,K,2), "visibility" (B,K)
        targets: dict with "mask" (B,1,H,W), "heatmaps" (B,K,h,w),
                 "keypoints" (B,K,2), "visibility" (B,K)
    """

    def __init__(
        self,
        seg_weight=1.0,
        heatmap_weight=5.0,
        offset_weight=1.0,
        vis_weight=1.0,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.vis_weight = vis_weight
        self.dice_loss = DiceLoss()

    def forward(self, pred, targets):
        # Segmentation loss: BCE + Dice
        seg_bce = F.binary_cross_entropy_with_logits(
            pred["seg_logits"], targets["mask"]
        )
        seg_dice = self.dice_loss(pred["seg_logits"], targets["mask"])
        seg_loss = seg_bce + seg_dice

        # Visibility mask shared by heatmap/offset losses.
        vis = targets["visibility"]  # (B, K)
        num_visible = vis.sum().clamp(min=1.0)

        # Heatmap loss (MSE), masked to visible keypoints only.
        vis_mask_hm = vis.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        heatmap_diff = (pred["heatmaps"] - targets["heatmaps"]) ** 2
        heatmap_h, heatmap_w = pred["heatmaps"].shape[2], pred["heatmaps"].shape[3]
        heatmap_loss = (heatmap_diff * vis_mask_hm).sum() / (
            num_visible * heatmap_h * heatmap_w
        )

        # Offset loss (L1), masked to visible keypoints only.
        vis_mask_offset = vis.unsqueeze(-1)  # (B, K, 1)
        offset_diff = torch.abs(pred["offsets"] - targets["keypoints"]) * vis_mask_offset
        offset_loss = offset_diff.sum() / (num_visible * 2)

        # Visibility loss (BCE) — computed over all keypoints, visible or not.
        visibility_loss = F.binary_cross_entropy_with_logits(
            pred["visibility"], targets["visibility"]
        )

        total = (
            self.seg_weight * seg_loss
            + self.heatmap_weight * heatmap_loss
            + self.offset_weight * offset_loss
            + self.vis_weight * visibility_loss
        )

        components = {
            "seg_loss": seg_loss.item(),
            "heatmap_loss": heatmap_loss.item(),
            "offset_loss": offset_loss.item(),
            "visibility_loss": visibility_loss.item(),
        }
        return total, components

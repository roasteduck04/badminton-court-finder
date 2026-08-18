import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.court_geometry import (
    CORNER_INDICES,
    COURT_KEYPOINTS_TEMPLATE,
    get_collinear_groups,
)


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


# Precompute template distance ratios for the ratio loss.
_TPL = COURT_KEYPOINTS_TEMPLATE
_COLLINEAR_GROUPS = get_collinear_groups()

# For each collinear group, precompute consecutive distance ratios from template.
# ratio_i = dist(group[0], group[i+1]) / dist(group[0], group[-1])
_RATIO_SPECS = []  # list of (group_indices, expected_ratios)
for group in _COLLINEAR_GROUPS:
    if len(group) < 3:
        continue
    dists = [np.linalg.norm(_TPL[group[i]] - _TPL[group[0]]) for i in range(len(group))]
    total = dists[-1]
    if total < 1e-9:
        continue
    ratios = [d / total for d in dists[1:-1]]  # exclude 0 and 1
    _RATIO_SPECS.append((group, ratios))


class CourtVisionLoss(nn.Module):
    """Combined multi-task loss for CourtVisionNet.

    Components:
        seg_loss: BCE + Dice for segmentation
        heatmap_loss: MSE for keypoint heatmaps (visible keypoints only)
        offset_loss: L1 for keypoint offset regression (visible keypoints only)
        visibility_loss: BCE for visibility classification
        collinear_loss: cross-product area penalty for non-collinear points
        ratio_loss: MSE on distance ratios along collinear groups vs template
        convex_loss: ReLU penalty on non-convex outer corner quadrilateral

    Expects:
        pred: dict with "seg_logits" (B,1,H,W), "heatmaps" (B,K,h,w),
              "offsets" (B,K,2), "visibility" (B,K)
        targets: dict with "mask" (B,1,H,W), "heatmaps" (B,K,h,w),
                 "keypoints" (B,K,2), "visibility" (B,K)
    """

    def __init__(
        self,
        seg_weight=1.0,
        heatmap_weight=100.0,
        offset_weight=1.0,
        vis_weight=1.0,
        collinear_weight=0.0,
        ratio_weight=0.0,
        convex_weight=0.0,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.vis_weight = vis_weight
        self.collinear_weight = collinear_weight
        self.ratio_weight = ratio_weight
        self.convex_weight = convex_weight
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

        # Geometric consistency losses — computed on pred["offsets"] (B, K, 2),
        # gated by target visibility so unseen keypoints don't inject noise.
        # These assume the full NUM_KEYPOINTS=30 layout (indices reference
        # specific court keypoints), so skip them when the caller passes a
        # smaller/non-standard keypoint set (e.g. toy tests) or the
        # corresponding weight is zero — both semantically a no-op anyway.
        kps = pred["offsets"]  # (B, K, 2)
        num_kps = kps.shape[1]
        max_group_idx = max(i for group in _COLLINEAR_GROUPS for i in group)

        zero = torch.tensor(0.0, device=kps.device, dtype=kps.dtype)
        if self.collinear_weight != 0.0 and num_kps > max_group_idx:
            collinear_loss = self._collinear_loss(kps, vis)
        else:
            collinear_loss = zero
        if self.ratio_weight != 0.0 and num_kps > max_group_idx:
            ratio_loss = self._ratio_loss(kps, vis)
        else:
            ratio_loss = zero
        if self.convex_weight != 0.0 and num_kps > max(CORNER_INDICES):
            convex_loss = self._convex_loss(kps, vis)
        else:
            convex_loss = zero

        total = (
            self.seg_weight * seg_loss
            + self.heatmap_weight * heatmap_loss
            + self.offset_weight * offset_loss
            + self.vis_weight * visibility_loss
            + self.collinear_weight * collinear_loss
            + self.ratio_weight * ratio_loss
            + self.convex_weight * convex_loss
        )

        components = {
            "seg_loss": seg_loss.item(),
            "heatmap_loss": heatmap_loss.item(),
            "offset_loss": offset_loss.item(),
            "visibility_loss": visibility_loss.item(),
            "collinear_loss": collinear_loss.item(),
            "ratio_loss": ratio_loss.item(),
            "convex_loss": convex_loss.item(),
        }
        return total, components

    @staticmethod
    def _collinear_loss(kps, vis):
        """Cross-product area penalty for non-collinear keypoints.

        For every consecutive triplet within a collinear group (same court
        line), the signed parallelogram area of (p1, p2, p3) is zero iff the
        three points are exactly collinear. Squaring and averaging that area
        gives a differentiable penalty for deviation from the line.
        """
        total = torch.tensor(0.0, device=kps.device, dtype=kps.dtype)
        count = 0
        for group in _COLLINEAR_GROUPS:
            for b in range(kps.shape[0]):
                visible_in_group = [i for i in group if vis[b, i] > 0.5]
                if len(visible_in_group) < 3:
                    continue
                for t in range(len(visible_in_group) - 2):
                    i, j, k = (
                        visible_in_group[t],
                        visible_in_group[t + 1],
                        visible_in_group[t + 2],
                    )
                    p1 = kps[b, i]
                    p2 = kps[b, j]
                    p3 = kps[b, k]
                    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (
                        p3[0] - p1[0]
                    )
                    total = total + cross**2
                    count += 1
        if count == 0:
            return total
        return total / count

    @staticmethod
    def _ratio_loss(kps, vis):
        """MSE on distance ratios along collinear groups vs template ratios.

        Distance ratios are scale/translation invariant, so this constrains
        the *spacing* of keypoints along each court line to match the known
        real-world court geometry, independent of overall court size/position
        in the image.
        """
        total = torch.tensor(0.0, device=kps.device, dtype=kps.dtype)
        count = 0
        for group, expected_ratios in _RATIO_SPECS:
            for b in range(kps.shape[0]):
                if not all(vis[b, i] > 0.5 for i in group):
                    continue
                p_first = kps[b, group[0]]
                p_last = kps[b, group[-1]]
                total_dist = torch.norm(p_last - p_first).clamp(min=1e-8)
                for idx, expected in enumerate(expected_ratios):
                    p_mid = kps[b, group[idx + 1]]
                    dist = torch.norm(p_mid - p_first)
                    ratio = dist / total_dist
                    total = total + (ratio - expected) ** 2
                    count += 1
        if count == 0:
            return total
        return total / count

    @staticmethod
    def _convex_loss(kps, vis):
        """ReLU penalty on non-convex outer corner quadrilateral.

        Walks the four outer corners (CORNER_INDICES, cyclic TL->TR->BR->BL)
        and penalizes any turn whose cross product has the wrong sign, i.e.
        any concave/self-intersecting ("bowtie") configuration.
        """
        corner_idx = CORNER_INDICES  # [0, 25, 29, 4] cyclic
        total = torch.tensor(0.0, device=kps.device, dtype=kps.dtype)
        count = 0
        for b in range(kps.shape[0]):
            if not all(vis[b, i] > 0.5 for i in corner_idx):
                continue
            corners = torch.stack([kps[b, i] for i in corner_idx])  # (4, 2)
            for c in range(4):
                p1 = corners[c]
                p2 = corners[(c + 1) % 4]
                p3 = corners[(c + 2) % 4]
                cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (
                    p3[0] - p2[0]
                )
                # All cross products should be positive (CCW) — penalize negative
                total = total + F.relu(-cross)
            count += 1
        if count == 0:
            return total
        return total / count

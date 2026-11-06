import torch.nn as nn

from src.models.backbone import CourtBackbone
from src.models.segmentation_head import SegmentationHead
from src.models.keypoint_head import KeypointHead


def resolve_heatmap_stride(image_size, heatmap_size, heatmap_stride=None):
    """Return the heatmap stride (2 or 4) for the given sizes.

    With ``heatmap_stride=None`` it is derived from image_size / heatmap_size
    (2 for 320 px heatmaps from 640 px input, 4 for 160 px). An explicit
    stride must agree with heatmap_size.
    """
    if heatmap_stride is None:
        return 2 if image_size / heatmap_size <= 3 else 4
    if heatmap_stride not in (2, 4):
        raise ValueError(f"heatmap_stride must be 2 or 4, got {heatmap_stride}")
    if image_size // heatmap_stride != heatmap_size:
        raise ValueError(
            f"heatmap_size={heatmap_size} is inconsistent with image_size={image_size} "
            f"and heatmap_stride={heatmap_stride} (expected {image_size // heatmap_stride})"
        )
    return heatmap_stride


class CourtVisionNet(nn.Module):
    """Full CourtVisionNet: shared backbone + dual segmentation/keypoint heads.

    Combines:
        - CourtBackbone: ResNet-50 + FPN feature extractor (normalizes its
          [0, 1] input internally)
        - SegmentationHead: full-resolution court-line mask
        - KeypointHead: heatmaps, soft-argmax coordinates and visibility

    Forward output: seg_logits (B,1,H,W), heatmaps (B,K,h,w) logits,
    offsets (B,K,2) normalized soft-argmax coords, visibility (B,K) logits,
    beta () soft-argmax sharpness.
    """

    def __init__(
        self,
        in_channels=7,
        num_keypoints=30,
        image_size=640,
        heatmap_size=160,
        pretrained=True,
        heatmap_stride=None,
        soft_argmax_beta=10.0,
        normalize_input=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.heatmap_stride = resolve_heatmap_stride(image_size, heatmap_size, heatmap_stride)
        self._backbone_frozen = False

        self.backbone = CourtBackbone(
            in_channels=in_channels, pretrained=pretrained, normalize_input=normalize_input,
        )
        self.seg_head = SegmentationHead(in_channels=256, image_size=image_size)
        self.kpt_head = KeypointHead(
            in_channels=256, num_keypoints=num_keypoints, heatmap_size=heatmap_size,
            soft_argmax_beta=soft_argmax_beta, heatmap_stride=self.heatmap_stride,
        )

    def forward(self, x):
        features = self.backbone(x)
        seg_logits = self.seg_head(features, out_size=x.shape[-2:])
        kpt_out = self.kpt_head(features)

        return {
            "seg_logits": seg_logits,
            "heatmaps": kpt_out["heatmaps"],
            "offsets": kpt_out["offsets"],
            "visibility": kpt_out["visibility"],
            "beta": kpt_out["beta"],
        }

    # ------------------------------------------------------------------
    # Freezing and parameter groups
    # ------------------------------------------------------------------

    def _pretrained_parameters(self):
        for module in self.backbone.pretrained_modules():
            yield from module.parameters()

    def freeze_backbone(self):
        """Freeze the pretrained ResNet stem + stages (not the FPN or heads).

        Their BatchNorm layers are put in eval mode and stay there across
        ``model.train()`` until ``unfreeze_backbone()``.
        """
        for param in self._pretrained_parameters():
            param.requires_grad = False
        self._backbone_frozen = True
        self._set_frozen_bn_eval()

    def unfreeze_backbone(self):
        """Make the pretrained stem + stages trainable again."""
        for param in self._pretrained_parameters():
            param.requires_grad = True
        self._backbone_frozen = False
        if self.training:
            for module in self.backbone.pretrained_modules():
                module.train()

    def _set_frozen_bn_eval(self):
        for module in self.backbone.pretrained_modules():
            for m in module.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode and self._backbone_frozen:
            self._set_frozen_bn_eval()
        return self

    def parameter_groups(self):
        """{"backbone": pretrained stem+stage params, "other": everything else}.

        Every parameter appears exactly once, regardless of requires_grad.
        """
        backbone_ids = {id(p) for p in self._pretrained_parameters()}
        backbone = [p for p in self.parameters() if id(p) in backbone_ids]
        other = [p for p in self.parameters() if id(p) not in backbone_ids]
        return {"backbone": backbone, "other": other}

    def no_weight_decay_params(self):
        """Parameters that should get weight_decay=0 (the soft-argmax beta)."""
        return [self.kpt_head.beta_raw]

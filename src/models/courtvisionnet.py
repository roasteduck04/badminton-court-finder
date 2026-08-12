import torch.nn as nn

from src.models.backbone import CourtBackbone
from src.models.segmentation_head import SegmentationHead
from src.models.keypoint_head import KeypointHead


class CourtVisionNet(nn.Module):
    """Full CourtVisionNet: shared backbone + dual segmentation/keypoint heads.

    Combines:
        - CourtBackbone: ResNet-50 + FPN feature extractor
        - SegmentationHead: full-resolution court-line mask
        - KeypointHead: heatmaps, offsets, and visibility for court keypoints
    """

    def __init__(
        self,
        in_channels=7,
        num_keypoints=30,
        image_size=640,
        heatmap_size=160,
        pretrained=True,
    ):
        super().__init__()
        self.backbone = CourtBackbone(in_channels=in_channels, pretrained=pretrained)
        self.seg_head = SegmentationHead(in_channels=256, image_size=image_size)
        self.kpt_head = KeypointHead(
            in_channels=256, num_keypoints=num_keypoints, heatmap_size=heatmap_size
        )

    def forward(self, x):
        features = self.backbone(x)
        seg_logits = self.seg_head(features)
        kpt_out = self.kpt_head(features)

        return {
            "seg_logits": seg_logits,
            "heatmaps": kpt_out["heatmaps"],
            "offsets": kpt_out["offsets"],
            "visibility": kpt_out["visibility"],
        }

    def freeze_backbone(self):
        """Freeze backbone parameters so only the heads are trained.

        Useful for a warm-up phase where the pretrained ResNet-50 features
        are kept fixed while the segmentation/keypoint heads are trained
        from scratch.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters, e.g. for a later fine-tuning phase."""
        for param in self.backbone.parameters():
            param.requires_grad = True

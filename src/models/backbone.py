import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork


class CourtBackbone(nn.Module):
    """ResNet-50 + FPN backbone with configurable input channels.

    When in_channels > 3 and pretrained=True, the first conv layer is expanded:
    RGB channels get pretrained weights, extra channels get He-initialized weights.
    """

    def __init__(self, in_channels=7, pretrained=True):
        super().__init__()

        weights = "IMAGENET1K_V2" if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Modify first conv for multi-channel input
        original_conv = resnet.conv1
        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                # Copy pretrained weights for RGB channels
                with torch.no_grad():
                    new_conv.weight[:, :3] = original_conv.weight
                    # He-initialize extra channels
                    nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode="fan_out", nonlinearity="relu")
            resnet.conv1 = new_conv

        # Extract layer stages
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # stride 4,  256 channels
        self.layer2 = resnet.layer2  # stride 8,  512 channels
        self.layer3 = resnet.layer3  # stride 16, 1024 channels
        self.layer4 = resnet.layer4  # stride 32, 2048 channels

        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
        )

    def forward(self, x):
        c1 = self.stem(x)      # stride 4
        c2 = self.layer1(c1)   # stride 4,  256 ch
        c3 = self.layer2(c2)   # stride 8,  512 ch
        c4 = self.layer3(c3)   # stride 16, 1024 ch
        c5 = self.layer4(c4)   # stride 32, 2048 ch

        fpn_input = {
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
        }
        fpn_output = self.fpn(fpn_input)

        return {
            "p2": fpn_output["c2"],  # (B, 256, H/4, W/4)
            "p3": fpn_output["c3"],  # (B, 256, H/8, W/8)
            "p4": fpn_output["c4"],  # (B, 256, H/16, W/16)
            "p5": fpn_output["c5"],  # (B, 256, H/32, W/32)
        }

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork

# ImageNet statistics for the RGB channels (torchvision ResNet weights).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# The derived channels (grayscale, CLAHE, Canny, court-colour mask) are all in
# [0, 1]; mean 0.5 / std 0.25 maps them to roughly [-2, 2], the same range as
# ImageNet-normalized RGB, without needing dataset statistics.
EXTRA_CHANNEL_MEAN = 0.5
EXTRA_CHANNEL_STD = 0.25


def input_normalization(in_channels):
    """Per-channel (mean, std) lists used by the model's input normalization."""
    mean = list(IMAGENET_MEAN[:min(3, in_channels)])
    std = list(IMAGENET_STD[:min(3, in_channels)])
    mean += [EXTRA_CHANNEL_MEAN] * max(0, in_channels - 3)
    std += [EXTRA_CHANNEL_STD] * max(0, in_channels - 3)
    return mean, std


def adapt_input_conv(rgb_weight, in_channels):
    """Build first-conv weights for ``in_channels`` from pretrained RGB weights.

    The RGB filters are copied and the extra-channel filters are initialized
    to the mean of the RGB filters; all filters are then scaled by
    3 / in_channels (the timm ``adapt_input_conv`` convention). For an input
    whose channels carry the same signal (e.g. grey content, where RGB,
    grayscale and CLAHE agree) the stem output is then exactly what the
    pretrained network saw, so the pretrained BN statistics stay valid. This
    keeps the stem scale far closer to pretrained than Kaiming noise on the
    extra channels, which added a large random term to every stem activation.
    """
    out_c, _, kh, kw = rgb_weight.shape
    if in_channels == 3:
        return rgb_weight.clone()
    if in_channels < 3:
        mean = rgb_weight.mean(dim=1, keepdim=True)
        return mean.repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
    w = rgb_weight.new_empty(out_c, in_channels, kh, kw)
    w[:, :3] = rgb_weight
    w[:, 3:] = rgb_weight.mean(dim=1, keepdim=True)
    return w * (3.0 / in_channels)


class CourtBackbone(nn.Module):
    """ResNet-50 + FPN backbone with configurable input channels.

    Expects inputs in [0, 1] (the data pipeline does no normalization) and
    normalizes internally when ``normalize_input`` is True: RGB with ImageNet
    mean/std, extra channels with mean 0.5 / std 0.25.

    The pretrained part (``stem`` + ``layer1..4``) is what
    ``pretrained_modules()`` returns and what the model freezes; the FPN is
    randomly initialized and always trainable. State-dict keys are unchanged
    from earlier versions (``backbone.stem.*``, ``backbone.fpn.*``) so old
    checkpoints still map onto this module.

    Returns P2-P5 FPN features plus ``s2``, the stride-2 stem activation
    (conv1+bn1+relu, 64 channels) used by the stride-2 heatmap head.
    """

    def __init__(self, in_channels=7, pretrained=True, normalize_input=True):
        super().__init__()
        self.in_channels = in_channels
        self.normalize_input = normalize_input

        weights = "IMAGENET1K_V2" if pretrained else None
        resnet = models.resnet50(weights=weights)

        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(adapt_input_conv(resnet.conv1.weight, in_channels))
            resnet.conv1 = new_conv

        mean, std = input_normalization(in_channels)
        # Non-persistent: not part of the state dict, so checkpoints from any
        # version load without missing/unexpected keys.
        self.register_buffer("input_mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

        # stem[0..2] -> stride 2 (s2 feature), stem[3] = maxpool -> stride 4
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # stride 4,  256 channels
        self.layer2 = resnet.layer2  # stride 8,  512 channels
        self.layer3 = resnet.layer3  # stride 16, 1024 channels
        self.layer4 = resnet.layer4  # stride 32, 2048 channels

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
        )

    def pretrained_modules(self):
        """The ImageNet-pretrained part: stem + the four ResNet stages."""
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def normalize(self, x):
        if not self.normalize_input:
            return x
        return (x - self.input_mean.to(x.dtype)) / self.input_std.to(x.dtype)

    def forward(self, x):
        x = self.normalize(x)
        s2 = self.stem[2](self.stem[1](self.stem[0](x)))  # stride 2, 64 ch
        c1 = self.stem[3](s2)  # stride 4
        c2 = self.layer1(c1)   # stride 4,  256 ch
        c3 = self.layer2(c2)   # stride 8,  512 ch
        c4 = self.layer3(c3)   # stride 16, 1024 ch
        c5 = self.layer4(c4)   # stride 32, 2048 ch

        fpn_output = self.fpn({"c2": c2, "c3": c3, "c4": c4, "c5": c5})

        return {
            "s2": s2,                # (B, 64, H/2, W/2)
            "p2": fpn_output["c2"],  # (B, 256, H/4, W/4)
            "p3": fpn_output["c3"],  # (B, 256, H/8, W/8)
            "p4": fpn_output["c4"],  # (B, 256, H/16, W/16)
            "p5": fpn_output["c5"],  # (B, 256, H/32, W/32)
        }

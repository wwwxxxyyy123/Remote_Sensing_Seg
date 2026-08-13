import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU6 (torchvision MobileNetV2 style).

    The optional `dilation` argument only affects the forward sampling pattern
    (atrous convolution) and does NOT change the weight tensor shape, so
    torchvision pretrained state dict keys still align directly.
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
        # padding must scale with dilation to preserve spatial size when stride=1
        padding = (kernel_size - 1) // 2 * dilation
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      groups=groups, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """Inverted Residual block with linear bottleneck (MobileNetV2).

    Module structure mirrors torchvision so pretrained state dict keys align
    directly: self.conv is a Sequential whose entries are ConvBNReLU sub-modules
    (for the pw-expand and dw) followed by a plain Conv2d + BatchNorm2d (pw-linear).

    The `dilation` argument is only applied to the 3x3 depthwise convolution;
    the 1x1 pointwise convolutions are dilation-agnostic.  Because dilation
    never changes the weight tensor shape, pretrained weights still load.
    """
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw-expand (ConvBNReLU -> keys conv.0.0 / conv.0.1)
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        # dw (ConvBNReLU -> keys conv.{0|1}.0 / conv.{0|1}.1)
        # Only the depthwise conv uses dilation (DeepLabV3+ atrous strategy).
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3,
                                 stride=stride, groups=hidden_dim, dilation=dilation))
        # pw-linear (no activation): Conv2d + BatchNorm2d
        layers.append(nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


# MobileNetV2 InvertedResidual configuration: (expand_ratio t, out_channels c, num_blocks n, stride s)
# The final (6, 320, 1, 1) stage is omitted; c4 uses the 160-channel stage to match the decoder.
#
# Stage -> features index mapping (features.0 is the stem):
#   stage 0: (1,  16, 1, 1) -> features.1
#   stage 1: (6,  24, 2, 2) -> features.2-3
#   stage 2: (6,  32, 3, 2) -> features.4-6
#   stage 3: (6,  64, 4, 2) -> features.7-10
#   stage 4: (6,  96, 3, 1) -> features.11-13
#   stage 5: (6, 160, 3, 2) -> features.14-16
MOBILENETV2_CFGS = [
    (1,  16, 1, 1),   # features.1                       -> H/2  (16)
    (6,  24, 2, 2),   # features.2 (s=2), features.3     -> H/4  (24)
    (6,  32, 3, 2),   # features.4 (s=2), features.5, 6  -> H/8  (32)
    (6,  64, 4, 2),   # features.7 (s=2), features.8~10  -> H/16 (64)
    (6,  96, 3, 1),   # features.11~13                   -> H/16 (96)
    (6, 160, 3, 2),   # features.14 (s=2), features.15,16-> H/32 (160) / H/16 with OS=16
]


def _atrous_strategy(output_stride):
    """DeepLabV3+ atrous convolution strategy for MobileNetV2.

    Returns:
        dilations: dict {stage_idx -> dilation_rate} for stages whose depthwise
            convolutions should use atrous convolution.
        stride_overrides: dict {stage_idx -> new_stride} for the first block of
            stages whose stride-2 downsample should be replaced by stride-1.

    Rationale (DeepLabV3+ paper, Sec. 3): when lowering output_stride, the
    last downsampling operation(s) are removed (stride 2 -> 1) and subsequent
    depthwise convolutions use dilation to preserve the original receptive
    field.  Dilation does NOT change the weight tensor shape, so torchvision
    pretrained weights still load directly.
    """
    if output_stride == 32:
        # Original MobileNetV2 downsampling, no atrous conv.
        return {}, {}
    if output_stride == 16:
        # Keep features.14 at H/16 (stride 2->1); the 160-channel stage (stage 5)
        # uses dilation=2 in its depthwise convs to keep the original receptive field.
        return {5: 2}, {5: 1}
    if output_stride == 8:
        # Keep features.7 at H/8 (stride 2->1); stages 3 & 4 (64/96-channel) use
        # dilation=2; stage 5 (160-channel) keeps stride 2->1 with dilation=4.
        return {3: 2, 4: 2, 5: 4}, {3: 1, 5: 1}
    raise ValueError(f"output_stride must be 8, 16 or 32, got {output_stride}")


class MobileNetV2(nn.Module):
    """MobileNetV2 encoder outputting 4-scale feature maps.

    Output scales (with output_stride=16):
        c1: [N, 16,  H/2,  W/2]   (low-level, after features.1)
        c2: [N, 24,  H/4,  W/4]   (after features.3)
        c3: [N, 32,  H/8,  W/8]   (after features.6)
        c4: [N, 160, H/16, W/16]  (high-level, after features.16)

    DeepLabV3+ atrous strategy (only the 3x3 depthwise convs use dilation):
      - output_stride=16: features.14 stride 2->1, dilation=2 on stage 5
        (features.14/15/16).
      - output_stride=8:  additionally features.7 stride 2->1, dilation=2 on
        stages 3 & 4 (features.7~13) and dilation=4 on stage 5 (features.14~16).

    Dilation and stride overrides do NOT change weight tensor shapes, so the
    module structure still mirrors torchvision's mobilenet_v2 and pretrained
    state dict keys (features.*) align directly.
    """
    def __init__(self, input_channels=3, output_stride=16):
        super().__init__()
        assert output_stride in (8, 16, 32), f"output_stride must be 8, 16 or 32, got {output_stride}"
        self.output_stride = output_stride
        self.features = nn.Sequential()

        # features.0: stem (input -> 32, k=3, s=2, ReLU6) -> H/2
        self.features.append(ConvBNReLU(input_channels, 32, kernel_size=3, stride=2))

        # Resolve the atrous strategy once for the whole backbone.
        dilations, stride_overrides = _atrous_strategy(output_stride)

        # features.1 ~ features.16: InvertedResidual blocks
        input_ch = 32
        for stage_idx, (t, c, n, s) in enumerate(MOBILENETV2_CFGS):
            dilation = dilations.get(stage_idx, 1)
            for i in range(n):
                stride = s if i == 0 else 1
                # Override the first block's stride to keep output_stride small.
                if i == 0 and stage_idx in stride_overrides:
                    stride = stride_overrides[stage_idx]
                self.features.append(
                    InvertedResidual(input_ch, c, stride, t, dilation=dilation)
                )
                input_ch = c

    def forward(self, x):
        x = self.features[0](x)
        x = self.features[1](x)

        x = self.features[2](x)
        x = self.features[3](x)   # [N, 24, H/4, W/4]
        c1 = x

        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)   # [N, 32, H/8, W/8]
        c2 = x

        for i in range(7, 14):
            x = self.features[i](x)
        c3 = x  # [N, 96, H/16, W/16]

        for i in range(14, 17):
            x = self.features[i](x)
        c4 = x  # [N, 160, H/16, W/16]

        return c1, c2, c3, c4

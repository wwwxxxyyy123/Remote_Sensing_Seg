from .block import ASPP
import torch.nn as nn
import torch
import torch.nn.functional as F
from .MobileNetV2 import MobileNetV2

class Encoder(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.mobilenet_v2 = MobileNetV2(input_channels)
        self.aspp = ASPP(input_channels=160, output_channels=256)
    def forward(self, x):
        low_level, c2, c3, high_level = self.mobilenet_v2(x)
        high_level = self.aspp(high_level)
        return high_level, low_level # [N, 256, H/16, W/16], [N, 24, H/4, W/4]
    
class Decoder(nn.Module):
    def __init__(self, aspp_channels=256, low_channels=24, low_proj_channels=48, decode_channels=256, out_channels=21):
        super().__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, low_proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_proj_channels),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(aspp_channels + low_proj_channels, decode_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(decode_channels, out_channels, kernel_size=1)
        )

    def forward(self, high_level, low_level):
        # high_level: [N, 256, H/16, W/16]  (ASPP output)
        # low_level:  [N, 24, H/4, W/4]    (layer1 output)

        high_up = F.interpolate(
            high_level, 
            size=low_level.shape[2:],
            mode='bilinear', 
            align_corners=False
        )  # -> [N, 256, H/4, W/4]

        low_proc = self.low_conv(low_level)  # -> [N, 48, H/4, W/4]

        fused = torch.cat([high_up, low_proc], dim=1)  # -> [N, 304, H/4, W/4]

        out = self.final_conv(fused)  # -> [N, out_channels, H/4, W/4]

        out = F.interpolate(
            out, 
            scale_factor=4,
            mode='bilinear',
            align_corners=False
        )  # -> [N, out_channels, H, W]
        
        return out

class DeepLabV3Plus(nn.Module):
    def __init__(self, input_channels=3, cls_num=21):
        super().__init__()
        self.encoder = Encoder(input_channels)
        self.decoder = Decoder(aspp_channels=256, low_channels=24, out_channels=cls_num)

    def forward(self, x):
        high_level, low_level = self.encoder(x)
        out = self.decoder(high_level, low_level)
        return out
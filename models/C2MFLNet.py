from .block import ASPP, CBAM, CoordAtt
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
        return high_level, low_level, c2, c3 # [N, 256, H/16, W/16], [N, 24, H/4, W/4], [N, 32, H/8, W/8], [N, 96, H/16, W/16]
    

class Decoder(nn.Module):
    def __init__(self, aspp_channels=256, low_channels=24, c2_channels=32, 
                 low_proj_channels=48, mid_proj_channels=64, unified_channels=256, decode_channels=256, out_channels=21):
        super().__init__()

        # Projection layers
        self.proj_c2 = nn.Sequential(
            nn.Conv2d(c2_channels, mid_proj_channels, 1, bias=False),
            nn.BatchNorm2d(mid_proj_channels),
            nn.ReLU(inplace=True)
        )
        self.proj_low = nn.Sequential(
            nn.Conv2d(low_channels, low_proj_channels, 1, bias=False),
            nn.BatchNorm2d(low_proj_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion modules: Concatenateation → unified_channels
        # First fusion: 256 + c2_proj (64) → 320 → 256
        self.fuse_8 = nn.Sequential(
            nn.Conv2d(aspp_channels + mid_proj_channels, unified_channels, 1, bias=False),
            nn.BatchNorm2d(unified_channels),
            nn.ReLU(inplace=True)
        )

        # Second fusion: 256 + low_proj (48) → 304 → 256
        self.fuse_4 = nn.Sequential(
            nn.Conv2d(unified_channels + low_proj_channels, unified_channels, 1, bias=False),
            nn.BatchNorm2d(unified_channels),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(unified_channels, decode_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(decode_channels, decode_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decode_channels, out_channels, 1)
        )

    def forward(self, high_level, low_level, c2):
        # high_level: [N, 256, H/16, W/16]
        # c2:         [N, 32,  H/8,  W/8]
        # low_level:  [N, 24,  H/4,  W/4]

        # Projection layers
        c2_proc = self.proj_c2(c2) # [N, 64, H/8, W/8]
        low_proc = self.proj_low(low_level) # [N, 48, H/4, W/4]

        # First fusion H/8
        up_8 = F.interpolate(high_level, size=c2.shape[2:], mode='bilinear', align_corners=False)
        fused_8 = torch.cat([up_8, c2_proc], dim=1)          # [N, 320, H/8, W/8]
        fused_8 = self.fuse_8(fused_8)                       # [N, 256, H/8, W/8]

        # Second fusion H/4
        up_4 = F.interpolate(fused_8, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        fused_4 = torch.cat([up_4, low_proc], dim=1)         # [N, 304, H/4, W/4]
        fused_4 = self.fuse_4(fused_4)                       # [N, 256, H/4, W/4]

        # Final prediction head
        out = self.final_conv(fused_4)                       # [N, out_channels, H/4, W/4]
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        return out

class C2MFLNet(nn.Module):
    def __init__(self, input_channels=3, cls_num=21):
        super().__init__()
        self.encoder = Encoder(input_channels)
        self.decoder = Decoder(aspp_channels=256, low_channels=24, 
                               c2_channels=32, out_channels=cls_num)

    def forward(self, x):
        high_level, low_level, c2, c3 = self.encoder(x)
        out = self.decoder(high_level, low_level, c2)
        return out
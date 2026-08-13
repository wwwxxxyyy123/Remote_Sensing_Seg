from .block import DoubleConv, Down, Up, OutConv
import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, input_channels, cls_num, bilinear=False):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.cls_num = cls_num
        self.bilinear = bilinear

        # ---------- Encoder ----------
        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # ---------- Decoder ----------
        self.up1 = Up(1024 // factor, 512, 512 // factor, bilinear)
        self.up2 = Up(512 // factor, 256, 256 // factor, bilinear)
        self.up3 = Up(256 // factor, 128, 128 // factor, bilinear)  
        self.up4 = Up(128 // factor, 64, 64, bilinear)
        self.outc = OutConv(64, cls_num)
    
    def forward(self, X):
        x1 = self.inc(X)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
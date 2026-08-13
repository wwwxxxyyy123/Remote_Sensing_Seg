import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, up_channels, skip_channels, out_channels, bilinear=True):
        """
        Args:
            up_channels: Number of channels in the feature map from the lower layer (needs upsampling)
            skip_channels: Number of channels in the feature map from the skip connection
            out_channels: Number of channels in the output feature map
            bilinear: Whether to use bilinear upsampling
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.concat_channels = up_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(up_channels, up_channels // 2, kernel_size=2, stride=2)
            self.concat_channels = up_channels // 2 + skip_channels
        self.conv = DoubleConv(self.concat_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm adapted for 2D features"""
    def forward(self, x):
        # x: (batch, channels, height, width)
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x

    
class OverlapPatch(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.norm = LayerNorm2d(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
    

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, reduction_ratio=1, drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio
        self.head_dim = dim // num_heads
        
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        
        self.proj = nn.Linear(dim, dim, bias=True)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W, f"Sequence length {N} doesn't match H*W ({H*W})"

        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_2d = self.sr(x_2d)
            H_red = H // self.reduction_ratio
            W_red = W // self.reduction_ratio
            N_red = H_red * W_red
            x_red = x_2d.reshape(B, C, -1).permute(0, 2, 1)
            x_red = self.norm(x_red)
        else:
            x_red = x
            N_red = N

        k = self.k(x_red).reshape(B, N_red, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x_red).reshape(B, N_red, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)

        out = self.proj(out)
        out = self.drop(out)

        return out

class DWConv(nn.Module):
    """Depth-wise Conv wrapper to match official SegFormer weight naming"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.dwconv = DWConv(hidden_features)

        self.fc2 = nn.Linear(hidden_features, out_features)

        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W, f"Sequence length {N} doesn't match H*W ({H*W})"

        x = self.fc1(x)

        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = self.act(x)
        x = x.reshape(B, -1, N).permute(0, 2, 1)

        x = self.fc2(x)
        x = self.drop(x)

        return x
    

class AtrousConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    

class ASPP(nn.Module):
    def __init__(self, input_channels, output_channels=256, dilation=[6, 12, 18]):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_1 = AtrousConv(input_channels, output_channels, 3, dilation[0], dilation[0])
        self.conv3x3_2 = AtrousConv(input_channels, output_channels, 3, dilation[1], dilation[1])
        self.conv3x3_3 = AtrousConv(input_channels, output_channels, 3, dilation[2], dilation[2])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(5 * output_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:] # (H, W)
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear')
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.fusion(x)


class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.act(y)
    

class ACFM(nn.Module):
    def __init__(self, mid_channels, high_channels, out_channels=256):
        super().__init__()
        self.mid_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.high_convt = nn.Sequential(
            nn.ConvTranspose2d(high_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.simam_module = Simam_module()

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, mid_x, high_x):
        mid_x = self.mid_conv(mid_x)
        high_x = self.high_convt(high_x)
        z = mid_x + high_x
        weight = self.simam_module(z)
        fused = self.final_conv(weight * mid_x + (1 - weight) * high_x)
        return fused


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip) 
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result
     
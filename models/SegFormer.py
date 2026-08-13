from .block import OverlapPatch, EfficientSelfAttention, MixFFN
from .SegFormerConfig import SEGFORMER_CONFIGS
import torch.nn as nn
import torch
import torch.nn.functional as F

class MixTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, reduction_ratio, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, reduction_ratio, drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, dim * mlp_ratio, dim, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x
        

class MixTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        reduction_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        num_layers=[2, 2, 2, 2],
        drop_rate=0.0
    ):
        super().__init__()
        self.num_stages = len(embed_dims)
        assert len(embed_dims) == len(num_heads) == len(reduction_ratios) == len(mlp_ratios) == len(num_layers)
        
        self.patch_embeddings = nn.ModuleList()
        self.block = nn.ModuleList()
        self.norm = nn.ModuleList()

        for i in range(self.num_stages):
            if i == 0:
                patch_embed = OverlapPatch(
                    in_channels=input_channels,
                    embed_dim=embed_dims[i],
                    kernel_size=7,
                    stride=4,
                    padding=3
                )
            else:
                patch_embed = OverlapPatch(
                    in_channels=embed_dims[i-1],
                    embed_dim=embed_dims[i],
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            
            stage_blocks = nn.ModuleList([
                MixTransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    reduction_ratio=reduction_ratios[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate
                )
                for _ in range(num_layers[i])
            ])

            self.patch_embeddings.append(patch_embed)
            self.block.append(stage_blocks)
            self.norm.append(nn.LayerNorm(embed_dims[i]))

    def forward(self, x):
        multi_scale_features = []

        for i in range(self.num_stages):
            x = self.patch_embeddings[i](x)
            H, W = x.shape[2], x.shape[3]
            x_flat = x.flatten(2).permute(0, 2, 1)

            for block in self.block[i]:
                x_flat = block(x_flat, H, W)

            x_flat = self.norm[i](x_flat)

            x = x_flat.permute(0, 2, 1).reshape(x_flat.shape[0], -1, H, W)
            multi_scale_features.append(x)

        return multi_scale_features
    
class All_MLP_Decoder(nn.Module):
    def __init__(self, encoder_dims, decoder_dim=256, out_channels=150):
        super().__init__()
        # Channel projection layers
        self.proj_layers = nn.ModuleList()
        for dim in encoder_dims:
            self.proj_layers.append(nn.Conv2d(dim, decoder_dim, kernel_size=1))

        # Feature fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )

        # Classification layer
        self.pred = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)

    def forward(self, multi_scale_features):
        target_h, target_w = multi_scale_features[0].shape[2], multi_scale_features[0].shape[3]
         
        # Channel projection
        proj_features = []
        for i, feat in enumerate(multi_scale_features):
            feat = self.proj_layers[i](feat)
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat, 
                    size=(target_h, target_w), 
                    mode='bilinear'
                )
            proj_features.append(feat)
            
        # Feature fusion
        fused_feat = self.fuse(torch.cat(proj_features, dim=1))
        # Classification
        out = self.pred(fused_feat)
        return out

class SegFormer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        multi_scale_features = self.encoder(x)
        out = self.decoder(multi_scale_features)
        logits = F.interpolate(
            out,
            size=x.shape[2:],       # (H, W)
            mode='bilinear',
            align_corners=False
        )
        return logits
        
        
def SegFormer_init(model_name='b0', input_channels=3, cls_num=150, decoder_dim=256, drop_rate=0.0):
    config = SEGFORMER_CONFIGS[model_name].copy()
    
    # Create encoder
    encoder = MixTransformerEncoder(
        input_channels=input_channels,
        embed_dims=config['embed_dims'],
        num_heads=config['num_heads'],
        reduction_ratios=config['reduction_ratios'],
        mlp_ratios=config['mlp_ratios'],
        num_layers=config['num_layers'],
        drop_rate=drop_rate
    )
    
    # Create decoder
    decoder = All_MLP_Decoder(
        encoder_dims=config['embed_dims'],
        decoder_dim=decoder_dim,
        out_channels=cls_num
    )
    
    # Assemble
    model = SegFormer(encoder, decoder)
    return model

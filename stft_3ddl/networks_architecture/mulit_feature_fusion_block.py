import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
import os
import gc


class ConcatFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)


    def forward(self, feat1, feat2):
        fused = torch.cat([feat1, feat2], dim=1)
        return self.reduce_conv(fused)


class AddFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat1, feat2):
        return feat1 + feat2

class WeightedAddFusion(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: 权重系数，控制feat1与feat2的加权比例
        feat_fused = alpha * feat1 + (1 - alpha) * feat2
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, feat1, feat2):
        return self.alpha * feat1 + (1 - self.alpha) * feat2

class LearnableAddFusion(nn.Module):
    def __init__(self, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, feat1, feat2):
        alpha = torch.clamp(self.alpha, 0.0, 1.0)  # 保证在合理范围内
        return alpha * feat1 + (1 - alpha) * feat2


class SEFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se1 = SELayer(channels, reduction)
        self.se2 = SELayer(channels, reduction)

    def forward(self, feat1, feat2):
        return self.se1(feat1) + self.se2(feat2)


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.size()
        x1 = feat1.flatten(2).transpose(1, 2)  # (b, hw, c)
        x2 = feat2.flatten(2).transpose(1, 2)
        attn_output, _ = self.attn(x1, x2, x2)
        return attn_output.transpose(1, 2).view(b, c, h, w)


class LiteCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, downsample=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.downsample = downsample

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.shape
        if self.downsample:
            feat1_ds = F.avg_pool2d(feat1, 2)
            feat2_ds = F.avg_pool2d(feat2, 2)
        else:
            feat1_ds, feat2_ds = feat1, feat2

        b, c, h_ds, w_ds = feat1_ds.shape
        x1 = feat1_ds.flatten(2).transpose(1, 2)  # (b, hw, c)
        x2 = feat2_ds.flatten(2).transpose(1, 2)
        attn_output, _ = self.attn(x1, x2, x2)
        attn_output = attn_output.transpose(1, 2).view(b, c, h_ds, w_ds)

        if self.downsample:
            attn_output = F.interpolate(attn_output, size=(h, w), mode='bilinear', align_corners=False)

        return attn_output


class AdaptiveCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, threshold=4096):  # threshold = H×W 阈值
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.threshold = threshold

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.shape
        area = h * w

        # 根据输入尺寸判断是否下采样
        if area > self.threshold:
            feat1 = F.avg_pool2d(feat1, kernel_size=2)
            feat2 = F.avg_pool2d(feat2, kernel_size=2)
            downsampled = True
        else:
            downsampled = False

        b, c, h, w = feat1.shape
        q = feat1.flatten(2).transpose(1, 2)
        k = feat2.flatten(2).transpose(1, 2)
        v = k
        out, _ = self.attn(q, k, v)
        out = out.transpose(1, 2).view(b, c, h, w)

        if downsampled:
            out = F.interpolate(out, size=(h * 2, w * 2), mode='bilinear', align_corners=False)

        return out
    
class CrossModalSEFusion(nn.Module):
    def __init__(self, embed_dim, reduction=16):
        super().__init__()
        channels = embed_dim
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 2 * channels, 1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)  # (B, 2C, H, W)
        weights = self.fc(x)  # (B, 2C, 1, 1)
        w1, w2 = torch.split(weights, feat1.size(1), dim=1)
        return w1 * feat1 + w2 * feat2




class BilinearFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bilinear = nn.Bilinear(channels, channels, channels)

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.size()
        x1 = feat1.view(b, c, -1).transpose(1, 2)  # (b, hw, c)
        x2 = feat2.view(b, c, -1).transpose(1, 2)
        fused = self.bilinear(x1, x2)  # (b, hw, c)
        return fused.transpose(1, 2).view(b, c, h, w)


class LiteBilinearFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, feat1, feat2):
        fused = feat1 * feat2  # element-wise multiply
        return self.fusion(fused)



class TransformerFusion(nn.Module):
    def __init__(self, d_model, num_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.size()
        x = torch.stack([feat1, feat2], dim=1).mean(-1).mean(-1)  # (b, 2, c)
        fused = self.transformer(x).mean(1)  # (b, c)
        return fused.view(b, c, 1, 1).expand(-1, -1, h, w)

# 动态权重融合模块（Learnable Fusion Weighting）
class DynamicWeightedFusion(nn.Module):
    def __init__(self, channels, reduction=16, spatial=False):
        super().__init__()
        self.spatial = spatial
        if spatial:
            self.conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, 2, kernel_size=1),
                nn.Softmax(dim=1)  # across phase & intensity
            )
        else:
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels * 2, channels // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, 2, kernel_size=1),
                nn.Softmax(dim=1)
            )

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)  # (B, 2C, H, W)
        if self.spatial:
            weights = self.conv(x)  # (B, 2, H, W)
            w1, w2 = weights[:, 0:1], weights[:, 1:2]
        else:
            weights = self.fc(x)  # (B, 2, 1, 1)
            w1, w2 = weights[:, 0:1], weights[:, 1:2]
        return w1 * feat1 + w2 * feat2


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    fusion_modules = {
        'concat': ConcatFusion(),
        'add': AddFusion(),
        'se': SEFusion(channels=64),
        'cross_attention': CrossAttentionFusion(embed_dim=64),
        'bilinear': BilinearFusion(channels=64),
        'transformer': TransformerFusion(d_model=64),
        'dynamic_weight': DynamicWeightedFusion(channels=64),
    }

    input1 = torch.randn(8, 64, 32, 32).cuda()
    input2 = torch.randn(8, 64, 32, 32).cuda()

    for name, fusion in fusion_modules.items():
        fusion = fusion.cuda()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        output = fusion(input1, input2)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"{name:>16} | output shape: {output.shape} | time: {elapsed_time*1000:.2f} ms | peak mem: {max_mem:.2f} MB")
        del fusion, output
        torch.cuda.empty_cache()
        gc.collect()

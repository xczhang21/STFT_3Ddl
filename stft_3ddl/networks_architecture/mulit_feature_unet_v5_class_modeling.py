from collections import OrderedDict
import torch
import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.query_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.key_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, q_feat, kv_feat):
        B, C, H, W = q_feat.shape
        q = self.query_conv(q_feat)  # [B, C, H, W]
        k = self.key_conv(kv_feat)   # [B, C, H, W]
        v = self.value_conv(kv_feat) # [B, C, H, W]

        # 线性注意力近似计算
        k_softmax = torch.softmax(k.view(B, C, -1), dim=-1)  # [B, C, HW]
        context = torch.bmm(v.view(B, C, -1), k_softmax.transpose(1, 2))  # [B, C, C]
        q_flat = q.view(B, C, -1)  # [B, C, HW]
        out = torch.bmm(context, q_flat).view(B, C, H, W)
        return out

class MFUNetV5(nn.Module):
    def __init__(self, config):
        super(MFUNetV5, self).__init__()
        init_features = config.init_features
        in_channels = config.in_channels
        out_channels = config.out_channels

        features = init_features

        self.encoder1 = MFUNetV5._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = MFUNetV5._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = MFUNetV5._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = MFUNetV5._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = MFUNetV5._block(features * 8, features * 16, name="bottleneck")

        self.attn1 = LinearAttention(features)
        self.attn2 = LinearAttention(features * 2)
        self.attn3 = LinearAttention(features * 4)
        self.attn4 = LinearAttention(features * 8)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = MFUNetV5._block(features * 16, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = MFUNetV5._block(features * 8, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = MFUNetV5._block(features * 4, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = MFUNetV5._block(features * 2, features, name="dec1")

        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(features * 100, out_channels)

    def forward(self, x1, x2):
        e1_1 = self.encoder1(x1)
        e2_1 = self.encoder1(x2)
        enc1 = self.attn1(e1_1, e2_1)
        x = self.pool1(enc1)

        e1_2 = self.encoder2(x)
        e2_2 = self.encoder2(x)
        enc2 = self.attn2(e1_2, e2_2)
        x = self.pool2(enc2)

        e1_3 = self.encoder3(x)
        e2_3 = self.encoder3(x)
        enc3 = self.attn3(e1_3, e2_3)
        x = self.pool3(enc3)

        e1_4 = self.encoder4(x)
        e2_4 = self.encoder4(x)
        enc4 = self.attn4(e1_4, e2_4)
        x = self.pool4(enc4)

        x = self.bottleneck(x)

        x = self.upconv4(x)
        x = torch.cat((x, enc4), dim=1)
        x = self.decoder4(x)

        x = self.upconv3(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.decoder3(x)

        x = self.upconv2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.decoder2(x)

        x = self.upconv1(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.decoder1(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

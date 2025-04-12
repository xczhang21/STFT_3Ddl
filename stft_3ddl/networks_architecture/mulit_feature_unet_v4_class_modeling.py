from collections import OrderedDict
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class MFUNetV4(nn.Module):
    def __init__(self, config):
        super(MFUNetV4, self).__init__()
        init_features = config.init_features
        in_channels = config.in_channels
        out_channels = config.out_channels

        features = init_features

        self.encoder1 = MFUNetV4._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = MFUNetV4._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = MFUNetV4._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = MFUNetV4._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = MFUNetV4._block(features * 8, features * 16, name="bottleneck")

        self.fuse1 = nn.Sequential(nn.Conv2d(features * 2, features, kernel_size=1), CBAM(features))
        self.fuse2 = nn.Sequential(nn.Conv2d(features * 4, features * 2, kernel_size=1), CBAM(features * 2))
        self.fuse3 = nn.Sequential(nn.Conv2d(features * 8, features * 4, kernel_size=1), CBAM(features * 4))
        self.fuse4 = nn.Sequential(nn.Conv2d(features * 16, features * 8, kernel_size=1), CBAM(features * 8))

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = MFUNetV4._block(features * 16, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = MFUNetV4._block(features * 8, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = MFUNetV4._block(features * 4, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = MFUNetV4._block(features * 2, features, name="dec1")

        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(features * 100, out_channels)

    def forward(self, x1, x2):
        e1_1 = self.encoder1(x1)
        e2_1 = self.encoder1(x2)
        enc1 = self.fuse1(torch.cat([e1_1, e2_1], dim=1))
        x = self.pool1(enc1)

        e1_2 = self.encoder2(x)
        e2_2 = self.encoder2(x)
        enc2 = self.fuse2(torch.cat([e1_2, e2_2], dim=1))
        x = self.pool2(enc2)

        e1_3 = self.encoder3(x)
        e2_3 = self.encoder3(x)
        enc3 = self.fuse3(torch.cat([e1_3, e2_3], dim=1))
        x = self.pool3(enc3)

        e1_4 = self.encoder4(x)
        e2_4 = self.encoder4(x)
        enc4 = self.fuse4(torch.cat([e1_4, e2_4], dim=1))
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

from collections import OrderedDict
import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FiLM, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels * 2),
            nn.ReLU(),
            nn.Linear(out_channels * 2, out_channels * 2)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        pooled = self.global_pool(x).view(b, c)
        params = self.fc(pooled)
        gamma, beta = params.chunk(2, dim=1)
        return gamma.view(b, -1, 1, 1), beta.view(b, -1, 1, 1)

class MFUNetV2(nn.Module):
    def __init__(self, config):
        super(MFUNetV2, self).__init__()
        init_features = config.init_features
        in_channels = config.in_channels
        out_channels = config.out_channels

        features = init_features
        self.shared_encoder1 = MFUNetV2._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shared_encoder2 = MFUNetV2._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shared_encoder3 = MFUNetV2._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shared_encoder4 = MFUNetV2._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = MFUNetV2._block(features * 8, features * 16, name="bottleneck")

        self.film1 = FiLM(in_channels, features)
        self.film2 = FiLM(in_channels, features * 2)
        self.film3 = FiLM(in_channels, features * 4)
        self.film4 = FiLM(in_channels, features * 8)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = MFUNetV2._block(features * 16, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = MFUNetV2._block(features * 8, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = MFUNetV2._block(features * 4, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = MFUNetV2._block(features * 2, features, name="dec1")

        self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(features * 100, out_channels)

    def forward(self, x_main, x_mod):
        gamma1, beta1 = self.film1(x_mod)
        x = self.shared_encoder1(x_main)
        x = x * (1 + gamma1) + beta1
        enc1 = x
        x = self.pool1(x)

        gamma2, beta2 = self.film2(x_mod)
        x = self.shared_encoder2(x)
        x = x * (1 + gamma2) + beta2
        enc2 = x
        x = self.pool2(x)

        gamma3, beta3 = self.film3(x_mod)
        x = self.shared_encoder3(x)
        x = x * (1 + gamma3) + beta3
        enc3 = x
        x = self.pool3(x)

        gamma4, beta4 = self.film4(x_mod)
        x = self.shared_encoder4(x)
        x = x * (1 + gamma4) + beta4
        enc4 = x
        x = self.pool4(x)

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

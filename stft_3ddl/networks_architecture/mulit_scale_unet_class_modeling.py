from collections import OrderedDict

import torch
import torch.nn as nn

class MSUNet(nn.Module):

    def __init__(self, config):
        super(MSUNet, self).__init__()
        init_features = config.init_features
        in_channels = config.in_channels
        out_channels = config.out_channels        

        features = init_features
        self.encoder3_1 = MSUNet._block(in_channels, features, name="enc3_1")
        self.encoder2_1 = MSUNet._block(in_channels, features, name="enc2_1")
        self.encoder1_1 = MSUNet._block(in_channels, features, name="enc1_1")

        self.bottleneck_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = MSUNet._block(features, features*2, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder4 = MSUNet._block(features*2, features*2, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder3 = MSUNet._block(features*2, features*2, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder2 = MSUNet._block(features*2, features*2, name="dec2")

        self.decoder1 = MSUNet._block(features*2, features, name="dec1")



        # 以下是手动添加的回归头
        self.avgpool = nn.AdaptiveAvgPool2d((10,10))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3200, out_channels)


    def forward(self, x1, x2, x3):
        enc3_1 = self.encoder3_1(x3)

        enc2_1 = self.encoder2_1(x2)

        enc1_1 = self.encoder1_1(x1)
        bottleneck = self.bottleneck(self.bottleneck_pool(enc3_1))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc3_1), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc2_1), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1_1), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.decoder1(dec2)

        output = self.avgpool(dec1)
        output = self.flatten(output)
        output = self.fc(output)
        output = output.squeeze(-1)
        return output
        

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2", 
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )




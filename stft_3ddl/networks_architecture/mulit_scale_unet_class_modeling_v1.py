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

        self.pool3_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_2 = MSUNet._block(features, features * 2, name="enc3_2")
        self.encoder2_2 = MSUNet._block(features, features * 2, name="enc2_2")
        self.encoder1_2 = MSUNet._block(features, features * 2, name="enc1_2")

        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_3 = MSUNet._block(features * 2, features * 4, name="enc2_3")
        self.encoder1_3 = MSUNet._block(features * 2, features * 4, name="enc1_3")

        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1_4 = MSUNet._block(features * 4, features * 8, name="enc1_4")

        self.bottleneck_fusion1 = MSUNet._block(features*(2+4), features*4, name="bot_fus1")
        self.bottleneck_fusion2 = MSUNet._block(features*(4+8), features*8, name="bot_fus2")

        self.bottleneck_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = MSUNet._block(features * 8, features * 16, name="bottleneck")
        
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.fusion4_1 = MSUNet._block(features*(2+4), features*4, name="fus4_1")
        self.fusion4_2 = MSUNet._block(features*(4+8), features*8, name="fus4_2")
        self.decoder4 = MSUNet._block(features * 16, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.fusion3_1 = MSUNet._block(features*(1+2), features*2, name="fus3_1")
        self.fusion3_2 = MSUNet._block(features*(2+4), features*4, name="fus3_2")
        self.decoder3 = MSUNet._block(features * 8, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.fusion2_1 = MSUNet._block(features*(1+2), features*2, name="fus2_1")
        self.decoder2 = MSUNet._block(features * 4, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features * 1, kernel_size=2, stride=2
        )
        self.decoder1 = MSUNet._block(features * 2, features * 1, name="dec1")

        # 以下是手动添加的回归头
        self.avgpool = nn.AdaptiveAvgPool2d((10,10))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3200, out_channels)


    def forward(self, x1, x2, x3):
        enc3_1 = self.encoder3_1(x3)
        enc3_2 = self.encoder3_2(self.pool3_1(enc3_1))

        enc2_1 = self.encoder2_1(x2)
        enc2_2 = self.encoder2_2(self.pool2_1(enc2_1))
        enc2_3 = self.encoder2_3(self.pool2_2(enc2_2))

        enc1_1 = self.encoder1_1(x1)
        enc1_2 = self.encoder1_2(self.pool1_1(enc1_1))
        enc1_3 = self.encoder1_3(self.pool1_2(enc1_2))
        enc1_4 = self.encoder1_4(self.pool1_3(enc1_3))

        bottleneck_fus = self.bottleneck_fusion1(torch.cat((enc3_2, enc2_3), dim=1))
        bottleneck_fus = self.bottleneck_fusion2(torch.cat((bottleneck_fus, enc1_4), dim=1))

        bottleneck = self.bottleneck(self.bottleneck_pool(bottleneck_fus))
        
        dec4 = self.upconv4(bottleneck)
        fus4 = self.fusion4_1(torch.cat((enc3_2, enc2_3), dim=1))
        fus4 = self.fusion4_2(torch.cat((fus4, enc1_4), dim=1))
        dec4 = torch.cat((dec4, fus4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        fus3 = self.fusion3_1(torch.cat((enc3_1, enc2_2), dim=1))
        fus3 = self.fusion3_2(torch.cat((fus3, enc1_3,), dim=1))
        dec3 = torch.cat((dec3, fus3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        fus2 = self.fusion2_1(torch.cat((enc2_1, enc1_2), dim=1))
        dec2 = torch.cat((dec2, fus2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1_1), dim=1)
        dec1 = self.decoder1(dec1)

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




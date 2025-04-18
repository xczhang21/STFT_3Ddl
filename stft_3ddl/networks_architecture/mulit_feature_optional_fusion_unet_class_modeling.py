from collections import OrderedDict
from .mulit_feature_fusion_block import *

import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, config, fusion_type, channels):
        super().__init__()
        self.fusion_type = fusion_type
        self.config = config

        if fusion_type == 'concat':
            self.concat = ConcatFusion(channels=channels)
        elif fusion_type == 'add':
            self.add = AddFusion()
        elif fusion_type == 'se':
            self.se = SEFusion(channels=channels)
        elif fusion_type == 'cross_attention':
            self.cross_attention = CrossModalSEFusion(embed_dim=channels)
        elif fusion_type == 'bilinear':
            self.bilinear = LiteBilinearFusion(channels=channels)
        elif fusion_type == 'transformer':
            self.transformer = TransformerFusion(d_model=channels)
        elif fusion_type == 'dynamic_weight':
            self.dynamic_weight = DynamicWeightedFusion(channels=channels)
        elif fusion_type == 'weight_add':
            self.weight_add = WeightedAddFusion(alpha=config.alpha)
        elif fusion_type == 'learnable_add':
            self.learnable_add = LearnableAddFusion(init_alpha=config.init_alpha)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

    def forward(self, feat1, feat2):
        if self.fusion_type == 'concat':
            return self.concat(feat1, feat2)
        elif self.fusion_type == 'add':
            return self.add(feat1, feat2)
        elif self.fusion_type == 'se':
            return self.se(feat1, feat2)
        elif self.fusion_type == 'cross_attention':
            return self.cross_attention(feat1, feat2)
        elif self.fusion_type == 'bilinear':
            return self.bilinear(feat1, feat2)
        elif self.fusion_type == 'transformer':
            return self.transformer(feat1, feat2)
        elif self.fusion_type == 'dynamic_weight':
            return self.dynamic_weight(feat1, feat2)
        elif self.fusion_type == 'weight_add':
            return self.weight_add(feat1, feat2)
        elif self.fusion_type == 'learnable_add':
            return self.learnable_add(feat1, feat2)
        else:
            raise ValueError(f"Unsupported fusion type during forward: {self.fusion_type}")


class MFOFUNet(nn.Module):

    def __init__(self, config):
        super(MFOFUNet, self).__init__()
        init_features = config.init_features
        in_channels = config.in_channels
        out_channels = config.out_channels
        fusion_type = config.fusion_type

        features = init_features
        self.encoder1_1 = MFOFUNet._block(in_channels, features, name="enc1_1")
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_1 = MFOFUNet._block(in_channels, features, name="enc2_1")
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fuse1 = FusionModule(config, fusion_type=fusion_type, channels=features)
        
        self.encoder1_2 = MFOFUNet._block(features, features * 2, name="enc1_2")
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_2 = MFOFUNet._block(features, features * 2, name="enc2_2")
        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fuse2 = FusionModule(config, fusion_type=fusion_type, channels=features * 2)

        self.encoder1_3 = MFOFUNet._block(features * 2, features * 4, name="enc1_3")
        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_3 = MFOFUNet._block(features * 2, features * 4, name="enc2_3")
        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fuse3 = FusionModule(config, fusion_type=fusion_type, channels=features * 4)

        self.encoder1_4 = MFOFUNet._block(features * 4, features * 8, name="enc1_4")
        self.pool1_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_4 = MFOFUNet._block(features * 4, features * 8, name="enc2_4")
        self.pool2_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fuse4 = FusionModule(config, fusion_type=fusion_type, channels=features * 8)

        self.fuse_bottleneck = FusionModule(config, fusion_type=fusion_type, channels=features * 16)
        self.pool_bottleneck = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = MFOFUNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = MFOFUNet._block(features * 16, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = MFOFUNet._block(features * 8, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = MFOFUNet._block(features * 4, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features *1, kernel_size=2, stride=2
        )
        self.decoder1 = MFOFUNet._block(features * 2, features * 1, name="dec1")

        # 以下是手动添加的回归头
        self.avgpool = nn.AdaptiveAvgPool2d((10,10))
        self.flatten = nn.Flatten()
        # self.dropout_fc = nn.Dropout(p=0.5)
        # self.fc_1 = nn.Linear(in_features=out_channels, out_features=2048, )
        self.fc = nn.Linear(features *100, out_channels)

    def forward(self, x1, x2):
        enc1_1 = self.encoder1_1(x1)
        enc2_1 = self.encoder2_1(x2)
        f1 = self.fuse1(enc1_1, enc2_1)

        enc1_2 = self.encoder1_2(self.pool1_1(enc1_1))
        enc2_2 = self.encoder2_2(self.pool2_1(enc2_1))
        f2 = self.fuse2(enc1_2, enc2_2)

        enc1_3 = self.encoder1_3(self.pool1_2(enc1_2))
        enc2_3 = self.encoder2_3(self.pool2_2(enc2_2))
        f3 = self.fuse3(enc1_3, enc2_3)

        enc1_4 = self.encoder1_4(self.pool1_3(enc1_3))
        enc2_4 = self.encoder2_4(self.pool2_3(enc2_3))
        f4 = self.fuse4(enc1_4, enc2_4)

        bottleneck = self.bottleneck(self.pool_bottleneck(f4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, f4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, f3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, f2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, f1), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.avgpool(dec1)
        output = self.flatten(output)
        # output = self.dropout_fc(output)
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
                    # (name + "drop1", nn.Dropout(p=0.3)),
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
                    # (name + "drop2", nn.Dropout(p=0.3)),
                ]
            )
        )
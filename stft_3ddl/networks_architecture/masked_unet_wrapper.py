import torch
import torch.nn as nn
from .unet_class_modeling import UNet

class MaskedUNetWrapper(nn.Module):
    def __init__(self, config):
        super(MaskedUNetWrapper, self).__init__()
        self.encoder = UNet(config)
        self.reconstruct_head = nn.Conv2d(config.init_features, config.in_channels, kernel_size=1)

    def forward(self, x, mask):
        # 前向传播，复用 UNet 主干
        enc1 = self.encoder.encoder1(x)
        enc2 = self.encoder.encoder2(self.encoder.pool1(enc1))
        enc3 = self.encoder.encoder3(self.encoder.pool2(enc2))
        enc4 = self.encoder.encoder4(self.encoder.pool3(enc3))
        bottleneck = self.encoder.bottleneck(self.encoder.pool4(enc4))

        dec4 = self.encoder.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.encoder.decoder4(dec4)
        dec3 = self.encoder.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.encoder.decoder3(dec3)
        dec2 = self.encoder.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.encoder.decoder2(dec2)
        dec1 = self.encoder.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.encoder.decoder1(dec1)

        pred = self.reconstruct_head(dec1)
        return pred, mask

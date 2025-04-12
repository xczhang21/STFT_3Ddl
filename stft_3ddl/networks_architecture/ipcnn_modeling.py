"""
Wu H, Zhou B, Zhu K, et al. Pattern recognition in distributed fiber-optic acoustic sensor using an intensity and phase stacked convolutional neural network with data augmentation[J]. Optics express, 2021, 29(3): 3269-3283.
"""
"""
已复现IPCNN，但对模型不做修改则模型不收敛，
对模型做修改(加上BN)后，模型收敛，但还是达不到作者论文中的88.2%
"""
from collections import OrderedDict
import torch
import torch.nn as nn

class BaseCNN(nn.Module):
    def __init__(self, in_channels, features):
        super(BaseCNN, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1)),
            # ("bn1", nn.BatchNorm2d(features)),
            ("relu1", nn.ReLU(inplace=False)),
            ("pool1", nn.MaxPool2d(kernel_size=2)),
            ("drop1", nn.Dropout(p=0.5)),

            ("conv2", nn.Conv2d(features, features * 2, kernel_size=3, padding=1)),
            # ("bn2", nn.BatchNorm2d(features*2)),
            ("relu2", nn.ReLU(inplace=False)),
            ("pool2", nn.MaxPool2d(kernel_size=2)),
            ("drop2", nn.Dropout(p=0.5)),

            ("conv3", nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1)),
            # ("bn3", nn.BatchNorm2d(features*4)),
            ("relu3", nn.ReLU(inplace=False)),
            ("pool3", nn.MaxPool2d(kernel_size=2)),
            ("drop3", nn.Dropout(p=0.5)),

            ("conv4", nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1)),
            # ("bn4", nn.BatchNorm2d(features*4)),
            ("relu4", nn.ReLU(inplace=False)),
            ("pool4", nn.MaxPool2d(kernel_size=2)),
            ("drop4", nn.Dropout(p=0.5)),

            ("gap", nn.AdaptiveAvgPool2d((10, 10)))
        ]))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.net(x)
        x = self.flatten(x)
        # x = x.view(x.size(0), -1)
        return x


class IPCNN(nn.Module):
    def __init__(self, config):
        super(IPCNN, self).__init__()
        in_channels = config.in_channels
        features = config.init_features
        num_classes = config.out_channels

        self.i_branch = BaseCNN(in_channels, features)
        self.p_branch = BaseCNN(in_channels, features)

        self.classifier = nn.Linear(features * 800, num_classes)

    def forward(self, i_x, p_x):
        i_feat = self.i_branch(i_x)
        p_feat = self.p_branch(p_x)
        combined = torch.cat([i_feat, p_feat], dim=1).clone()
        logits = self.classifier(combined)
        return logits

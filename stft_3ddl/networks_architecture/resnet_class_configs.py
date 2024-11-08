import ml_collections

def get_ResNet50_config():
    """

    """
    config = ml_collections.ConfigDict()
    config.block = 'Bottleneck'
    config.in_channels = 3
    config.encoder_channels = [3, 4, 23, 3]
    config.num_classes = 10
    config.zero_init_residual = None
    config.groups = None
    config.width_per_group = None
    config.replace_stride_with_dilation = None
    config.norm_layer = None
    return config


# 模型测试
from resnet_class_modeling import ResNet
import torch
import torch.nn as nn

config = get_ResNet50_config()
model = ResNet(config)
input_data = torch.randn(32, 3, 128, 128)

output = model(input_data)
print(output.shape)

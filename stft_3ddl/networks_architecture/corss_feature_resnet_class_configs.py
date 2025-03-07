# 结构有问题，未完成
import ml_collections

def get_CFResNet_config():
    """
    
    """
    config = ml_collections.ConfigDict()
    config.block = 'Bottleneck'
    config.in_channels = 3
    config.encoder_num = [3, 4, 23, 3]
    config.num_classes = 10
    config.zero_init_residual = None
    config.groups = None
    config.width_per_group = None
    config.replace_stride_with_dilation = None
    config.norm_layer = None
    return config

# 模型测试
if __name__ == '__main__':
    from corss_feature_resnet_class_modeling import CFResNet
    import torch
    import torch.nn as nn

    config = get_CFResNet_config()
    config.in_channels = 1
    model = CFResNet(config)
    input_data1 = torch.randn(32, 1, 128, 128)
    input_data2 = torch.randn(32, 1, 128, 128)

    output = model(input_data1, input_data2)
    print(output.shape)
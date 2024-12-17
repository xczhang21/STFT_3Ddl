import ml_collections

def get_UNet_config():
    """
    """
    config = ml_collections.ConfigDict()
    config.in_channels = 1
    config.out_channels = 10
    config.init_features = 32
    # 下面的encoder_channels是为了配合整体框架写了一个无用的
    config.encoder_channels = [3]
    return config

# 模型测试
if __name__ == '__main__':
    from unet_class_modeling import UNet
    import torch
    import torch.nn as nn

    config = get_UNet_config()
    config.in_channels = 1
    model = UNet(config)
    input_data = torch.randn(32, 1, 128, 128)

    output = model(input_data)
    print(output.shape)
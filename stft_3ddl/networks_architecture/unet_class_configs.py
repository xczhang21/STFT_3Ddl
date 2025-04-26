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

    # 测试MaskedUNetWrapper
    print("\n Testing MaskedUNetWrapper...")

    from masked_unet_wrapper import MaskedUNetWrapper

    # 用一样的config, 但要把out_channels 改成输入通道（重建）
    config.out_channels = config.in_channels
    model = MaskedUNetWrapper(config)

    x = torch.randn(32, 1, 128, 128)
    mask = torch.zeros_like(x)
    mask[:, :, 32:96, 32:96] = 1 # 中间区域做mask

    pred, mask_out = model(x, mask)
    print("MaskedUNetWrapper pred shape:", pred.shape)
    print("Mask shape:", mask_out.shape)
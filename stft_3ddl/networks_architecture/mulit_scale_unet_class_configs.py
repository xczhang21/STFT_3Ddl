import ml_collections

def get_MSUNet_config():
    """
    """
    config = ml_collections.ConfigDict()
    config.in_channels = 1
    config.out_channels = 10
    config.init_features = 32
    # 下面的encoder_channels是为了配合整个框架写的一个无用的
    config.encoder_channels = [3]
    return config
    

# 模型测试
if __name__ == '__main__':
    from mulit_scale_unet_class_modeling import MSUNet
    import torch
    import torch.nn as nn

    config = get_MSUNet_config()
    config.in_channels = 1
    model = MSUNet(config)
    model.cuda()
    input1 = torch.randn(32, 1, 256, 256).cuda()
    input2 = torch.randn(32, 1, 128, 128).cuda()
    input3 = torch.randn(32, 1, 64, 64).cuda()

    output = model(input1, input2, input3)
    print(output.shape)

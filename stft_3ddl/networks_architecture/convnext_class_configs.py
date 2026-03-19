import ml_collections


def get_ConvNeXt_config():
    config = ml_collections.ConfigDict()
    config.in_channels = 2
    # config.cnblock = [config.in_channels, 256, 3], [256, 512, 3], [512, 1024, 27], [1024, None, 3]
    config.cnblock = [128, 256, 3], [256, 512, 3], [512, 1024, 27], [1024, None, 3]
    # config.cnblock = [config.in_channels, 192, 3], [192, 384, 3], [384, 768, 9], [768, None, 3]


    config.stochastic_depth_prob = 0.2
    config.layer_scale = 1e-6
    config.num_classes = 10
    config.block = None
    config.norm_layer = None

    return config


# 模型测试
if __name__ == '__main__':
    from convnext_class_modeling import ConvNeXt
    import torch


    cfg = get_ConvNeXt_config()
    cfg.in_channels = 2
    model = ConvNeXt(cfg)
    x = torch.randn(32, 2, 128, 128)
    y = model(x)
    print(y.shape)
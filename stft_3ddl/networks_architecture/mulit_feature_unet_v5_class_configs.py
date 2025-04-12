import ml_collections

def get_MFUNetV5_config():
    config = ml_collections.ConfigDict()
    config.in_channels = 1  # 每个输入图像通道数
    config.out_channels = 10  # 分类类别数量
    config.init_features = 64  # 初始特征通道数
    config.encoder_channels = [3]  # 占位字段
    return config

# 模型测试
if __name__ == '__main__':
    from mulit_feature_unet_v5_class_modeling import MFUNetV5
    import torch

    config = get_MFUNetV5_config()
    model = MFUNetV5(config).cuda()

    x1 = torch.randn(32, 1, 256, 256).cuda()
    x2 = torch.randn(32, 1, 256, 256).cuda()

    output = model(x1, x2)
    print(output.shape)  # 应输出 [32, 10]

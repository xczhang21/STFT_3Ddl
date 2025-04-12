import ml_collections

def get_MFUNetV2_config():
    config = ml_collections.ConfigDict()
    config.in_channels = 1  # 单通道输入，可用于 intensity 或 phase
    config.out_channels = 10  # 分类类别数量
    config.init_features = 64  # 初始特征维度
    config.encoder_channels = [3]  # 保留字段，占位无用
    return config

# 模型测试
if __name__ == '__main__':
    from mulit_feature_unet_v2_class_modeling import MFUNetV2
    import torch

    config = get_MFUNetV2_config()
    model = MFUNetV2(config).cuda()

    # 主输入 intensity，调制输入 phase
    x_main = torch.randn(32, 1, 256, 256).cuda()
    x_mod = torch.randn(32, 1, 256, 256).cuda()

    output = model(x_main, x_mod)
    print(output.shape)  # 应该是 [32, 10]

import ml_collections

def get_Swin_timm_config():
    """

    字段说明：
        model_name:        timm 的 Swin 模型名
                           例如 'swin_tiny_patch4_window7_224'
        in_channels:       输入通道数（你的数据：1 / 2 / 3）
        num_classes:       最终分类类别数
        img_size:          输入图像尺寸，例如 224 或 256

        pretrained:        是否加载 ImageNet 预训练权重

        drop_rate:         dropout，默认 0.0
        drop_path_rate:    drop-path (Stochastic Depth)，默认 0.1

        repeat_to_3ch:     当 pretrained=True 且官方预训练权重要求 3 通道时，
                           可将 1/2 通道复制成 3 通道（默认 False）

        enable_feat_hook:  是否注册 Swin 最后层的 forward hook，用于 Grad-CAM

    """

    config = ml_collections.ConfigDict()

    # ---- 基本配置 ----
    config.model_name = 'swin_tiny_patch4_window7_224'
    config.in_channels = 1
    config.num_classes = 10
    config.img_size = 224

    # ---- 训练相关 ----
    config.pretrained = False
    config.drop_rate = 0.0
    config.drop_path_rate = 0.1

    # ---- 特殊功能开关 ----
    # 若 image single-channel（1 通道），但想用 3 通道预训练权重，可设置 True
    config.repeat_to_3ch = False

    # 是否注册 hook 抓中间特征（Grad-CAM / 显著性可视化）
    config.enable_feat_hook = True

    return config


# ------------------------------
# 简单自测（可选）
# ------------------------------
if __name__ == "__main__":
    from swin_timm_class_modeling import Swin_timm
    import torch

    cfg = get_Swin_timm_config()
    cfg.in_channels = 2
    cfg.num_classes = 10
    cfg.img_size = 128
    cfg.model_name = 'swin_tiny_patch4_window7_224'
    """
    timm Swin支持的模型名:
        swin_tiny_patch4_window7_224
        swin_small_patch4_window7_224
        swin_base_patch4_window7_224
        swin_base_patch4_window12_384
        swin_large_patch4_window7_224
        swinv2_tiny_patch4_window8_256
        swinv2_small_patch4_window16_256
        swinv2_base_patch4_window12_256
        swinv2_large_patch4_window12_192
    """
    cfg.pretrained = False

    net = Swin_timm(cfg)
    x = torch.randn(32, 2, 128, 128)
    y = net(x)
    print("logits:", y.shape)  # (4, 5)

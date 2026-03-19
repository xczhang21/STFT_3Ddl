import ml_collections

def get_SwinTransformer_config():

    config = ml_collections.ConfigDict()
    config.img_size = 224
    config.patch_size = 4
    config.in_channels = 3
    config.num_classes = 1000
    config.embed_dim = 96
    config.depths = [2, 2, 6, 2]
    config.num_heads = [3, 6, 12, 24]
    config.window_size = 7
    config.mlp_ratio = 4
    config.qkv_bias = True
    config.qk_scale = None
    config.drop_rate = 0.
    config.attn_drop_rate = 0.
    config.drop_path_rate = 0.1
    config.norm_layer = 'nn.LayerNorm'
    config.ape = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.fused_window_process = False

    # 下面的encoder_channels是为了配合整体框架写了一个无用的
    config.encoder_channels = [3]
    return config

if __name__ == "__main__":
    from swintransformer_class_modeling import SwinTransformer
    import torch

    config = get_SwinTransformer_config()
    config.in_channels = 2
    config.num_classes = 10
    config.img_size = 256
    config.window_size = 8

    net = SwinTransformer(config)
    x = torch.randn(32, 2, 256, 256)
    y = net(x)
    print(y.shape)
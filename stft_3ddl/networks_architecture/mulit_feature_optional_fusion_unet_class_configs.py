import ml_collections
import torch
import time
import gc
# from .mulit_feature_optional_fusion_unet_class_modeling import MFOFUNet

def _get_base_config():
    config = ml_collections.ConfigDict()
    config.in_channels = 1
    config.out_channels = 10
    config.init_features = 64
    config.encoder_channels = [3]
    return config

def get_MFOFUNet_learnable_add_config():
    config = _get_base_config()
    config.fusion_type = 'learnable_add'
    config.init_alpha = 0.5
    return config

def get_MFOFUNet_weight_add_config():
    config = _get_base_config()
    config.fusion_type = 'weight_add'
    config.alpha = 0.5
    return config

def get_MFOFUNet_concat_config():
    config = _get_base_config()
    config.fusion_type = 'concat'
    return config

def get_MFOFUNet_add_config():
    config = _get_base_config()
    config.fusion_type = 'add'
    return config

def get_MFOFUNet_se_config():
    config = _get_base_config()
    config.fusion_type = 'se'
    return config

def get_MFOFUNet_cross_attention_config():
    config = _get_base_config()
    config.fusion_type = 'cross_attention'
    return config

def get_MFOFUNet_bilinear_config():
    config = _get_base_config()
    config.fusion_type = 'bilinear'
    return config

def get_MFOFUNet_transformer_config():
    config = _get_base_config()
    config.fusion_type = 'transformer'
    return config

def get_MFOFUNet_dynamic_weight_config():
    config = _get_base_config()
    config.fusion_type = 'dynamic_weight'
    return config

def test_model(config):
    model = MFOFUNet(config).cuda()
    input1 = torch.randn(32, 1, 256, 256).cuda()
    input2 = torch.randn(32, 1, 256, 256).cuda()

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = model(input1, input2)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    max_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"{config.fusion_type:>16} | output shape: {output.shape} | time: {elapsed_time*1000:.2f} ms | peak mem: {max_mem:.2f} MB")

    del model, input1, input2, output
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    from mulit_feature_optional_fusion_unet_class_modeling import MFOFUNet
    configs = [
        get_MFOFUNet_concat_config(),
        get_MFOFUNet_add_config(),
        get_MFOFUNet_se_config(),
        get_MFOFUNet_cross_attention_config(),
        get_MFOFUNet_bilinear_config(),
        get_MFOFUNet_transformer_config(),
        get_MFOFUNet_dynamic_weight_config(),
        get_MFOFUNet_weight_add_config(),
        get_MFOFUNet_learnable_add_config(),
    ]

    for cfg in configs:
        test_model(cfg)
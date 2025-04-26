from .resnet_class_configs import *
from .unet_class_configs import *
from .mulit_scale_unet_class_configs import *
from .mulit_feature_unet_class_configs import *
from .ipcnn_configs import *
from .mulit_feature_unet_v2_class_configs import *
from .mulit_feature_unet_v3_class_configs import *
from .mulit_feature_unet_v4_class_configs import *
from .mulit_feature_unet_v5_class_configs import *
from .mulit_feature_optional_fusion_unet_class_configs import *

CONFIGS = {
    'ResNet': get_ResNet_config(),
    'UNet': get_UNet_config(),
    'MSUNet': get_MSUNet_config(),
    'MFUNet': get_MFUNet_config(),
    'IPCNN': get_IPCNN_config(),
    'MFUNetV2': get_MFUNetV2_config(),
    'MFUNetV3': get_MFUNetV3_config(),
    'MFUNetV4': get_MFUNetV4_config(),
    'MFUNetV5': get_MFUNetV5_config(),
    "MFOFUNet":{
        "concat": get_MFOFUNet_concat_config(),
        "add": get_MFOFUNet_add_config(),
        "se": get_MFOFUNet_se_config(),
        "cross_attention": get_MFOFUNet_cross_attention_config(),
        "bilinear": get_MFOFUNet_bilinear_config(),
        "transformer": get_MFOFUNet_transformer_config(),
        "dynamic_weight": get_MFOFUNet_dynamic_weight_config(),
        "weight_add": get_MFOFUNet_weight_add_config(),
        "learnable_add": get_MFOFUNet_learnable_add_config(),
        "learnable_spatial_wise_add": get_MFOFUNet_learnable_spatial_wise_add_config(),
        "learnable_channel_wise_add": get_MFOFUNet_learnable_channel_wise_add_config(),
    },
    "MaskedUNetWrapper": get_UNet_config()

}
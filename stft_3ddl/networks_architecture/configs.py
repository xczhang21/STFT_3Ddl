from .resnet_class_configs import *
from .unet_class_configs import *
from .mulit_scale_unet_class_configs import *
from .mulit_feature_unet_class_configs import *
from .ipcnn_configs import *
from .mulit_feature_unet_v2_class_configs import *
from .mulit_feature_unet_v3_class_configs import *
from .mulit_feature_unet_v4_class_configs import *
from .mulit_feature_unet_v5_class_configs import *

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
}
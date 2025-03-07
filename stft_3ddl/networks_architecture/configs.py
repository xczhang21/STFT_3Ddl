from .resnet_class_configs import *
from .unet_class_configs import *
from .mulit_scale_unet_class_configs import *
from .mulit_feature_unet_class_configs import *

CONFIGS = {
    'ResNet': get_ResNet_config(),
    'UNet': get_UNet_config(),
    'MSUNet': get_MSUNet_config(),
    'MFUNet': get_MFUNet_config(),
}
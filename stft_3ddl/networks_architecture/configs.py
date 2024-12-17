from .resnet_class_configs import *
from .unet_class_configs import *

CONFIGS = {
    'ResNet': get_ResNet_config(),
    'UNet': get_UNet_config(),
}
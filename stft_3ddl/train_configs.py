import ml_collections
from datasets import datasets_config
from networks_architecture.configs import CONFIGS as CONFIGS

def _base_config():
    config = ml_collections.ConfigDict()
    config.base_lr = 0.01
    config.batch_size = 32
    config.n_gpu = 1
    config.seed = 1234
    config.max_epochs = 150
    config.is_pretrain = False
    config.task_type = "cla"
    config.net_name = "ResNet"
    return config


def _get_maskedunetwrapper_config():
    config = _base_config()
    config.net_name = "MaskedUNetWrapper"
    config.dataset_name = "das1k"

    config.max_epochs = 100
    config.task_type = "mae" # Masked Autoencoder
    return config
    

def get_phase_maskedunetwrapper_ss64_train(to_dataset_config):
    config = _get_maskedunetwrapper_config()

    config.prepro_method = "phase"
    config.dataset_spectrum_size = 64

    get_dataset_config_func = f"get_masked_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)(to_dataset_config)

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_phase_maskedunetwrapper_ss128_train(to_dataset_config):
    config = _get_maskedunetwrapper_config()

    config.prepro_method = "phase"
    config.dataset_spectrum_size = 128

    get_dataset_config_func = f"get_masked_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)(to_dataset_config)

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_phase_maskedunetwrapper_ss256_train(to_dataset_config):
    config = _get_maskedunetwrapper_config()

    config.prepro_method = "phase"
    config.dataset_spectrum_size = 256

    get_dataset_config_func = f"get_masked_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)(to_dataset_config)

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_maskedunetwrapper_ss64_train(to_dataset_config):
    config = _get_maskedunetwrapper_config()

    config.prepro_method = "intensity"
    config.dataset_spectrum_size = 64

    get_dataset_config_func = f"get_masked_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)(to_dataset_config)

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_intensity_maskedunetwrapper_ss128_train(to_dataset_config):
    config = _get_maskedunetwrapper_config()

    config.prepro_method = "intensity"
    config.dataset_spectrum_size = 128

    get_dataset_config_func = f"get_masked_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)(to_dataset_config)

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_maskedunetwrapper_ss256_train(to_dataset_config):
    config = _get_maskedunetwrapper_config()

    config.prepro_method = "intensity"
    config.dataset_spectrum_size = 256

    get_dataset_config_func = f"get_masked_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)(to_dataset_config)

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_pi_mfofunet_learnablespatialwiseadd_ss256_train(to_config):
    config = _base_config()

    config.net_name = "MFOFUNet"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 256
    config.fusion_method = "learnable_spatial_wise_add"
    
    # 假设 to_config 是类似 "learnable_add_res"或"learnable_add_nores"
    assert to_config.startswith(config.fusion_method), f"to_config '{to_config}' 必须以 '{config.fusion_method}' 开头"

    res = to_config[len(config.fusion_method)+1:]
    assert res in ["res", "nores"], f"res '{res}' 不是'res'或'nores'"

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    nets = CONFIGS[config.net_name]
    assert config.fusion_method in nets.keys(), f"Fusion method {config.fusion_method} is not drived from CONFIGS"
    config.net = nets[config.fusion_method]
    if res == "res":
        config.net.residual_fusion = True
    elif res == "nores":
        config.net.residual_fusion = False
    
    return config


def get_pi_mfofunet_learnablechannelwiseadd_ss256_train(to_config):
    config = _base_config()

    config.net_name = "MFOFUNet"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 256
    config.fusion_method = "learnable_channel_wise_add"
    
    # 假设 to_config 是类似 "learnable_add_res"或"learnable_add_nores"
    assert to_config.startswith(config.fusion_method), f"to_config '{to_config}' 必须以 '{config.fusion_method}' 开头"

    res = to_config[len(config.fusion_method)+1:]
    assert res in ["res", "nores"], f"res '{res}' 不是'res'或'nores'"

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    nets = CONFIGS[config.net_name]
    assert config.fusion_method in nets.keys(), f"Fusion method {config.fusion_method} is not drived from CONFIGS"
    config.net = nets[config.fusion_method]
    if res == "res":
        config.net.residual_fusion = True
    elif res == "nores":
        config.net.residual_fusion = False
    
    return config


def get_pi_mfofunet_learnableadd_ss256_train(to_config):
    config = _base_config()

    config.net_name = "MFOFUNet"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 256
    config.fusion_method = "learnable_add"
    
    # 假设 to_config 是类似 "learnable_add_res"或"learnable_add_nores"
    assert to_config.startswith(config.fusion_method), f"to_config '{to_config}' 必须以 '{config.fusion_method}' 开头"

    res = to_config[len(config.fusion_method)+1:]
    assert res in ["res", "nores"], f"res '{res}' 不是'res'或'nores'"

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    nets = CONFIGS[config.net_name]
    assert config.fusion_method in nets.keys(), f"Fusion method {config.fusion_method} is not drived from CONFIGS"
    config.net = nets[config.fusion_method]
    if res == "res":
        config.net.residual_fusion = True
    elif res == "nores":
        config.net.residual_fusion = False
    
    return config

    

def get_pi_mfofunet_weightadd_ss256_train(to_config):
    config = _base_config()

    config.net_name = "MFOFUNet"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 256
    config.fusion_method = "weight_add"

    # 假设 to_config 是类似 "weight_add0.7" 的字符串
    assert to_config.startswith(config.fusion_method), f"to_config '{to_config}' 必须以 '{config.fusion_method}' 开头"

    try:
        alpha = float(to_config[len(config.fusion_method):])
    except ValueError:
        raise ValueError(f"无法从 to_config '{to_config}' 中解析出 alpha 值")

    assert 0 <= alpha <= 1, f"alpha 值必须在 [0, 1] 范围内，当前值为 {alpha}"

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    nets = CONFIGS[config.net_name]
    assert config.fusion_method in nets.keys(), f"Fusion method {config.fusion_method} is not drived from CONFIGS"
    config.net = nets[config.fusion_method]
    config.net.alpha = alpha

    return config



def get_pi_mfofunet_ss256_train(to_config):
    config = _base_config()

    config.net_name = "MFOFUNet"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 256
    config.fusion_method = to_config

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    nets = CONFIGS[config.net_name]
    assert config.fusion_method in nets.keys(), f"Fusion method {config.fusion_method} is not drived from CONFIGS"
    config.net = nets[config.fusion_method]

    return config

def get_pi_mfunetv2_ss256_train():
    config = _base_config()

    config.net_name = "MFUNetV2"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_pi_mfunetv3_ss256_train():
    config = _base_config()

    config.net_name = "MFUNetV3"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_pi_mfunetv4_ss256_train():
    config = _base_config()

    config.net_name = "MFUNetV4"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_pi_mfunetv5_ss256_train():
    config = _base_config()

    config.net_name = "MFUNetV5"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_intensity_unet_resize_ss64_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_resize"
    config.prepro_method = "intensity"
    config.dataset_spectrum_size = 64

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_unet_resize_ss128_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_resize"
    config.prepro_method = "intensity"
    config.dataset_spectrum_size = 128

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_unet_resize_ss256_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_resize"
    config.prepro_method = "intensity"
    config.dataset_spectrum_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_phase_unet_resize_ss64_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_resize"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 64

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_phase_unet_resize_ss128_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_resize"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 128

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_phase_unet_resize_ss256_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_resize"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config



def get_intensity_unet_padding0_ss64_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_padding0"
    config.prepro_method = "intensity"
    config.dataset_spectrum_size = 64

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_unet_padding0_ss128_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_padding0"
    config.prepro_method = "intensity"
    config.dataset_spectrum_size = 128

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_unet_padding0_ss256_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_padding0"
    config.prepro_method = "intensity"
    config.dataset_spectrum_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_phase_unet_padding0_ss64_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_padding0"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 64

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_phase_unet_padding0_ss128_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_padding0"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 128

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_phase_unet_padding0_ss256_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k_padding0"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_pi_mfunet_ss64_train():
    config = _base_config()

    config.net_name = "MFUNet"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 64

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config
    
def get_pi_mfunet_ss128_train():
    config = _base_config()

    config.net_name = "MFUNet"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 128

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config
    
def get_pi_mfunet_ss256_train():
    config = _base_config()

    config.net_name = "MFUNet"
    config.dataset_name = "mfdas1k"
    config.prepro_method = "pi"
    config.dataset_spectrum_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"

    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config
    

def get_phase_msunet_ssms_train():
    config = _base_config()

    config.net_name = "MSUNet"
    config.dataset_name = "msdas1k"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = "ms"

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_msunet_ssms_train():
    config = _base_config()

    config.net_name = "MSUNet"
    config.dataset_name = "msdas1k"
    config.prepro_method = "intensity"
    config.dataset_spectrum_size = "ms"

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_phase_unet_gadf_ss64_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GADF"
    config.dataset_name = "das1k_gadf"
    config.prepro_method = 'phase'
    config.dataset_scale_size = 64

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_phase_unet_gadf_ss128_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GADF"
    config.dataset_name = "das1k_gadf"
    config.prepro_method = 'phase'
    config.dataset_scale_size = 128

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config
    
def get_phase_unet_gadf_ss256_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GADF"
    config.dataset_name = "das1k_gadf"
    config.prepro_method = 'phase'
    config.dataset_scale_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_intensity_unet_gadf_ss64_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GADF"
    config.dataset_name = "das1k_gadf"
    config.prepro_method = 'intensity'
    config.dataset_scale_size = 64

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_unet_gadf_ss128_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GADF"
    config.dataset_name = "das1k_gadf"
    config.prepro_method = 'intensity'
    config.dataset_scale_size = 128

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_unet_gadf_ss256_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GADF"
    config.dataset_name = "das1k_gadf"
    config.prepro_method = 'intensity'
    config.dataset_scale_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_phase_unet_gasf_ss64_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GASF"
    config.dataset_name = "das1k_gasf"
    config.prepro_method = 'phase'
    config.dataset_scale_size = 64

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_phase_unet_gasf_ss128_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GASF"
    config.dataset_name = "das1k_gasf"
    config.prepro_method = 'phase'
    config.dataset_scale_size = 128

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config
    
def get_phase_unet_gasf_ss256_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GASF"
    config.dataset_name = "das1k_gasf"
    config.prepro_method = 'phase'
    config.dataset_scale_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_intensity_unet_gasf_ss64_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GASF"
    config.dataset_name = "das1k_gasf"
    config.prepro_method = 'intensity'
    config.dataset_scale_size = 64

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_unet_gasf_ss128_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GASF"
    config.dataset_name = "das1k_gasf"
    config.prepro_method = 'intensity'
    config.dataset_scale_size = 128

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_unet_gasf_ss256_train():
    config = _base_config()

    config.net_name = "UNet"
    config.trainer_prefix = "GASF"
    config.dataset_name = "das1k_gasf"
    config.prepro_method = 'intensity'
    config.dataset_scale_size = 256

    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_scale_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_pi_unet_ss64_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'pi'
    config.dataset_spectrum_size = 64
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_pi_unet_ss128_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'pi'
    config.dataset_spectrum_size = 128
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_pi_unet_ss256_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'pi'
    config.dataset_spectrum_size = 256
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_intensity_unet_ss64_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'intensity'
    config.dataset_spectrum_size = 64
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_unet_ss128_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'intensity'
    config.dataset_spectrum_size = 128
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_unet_ss256_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'intensity'
    config.dataset_spectrum_size = 256
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_phase_unet_ss64_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 64
    
    get_dataset_config_func = f"get_{config.dataset_name}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_phase_unet_ss128_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 128
    
    get_dataset_config_func = f"get_{config.dataset_name}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_phase_unet_ss256_train():
    config = _base_config()

    config.net_name = "UNet"
    config.dataset_name = "das1k"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 256
    
    get_dataset_config_func = f"get_{config.dataset_name}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_pi_resnet_ss64_train():
    config = _base_config()

    config.net_name = "ResNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'pi'
    config.dataset_spectrum_size = 64
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_pi_resnet_ss128_train():
    config = _base_config()

    config.net_name = "ResNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'pi'
    config.dataset_spectrum_size = 128
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_pi_resnet_ss256_train():
    config = _base_config()

    config.net_name = "ResNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'pi'
    config.dataset_spectrum_size = 256
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_intensity_resnet_ss64_train():
    config = _base_config()

    config.net_name = "ResNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'intensity'
    config.dataset_spectrum_size = 64
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_resnet_ss128_train():
    config = _base_config()

    config.net_name = "ResNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'intensity'
    config.dataset_spectrum_size = 128
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_intensity_resnet_ss256_train():
    config = _base_config()

    config.net_name = "ResNet"
    config.dataset_name = "das1k"
    config.prepro_method = 'intensity'
    config.dataset_spectrum_size = 256
    
    get_dataset_config_func = f"get_{config.dataset_name}_{config.prepro_method}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_phase_resnet_ss64_train():
    config = _base_config()

    config.net_name = "ResNet"
    config.dataset_name = "das1k"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 64
    
    get_dataset_config_func = f"get_{config.dataset_name}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_phase_resnet_ss128_train():
    config = _base_config()

    config.net_name = "ResNet"
    config.dataset_name = "das1k"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 128
    
    get_dataset_config_func = f"get_{config.dataset_name}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config

def get_phase_resnet_ss256_train():
    config = _base_config()

    config.net_name = "ResNet"
    config.dataset_name = "das1k"
    config.prepro_method = "phase"
    config.dataset_spectrum_size = 256
    
    get_dataset_config_func = f"get_{config.dataset_name}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config


def get_test_train():
    """
    
    """
    config = ml_collections.ConfigDict()
    config.base_lr = 0.01
    config.batch_size = 32
    config.n_gpu = 1
    config.seed = 1234
    config.max_epochs = 150
    config.is_pretrain = False
    config.task_type = "cla"
    config.net_name = "ResNet"


    config.dataset_name = "das1k"
    config.dataset_spectrum_size = 64
    
    get_dataset_config_func = f"get_{config.dataset_name}_ssize{str(config.dataset_spectrum_size)}_config"
    
    # 判断数据集配置函数是否存在，是否可调用
    assert hasattr(datasets_config, get_dataset_config_func), f"Function '{get_dataset_config_func}' does not exist in the datasets_config."
    assert callable(getattr(datasets_config, get_dataset_config_func)), f"'{get_dataset_config_func}' is not callable."
    config.dataset = getattr(datasets_config, get_dataset_config_func)()

    # 判断模型配置文件是否存在
    assert config.net_name in CONFIGS.keys(), f"Net {config.net_name} is not drived from CONFIGS."
    config.net = CONFIGS[config.net_name]

    return config
    


# 测试
if __name__ == '__main__':
    # config = get_phase_msunet_ssms_train()
    # config = get_pi_mfunet_ss64_train()
    config = get_phase_unet_padding0_ss64_train()
    print(config)
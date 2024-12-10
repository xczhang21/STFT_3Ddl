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


def get_pi_ss64_train():
    config = _base_config()

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

def get_pi_ss128_train():
    config = _base_config()

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

def get_pi_ss256_train():
    config = _base_config()

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

def get_intensity_ss64_train():
    config = _base_config()

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

def get_intensity_ss128_train():
    config = _base_config()

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

def get_intensity_ss256_train():
    config = _base_config()

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

def get_ss64_train():
    config = _base_config()

    config.dataset_name = "das1k"
    config.prepro_method = None
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

def get_ss128_train():
    config = _base_config()

    config.dataset_name = "das1k"
    config.prepro_method = None
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

def get_ss256_train():
    config = _base_config()

    config.dataset_name = "das1k"
    config.prepro_method = None
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
    get_config_func = 'get_test_train1'

    # 判断是否存在该函数并可调用
    assert get_config_func in globals(), f"Function '{get_config_func}' does not exist in globals."
    assert callable(globals()[get_config_func]), f"'{get_config_func}' is not callable."
    config = globals()[get_config_func]()
    # config = get_test_train()
    print(config)
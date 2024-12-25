import ml_collections
from pathlib import Path

def _get_mfdask1k_config():
    """
    待MFUNetTrainer.py测试结果
    """
    config = ml_collections.ConfigDict()
    return config


def _get_msdas1k_config():
    """
    mulit scale 方法在MSUNetTrainer.py中的测试阶段就证明效果不好，故没有继续编写datasets_config
    """
    config = ml_collections.ConfigDict()
    return config

def _get_das1k_config():
    config = ml_collections.ConfigDict()
    config.num_channels=1
    config.num_classes=10
    config.root_path=str(Path("/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/"))
    config.list_dir=str(Path("/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/"))
    return config

def get_das1k_pi_ssize64_config():
    config = _get_das1k_config()
    config.num_channels = 2
    config.root_path = str(Path(config.root_path) / "pi/matrixs/scale_64")
    config.list_dir = str(Path(config.list_dir) / "pi")
    return config

def get_das1k_pi_ssize128_config():
    config = _get_das1k_config()
    config.num_channels = 2
    config.root_path = str(Path(config.root_path) / "pi/matrixs/scale_128")
    config.list_dir = str(Path(config.list_dir) / "pi")
    return config

def get_das1k_pi_ssize256_config():
    config = _get_das1k_config()
    config.num_channels = 2
    config.root_path = str(Path(config.root_path) / "pi/matrixs/scale_256")
    config.list_dir = str(Path(config.list_dir) / "pi")
    return config

def get_das1k_intensity_ssize64_config():
    config = _get_das1k_config()
    config.root_path = str(Path(config.root_path) / "intensity/matrixs/scale_64")
    config.list_dir = str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_intensity_ssize128_config():
    config = _get_das1k_config()
    config.root_path = str(Path(config.root_path) / "intensity/matrixs/scale_128")
    config.list_dir = str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_intensity_ssize256_config():
    config = _get_das1k_config()
    config.root_path = str(Path(config.root_path) / "intensity/matrixs/scale_256")
    config.list_dir = str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_ssize64_config():
    """
    该数据集为phase
    """
    config = _get_das1k_config()
    config.root_path=str(Path(config.root_path) / "phase/matrixs/scale_64")
    config.list_dir=str(Path(config.list_dir) / "phase")
    return config


def get_das1k_ssize128_config():
    """
    该数据集为phase
    """
    config = _get_das1k_config()
    config.root_path=str(Path(config.root_path) / "phase/matrixs/scale_128")
    config.list_dir=str(Path(config.list_dir) / "phase")
    return config


def get_das1k_ssize256_config():
    """
    该数据集为phase
    """
    config = _get_das1k_config()
    config.root_path=str(Path(config.root_path) / "phase/matrixs/scale_256")
    config.list_dir=str(Path(config.list_dir) / "phase")
    return config


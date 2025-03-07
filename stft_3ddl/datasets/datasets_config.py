import ml_collections
from pathlib import Path

def _get_das1k_gaf_config():
    config = ml_collections.ConfigDict()
    config.num_channels=1
    config.num_classes=10
    config.root_path=str(Path("/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_GAF"))
    # 数据集train和test拆分沿用STFT的
    config.list_dir=str(Path("/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/"))
    # class_names 用于生成混淆矩阵图
    config.class_names = ['carhorn', 'drilling', 'footsteps', 'handhammer', 'handsaw', 'jackhammer',
                          'rain', 'shoveling', 'thunderstorm', 'welding']
    return config


def _get_mfdas1k_config():
    """
    待MFUNetTrainer.py测试结果
    """
    """
    2025年2月25日：完成
    """
    config = ml_collections.ConfigDict()
    config.num_channels = 1
    config.num_classes = 10
    config.root_path = str(Path("/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/"))
    config.list_dir = str(Path("/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/"))
    # class_names 用于生成混淆矩阵图
    config.class_names = ['carhorn', 'drilling', 'footsteps', 'handhammer', 'handsaw', 'jackhammer',
                          'rain', 'shoveling', 'thunderstorm', 'welding']
    return config


def _get_msdas1k_config():
    """
    mulit scale 方法在MSUNetTrainer.py中的测试阶段就证明效果不好，故没有继续编写datasets_config
    """
    """
    2025年2月25日：完成
    """
    config = ml_collections.ConfigDict()
    config.num_channels = 1
    config.num_classes = 10
    config.root_path = str(Path("/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/"))
    config.list_dir = str(Path("/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/"))
    # class_names 用于生成混淆矩阵图
    config.class_names = ['carhorn', 'drilling', 'footsteps', 'handhammer', 'handsaw', 'jackhammer',
                          'rain', 'shoveling', 'thunderstorm', 'welding']
    return config


def _get_das1k_config():
    config = ml_collections.ConfigDict()
    config.num_channels=1
    config.num_classes=10
    config.root_path=str(Path("/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/"))
    config.list_dir=str(Path("/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/"))
    # class_names 用于生成混淆矩阵图
    config.class_names = ['carhorn', 'drilling', 'footsteps', 'handhammer', 'handsaw', 'jackhammer',
                          'rain', 'shoveling', 'thunderstorm', 'welding']
    # config.class_names = ['汽车喇叭', '钻孔', '脚步声', '手锤', '手锯', '电镐',
                    #   '雨', '铲', '雷雨', '焊接']
    return config

def get_mfdas1k_pi_ssize64_config():
    config = _get_mfdas1k_config()
    config.type = "mf"
    config.spectrum_size = 64
    config.root_path = str(Path(config.root_path))
    config.list_dir = str(Path(config.list_dir) / "phase")
    return config

def get_mfdas1k_pi_ssize128_config():
    config = _get_mfdas1k_config()
    config.type = "mf"
    config.spectrum_size = 128
    config.root_path = str(Path(config.root_path))
    config.list_dir = str(Path(config.list_dir) / "phase")
    return config

def get_mfdas1k_pi_ssize256_config():
    config = _get_mfdas1k_config()
    config.type = "mf"
    config.spectrum_size = 256
    config.root_path = str(Path(config.root_path))
    config.list_dir = str(Path(config.list_dir) / "phase")
    return config

def get_msdas1k_phase_ssizems_config():
    config = _get_msdas1k_config()
    config.type = "ms"
    config.spectrum_size = 'ms'
    config.root_path = str(Path(config.root_path) / "phase/matrixs")
    config.list_dir = str(Path(config.list_dir) / "phase")
    return config

def get_msdas1k_intensity_ssizems_config():
    config = _get_msdas1k_config()
    config.type = "ms"
    config.spectrum_size = 'ms'
    config.root_path = str(Path(config.root_path) / "intensity/matrixs")
    config.list_dir = str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_gadf_phase_ssize64_config():
    config = _get_das1k_gaf_config()
    config.type = "gadf"
    config.scale = 64
    config.root_path = str(Path(config.root_path) / "phase/scale_64")
    config.list_dir = str(Path(config.list_dir) / "phase")
    return config

def get_das1k_gadf_phase_ssize128_config():
    config = _get_das1k_gaf_config()
    config.type = "gadf"
    config.scale = 128
    config.root_path = str(Path(config.root_path) / "phase/scale_128")
    config.list_dir = str(Path(config.list_dir) / "phase")
    return config

def get_das1k_gadf_phase_ssize256_config():
    config = _get_das1k_gaf_config()
    config.type = "gadf"
    config.scale = 256
    config.root_path = str(Path(config.root_path) / "phase/scale_256")
    config.list_dir = str(Path(config.list_dir) / "phase")
    return config

def get_das1k_gadf_intensity_ssize64_config():
    config = _get_das1k_gaf_config()
    config.type = "gadf"
    config.scale = 64
    config.root_path = str(Path(config.root_path) / "intensity/scale_64")
    config.list_dir = str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_gadf_intensity_ssize128_config():
    config = _get_das1k_gaf_config()
    config.type = "gadf"
    config.scale = 128
    config.root_path = str(Path(config.root_path) / "intensity/scale_128")
    config.list_dir = str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_gadf_intensity_ssize256_config():
    config = _get_das1k_gaf_config()
    config.type = "gadf"
    config.scale = 256
    config.root_path = str(Path(config.root_path) / "intensity/scale_256")
    config.list_dir = str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_gasf_phase_ssize64_config():
    config = _get_das1k_gaf_config()
    config.type = "gasf"
    config.scale = 64
    config.root_path = str(Path(config.root_path) / "phase/scale_64")
    config.list_dir = str(Path(config.list_dir) / "phase")
    return config

def get_das1k_gasf_phase_ssize128_config():
    config = _get_das1k_gaf_config()
    config.type = "gasf"
    config.scale = 128
    config.root_path = str(Path(config.root_path) / "phase/scale_128")
    config.list_dir = str(Path(config.list_dir) / "phase")
    return config

def get_das1k_gasf_phase_ssize256_config():
    config = _get_das1k_gaf_config()
    config.type = "gasf"
    config.scale = 256
    config.root_path = str(Path(config.root_path) / "phase/scale_256")
    config.list_dir = str(Path(config.list_dir) / "phase")
    return config

def get_das1k_gasf_intensity_ssize64_config():
    config = _get_das1k_gaf_config()
    config.type = "gasf"
    config.scale = 64
    config.root_path = str(Path(config.root_path) / "intensity/scale_64")
    config.list_dir = str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_gasf_intensity_ssize128_config():
    config = _get_das1k_gaf_config()
    config.type = "gasf"
    config.scale = 128
    config.root_path = str(Path(config.root_path) / "intensity/scale_128")
    config.list_dir = str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_gasf_intensity_ssize256_config():
    config = _get_das1k_gaf_config()
    config.type = "gasf"
    config.scale = 256
    config.root_path = str(Path(config.root_path) / "intensity/scale_256")
    config.list_dir = str(Path(config.list_dir) / "intensity")
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


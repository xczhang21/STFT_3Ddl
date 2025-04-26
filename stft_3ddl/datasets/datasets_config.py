import ml_collections
from pathlib import Path
import re

def _get_masked_das1k_config(to_dataset_config):
    """
    从 to_dataset_config 字符串中解析出 mask_mode 和 mask_ratio。
    例如：'grid0.1' -> mask_mode='grid', mask_ratio=0.1
    """
    from .dataset_das1k_masked import MASK_MODEs  # ['grid', 'randmask', 'block']

    match = re.match(r'([a-zA-Z_]+)([0-9.]+)', to_dataset_config)
    if not match:
        raise ValueError(f"无效的 to_dataset_config 格式：'{to_dataset_config}'，应为如 'grid0.1'")

    mask_mode = match.group(1)
    if mask_mode not in MASK_MODEs:
        raise ValueError(f"mask_mode '{mask_mode}' 非法，应为 {MASK_MODEs}")

    try:
        mask_ratio = float(match.group(2))
    except ValueError:
        raise ValueError(f"mask_ratio 非法，无法转换为 float：'{match.group(2)}'")

    config = ml_collections.ConfigDict()
    config.num_channels = 1
    config.root_path = str(Path("/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/"))
    config.list_dir = str(Path("/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/"))
    config.mask_mode = mask_mode
    config.mask_ratio = mask_ratio

    return config


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

def _get_das1k_padding0_config():
    config = _get_das1k_config()
    config.root_path = str(Path("/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_padding0/"))
    return config

def _get_das1k_resize_config():
    config = _get_das1k_config()
    config.root_path = str(Path("/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_resize/"))
    return config

def get_masked_das1k_phase_ssize64_config(to_dataset_config):
    """
    to_dataset_config中保存mask_mode和mask_ratio
    grid0.1
    """
    config = _get_masked_das1k_config(to_dataset_config)
    config.root_path=str(Path(config.root_path) / "phase/matrixs/scale_64")
    config.list_dir=str(Path(config.list_dir) / "phase")
    return config

def get_masked_das1k_phase_ssize128_config(to_dataset_config):
    """
    to_dataset_config中保存mask_mode和mask_ratio
    grid0.1
    """
    config = _get_masked_das1k_config(to_dataset_config)
    config.root_path=str(Path(config.root_path) / "phase/matrixs/scale_128")
    config.list_dir=str(Path(config.list_dir) / "phase")
    return config

def get_masked_das1k_phase_ssize256_config(to_dataset_config):
    """
    to_dataset_config中保存mask_mode和mask_ratio
    grid0.1
    """
    config = _get_masked_das1k_config(to_dataset_config)
    config.root_path=str(Path(config.root_path) / "phase/matrixs/scale_256")
    config.list_dir=str(Path(config.list_dir) / "phase")
    return config

def get_masked_das1k_intensity_ssize64_config(to_dataset_config):
    """
    to_dataset_config中保存mask_mode和mask_ratio
    grid0.1
    """
    config = _get_masked_das1k_config(to_dataset_config)
    config.root_path=str(Path(config.root_path) / "intensity/matrixs/scale_64")
    config.list_dir=str(Path(config.list_dir) / "intensity")
    return config

def get_masked_das1k_intensity_ssize128_config(to_dataset_config):
    """
    to_dataset_config中保存mask_mode和mask_ratio
    grid0.1
    """
    config = _get_masked_das1k_config(to_dataset_config)
    config.root_path=str(Path(config.root_path) / "intensity/matrixs/scale_128")
    config.list_dir=str(Path(config.list_dir) / "intensity")
    return config

def get_masked_das1k_intensity_ssize256_config(to_dataset_config):
    """
    to_dataset_config中保存mask_mode和mask_ratio
    grid0.1
    """
    config = _get_masked_das1k_config(to_dataset_config)
    config.root_path=str(Path(config.root_path) / "intensity/matrixs/scale_256")
    config.list_dir=str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_resize_intensity_ssize64_config():
    config = _get_das1k_resize_config()
    config.root_path=str(Path(config.root_path) / "intensity/scale_64")
    config.list_dir=str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_resize_intensity_ssize128_config():
    config = _get_das1k_resize_config()
    config.root_path=str(Path(config.root_path) / "intensity/scale_128")
    config.list_dir=str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_resize_intensity_ssize256_config():
    config = _get_das1k_resize_config()
    config.root_path=str(Path(config.root_path) / "intensity/scale_256")
    config.list_dir=str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_resize_phase_ssize64_config():
    config = _get_das1k_resize_config()
    config.root_path=str(Path(config.root_path) / "phase/scale_64")
    config.list_dir=str(Path(config.list_dir) / "phase")
    return config

def get_das1k_resize_phase_ssize128_config():
    config = _get_das1k_resize_config()
    config.root_path=str(Path(config.root_path) / "phase/scale_128")
    config.list_dir=str(Path(config.list_dir) / "phase")
    return config

def get_das1k_resize_phase_ssize256_config():
    config = _get_das1k_resize_config()
    config.root_path=str(Path(config.root_path) / "phase/scale_256")
    config.list_dir=str(Path(config.list_dir) / "phase")
    return config

def get_das1k_padding0_intensity_ssize64_config():
    config = _get_das1k_padding0_config()
    config.root_path=str(Path(config.root_path) / "intensity/scale_64")
    config.list_dir=str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_padding0_intensity_ssize128_config():
    config = _get_das1k_padding0_config()
    config.root_path=str(Path(config.root_path) / "intensity/scale_128")
    config.list_dir=str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_padding0_intensity_ssize256_config():
    config = _get_das1k_padding0_config()
    config.root_path=str(Path(config.root_path) / "intensity/scale_256")
    config.list_dir=str(Path(config.list_dir) / "intensity")
    return config

def get_das1k_padding0_phase_ssize64_config():
    config = _get_das1k_padding0_config()
    config.root_path=str(Path(config.root_path) / "phase/scale_64")
    config.list_dir=str(Path(config.list_dir) / "phase")
    return config

def get_das1k_padding0_phase_ssize128_config():
    config = _get_das1k_padding0_config()
    config.root_path=str(Path(config.root_path) / "phase/scale_128")
    config.list_dir=str(Path(config.list_dir) / "phase")
    return config

def get_das1k_padding0_phase_ssize256_config():
    config = _get_das1k_padding0_config()
    config.root_path=str(Path(config.root_path) / "phase/scale_256")
    config.list_dir=str(Path(config.list_dir) / "phase")
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


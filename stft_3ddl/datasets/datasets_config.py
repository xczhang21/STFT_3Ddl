import ml_collections

def _get_das1k_config():
    """
    
    """
    config = ml_collections.ConfigDict()
    config.num_channels=1
    config.num_classes=10
    config.list_dir="/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase"
    return config


def get_das1k_ssize64_config():
    """
    
    """
    config = _get_das1k_config()
    config.root_path="/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/phase/matrixs/scale_64"
    return config


def get_das1k_ssize128_config():
    """
    
    """
    config = _get_das1k_config()
    config.root_path="/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/phase/matrixs/scale_128"
    return config


def get_das1k_ssize256_config():
    """
    
    """
    config = _get_das1k_config()
    config.root_path="/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/phase/matrixs/scale_256"
    return config

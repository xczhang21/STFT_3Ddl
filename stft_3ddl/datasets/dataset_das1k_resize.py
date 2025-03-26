"""
该文件用于处理das1k_resize数据集，
完全沿用dataset_das1k.py文件中的各种函数
"""
from dataset_das1k import RandomGenerator
from dataset_das1k import dataset_reader
from dataset_das1k import das1k_dataset as das1k_resize_dataset


# 模块测试
if __name__ == '__main__':
    base_dir = '/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_resize/phase/scale_64'
    list_dir = '/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase'
    train_datasets = das1k_resize_dataset(base_dir=base_dir, list_dir=list_dir, split='train')
    test_datasets = das1k_resize_dataset(base_dir=base_dir, list_dir=list_dir, split='test')
    print(train_datasets[0]['spectrum'].shape)

    base_dir = '/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_resize/intensity/scale_64'
    list_dir = '/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/intensity'
    train_datasets = das1k_resize_dataset(base_dir=base_dir, list_dir=list_dir, split='train')
    test_datasets = das1k_resize_dataset(base_dir=base_dir, list_dir=list_dir, split='test')
    print(train_datasets[0]['spectrum'].shape)
    
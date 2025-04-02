"""
该文件用于处理DAS1K数据集的：
phase(64, 64)、phase(128, 128)、phase(256, 256)
intensity(64, 64)、intensity(128, 128)、intensity(256, 256)
phase+intensity(2, 64, 64)、phase+intensity(2, 128, 128)、phase+intensity(2, 256, 256)
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms



class RandomGenerator(object):
    def __init__(self, split, data_aug=False):
        self.split = split
        self.data_aug = data_aug

    @classmethod
    def divisive_crop(self, spectrum):
        spectrum_array = torch.from_numpy(spectrum.astype(np.float32))
        shape_len = len(spectrum_array.shape)
        assert shape_len == 2 or shape_len == 3, f"len(spectrum_array.shape):{shape_len} not equal 2 or 3"
        if shape_len == 2:    
            spectrum_array = spectrum_array.unsqueeze(0)
        return spectrum_array
    
    @classmethod
    def label_convert(self, label):
        # 这地方处理label的方法可能存在问题，因为现在的问题是分类，不是回归
        label_array = np.array(float(label))
        label_array = torch.from_numpy(label_array.astype(np.float32))
        return label_array
    
    def train_augmentation(self, spectrum):
        """ 训练集数据增强（随机） """
        aug_transforms = [
            lambda x: x + torch.randn_like(x) * 0.02, # 加高斯噪声
            lambda x: torch.flip(x, [1]), # 水平翻转
            lambda x: x * (0.8 + 0.4 * torch.rand_like(x)), # 亮度随机调整
            lambda x: x + torch.sin(torch.linspace(0, 3.14, x.shape[-1])) * 0.05 # 加正弦干扰
        ]
        if random.random() > 0.5:
            aug_fn = random.choice(aug_transforms)
            spectrum = aug_fn(spectrum)
        return spectrum
    
    def test_augmentation(self, spectrum):
        """ 测试集数据增强（温和调整） """
        test_transform = transforms.Compose([
            lambda x: x * 0.9 + 0.1 * torch.mean(x), # 轻微平滑滤波，减少噪声影响
            lambda x: x.clamp(min=-1.0, max=1.0) # 限制范围，避免异常值
        ])
        return test_transform(spectrum)

    def __call__(self, sample):
        # 读取 sample 的数据
        spectrum, label = sample['spectrum'], sample['label']

        # 统一处理spectrum_array 和 label_array
        spectrum_array = self.divisive_crop(spectrum)
        label_array = self.label_convert(label)
        if self.data_aug:
            if self.split == 'train':
                # 训练集数据增强（随机扰动）
                spectrum_array = self.train_augmentation(spectrum_array)
            elif self.split == 'test':
                # 测试集数据增强
                spectrum_array = self.test_augmentation(spectrum_array)
                 
        sample = {'id':sample['id'], 'spectrum':spectrum_array, 'label':label_array}
        return sample

def dataset_reader(data_dir, sample_list, max_num_samples, split):
    datas = []
    if split == "train":
        # if max_num_samples != None and max_num_samples<len(sample_list):
            # sample_list = random.sample(sample_list, max_num_samples)
        # random.shuffle(sample_list)
        for sample in sample_list:
            sample = sample.rstrip()
            sample_id, label = sample.split(' ')
            data_path = os.path.join(data_dir, sample_id, f"{sample_id}.npz")
            orig_data = np.load(data_path, allow_pickle=True)
            orig_data = orig_data[orig_data.files[0]]
            data = {
                'id': orig_data.tolist()['data_id'],
                'spectrum': orig_data.tolist()['spectrum'],
                'label': label
            }
            datas.append(data)
    elif split == "test":
        if max_num_samples != None and max_num_samples<len(sample_list):
            sample_list = random.sample(sample_list, max_num_samples)
        random.shuffle(sample_list)
        for sample in sample_list:
            sample = sample.rstrip()
            sample_id, label = sample.split(' ')
            data_path = os.path.join(data_dir, sample_id, f"{sample_id}.npz")
            orig_data = np.load(data_path, allow_pickle=True)
            orig_data = orig_data[orig_data.files[0]]
            data = {
                'id': orig_data.tolist()['data_id'],
                'spectrum': orig_data.tolist()['spectrum'],
                'label': label
            }
            datas.append(data)
    return datas


class das1k_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, data_aug=False, max_num_samples=None, transform=None, crop=False):
        self.crop = crop
        self.transform = transform
        self.split = split
        self.data_aug = data_aug
        if self.split == 'train':
            self.sample_list = open(os.path.join(list_dir, 'train'+'.txt')).readlines()
        elif self.split == 'test':
            self.sample_list = open(os.path.join(list_dir, 'test'+'.txt')).readlines()
        self.data_dir = base_dir
        self.max_num_samples = max_num_samples
        self.datas = dataset_reader(self.data_dir, self.sample_list, self.max_num_samples, self.split)
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        if self.transform:
            sample = self.transform(sample)
        return sample





# 模块测试
if __name__ == '__main__':
    base_dir = '/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/phase/matrixs/scale_64'
    list_dir = '/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase'
    train_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='train')
    test_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='test')
    print(train_datasets[0]['spectrum'].shape)

    base_dir = '/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/intensity/matrixs/scale_64'
    list_dir = '/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/intensity'
    train_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='train')
    test_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='test')
    print(train_datasets[0]['spectrum'].shape)
    
    base_dir = '/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/pi/matrixs/scale_64'
    list_dir = '/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/pi'
    train_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='train')
    test_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='test')
    print(train_datasets[0]['spectrum'].shape)
    
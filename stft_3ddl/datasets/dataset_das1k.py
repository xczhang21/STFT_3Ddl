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



class RandomGenerator(object):
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
    
    def __call__(self, sample):
        spectrum, label = sample['spectrum'], sample['label']
        spectrum_array = self.divisive_crop(spectrum)
        label_array = self.label_convert(label)
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
    def __init__(self, base_dir, list_dir, split, max_num_samples=None, transform=None, crop=False):
        self.crop = crop
        self.transform = transform
        self.split = split
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
    base_dir = '/home/zhang03/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/phase/matrixs/scale_64'
    list_dir = '/home/zhang03/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase'
    train_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='train')
    test_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='test')
    print(train_datasets[0]['spectrum'].shape)

    base_dir = '/home/zhang03/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/intensity/matrixs/scale_64'
    list_dir = '/home/zhang03/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/intensity'
    train_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='train')
    test_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='test')
    print(train_datasets[0]['spectrum'].shape)
    
    base_dir = '/home/zhang03/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/pi/matrixs/scale_64'
    list_dir = '/home/zhang03/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/pi'
    train_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='train')
    test_datasets = das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='test')
    print(train_datasets[0]['spectrum'].shape)
    
"""
该文件用于生成三个尺寸的数据集
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class RandomGenerator(object):
    @classmethod
    def divisive_crop(self, spectrums):
        spectrums_array = [torch.from_numpy(spectrum.astype(np.float32)) for spectrum in spectrums]
        shape_lens = [len(spectrum_array.shape) for spectrum_array in spectrums_array]
        for i in range(len(shape_lens)):
            assert shape_lens[i] == 2 or shape_lens[i] ==3, f"len(spectrums_array.shape):{shape_lens[i]} not equal 2 or 3"
            if shape_lens[i] == 2:
                spectrums_array[i] = spectrums_array[i].unsqueeze(0)
        return spectrums_array
    
    @classmethod
    def label_convert(self, label):
        # 这地方处理label的方法可能存在问题，因为现在的问题是分类，不是回归
        label_array = np.array(float(label))
        label_array = torch.from_numpy(label_array.astype(np.float32))
        return label_array
    
    def __call__(self, sample):
        spectrums, label = sample['spectrums'], sample['label']
        spectrums_array = self.divisive_crop(spectrums)
        label_array = self.label_convert(label)
        sample = {'id':sample['id'], 'spectrums':spectrums_array, 'label':label_array}
        return sample



def dataset_reader(data_dir, sample_list, max_num_samples, split):
    datas = []
    scales = ['256', '128', '64']
    if split == "train":
        # if max_num_samples != None and max_num_samples<len(sample_list):
            # sample_list = random.sample(sample_list, max_num_samples)
        # random.shuffle(sample_list)
        for sample in sample_list:
            sample = sample.rstrip()
            sample_id, label = sample.split(' ')
            data_paths = [os.path.join(data_dir, f"scale_{scale}", sample_id, f"{sample_id}.npz") for scale in scales]
            orig_datas = [np.load(data_path, allow_pickle=True) for data_path in data_paths]
            orig_datas = [orig_data[orig_data.files[0]] for orig_data in orig_datas]
            data = {
                'id': orig_datas[0].tolist()['data_id'],
                'spectrums': [orig_data.tolist()['spectrum'] for orig_data in orig_datas],
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
            data_paths = [os.path.join(data_dir, f"scale_{scale}", sample_id, f"{sample_id}.npz") for scale in scales]
            orig_datas = [np.load(data_path, allow_pickle=True) for data_path in data_paths]
            orig_datas = [orig_data[orig_data.files[0]] for orig_data in orig_datas]
            data = {
                'id': orig_datas[0].tolist()['data_id'],
                'spectrums': [orig_data.tolist()['spectrum'] for orig_data in orig_datas],
                'label': label
            }
            datas.append(data)
    return datas
        

class msdas1k_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, max_num_samples=None, transform=None, crop=False):
        self.crop = crop
        self.transform = transform
        self.split = split
        if self.split == 'train':
            self.sample_list = open(os.path.join(list_dir, 'train' + '.txt')).readlines()
        elif self.split == 'test':
            self.sample_list = open(os.path.join(list_dir, 'test' + '.txt')).readlines()
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
    base_dir = '/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/phase/matrixs'
    list_dir = '/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase'
    train_datasets = msdas1k_dataset(base_dir=base_dir, list_dir=list_dir, split='train')
    test_datasets = msdas1k_dataset(base_dir=base_dir, list_dir=list_dir, split='test')
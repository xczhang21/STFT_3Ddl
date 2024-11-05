# 遍历DAS1K数据集，根据(taxis, faxis)和每条数据长度L计算出对应数据的nperseg和noverlap值
# 然后进行stft，生成(taxis, faxis):(256, 256), (128, 128), (64, 64)的频谱图
# 并保存频谱图和频谱数据

import matplotlib.pylab as plt
import numpy as np
import os
import sys
from pathlib import Path
from hdf5storage import loadmat
from scipy import signal

def get_class_names(directory_path):
    # 转换为Path对象
    directory = Path(directory_path)
    # 确保路径存在且为目录
    if not directory.is_dir():
        raise ValueError(f"{directory_path} 不是一个有效的目录路径")
    
    # 创建一个集合来存储类别名，避免重复
    category_names = set()
    
    # 遍历主文件夹下的所有子文件夹
    for subfolder in directory.iterdir():
        if subfolder.is_dir():  # 检查是否为子文件夹
            # 遍历子文件夹中的文件
            category_names.add(subfolder.stem)
    
    # 返回类别名列表
    return list(category_names)

def get_iterfile_names(directory_path):
    # 转换为Path对象
    directory = Path(directory_path)
    # 确保路径存在且为目录
    if not directory.is_dir():
        raise ValueError(f"{directory_path} 不是一个有效的目录路径")
    
    # 创建一个集合来存储文件名，避免重复
    file_names = set()
    
    # 遍历主文件夹下的所有子文件夹
    for subfolder in directory.iterdir():
        if subfolder.is_file():  # 检查是否为子文件
            file_names.add(subfolder.stem)
    
    # 返回文件名列表
    return list(file_names)

def read_raw_signal_file(dataset_path, class_name, iterfile_name):
    iterfile_path = Path(dataset_path) / class_name / f"{iterfile_name}.mat"
    # mat = loadmat('/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K/CARHORN/carhorn1.mat')
    mat = loadmat(str(iterfile_path))
    phase = mat[iterfile_name][0]
    intensity = mat[iterfile_name][1]
    # taxis = np.arange(len(phase))/10000
    # plt.figure(figsize=(15, 15))
    # plt.plot(taxis, phase)
    # plt.figure(figsize=(15, 15))
    # plt.plot(taxis, intensity)
    # print('test')
    return phase, intensity

def get_nperseg_noverlap(len_signal, lfaxis, ltaxis):
    nperseg = 2 * (lfaxis - 1)
    # nperseg = 2 * lfaxis
    # noverlap = int((len_signal - nperseg * (ltaxis - 1))/ltaxis)
    # noverlap = int((len_signal - nperseg * ltaxis)/ltaxis)
    noverlap = int((ltaxis * nperseg - len_signal)/(ltaxis-2))-1
    return nperseg, noverlap

def temp_check_get_nperseg_noverlap(signal_data, fs, window, nperseg, noverlap, lfaxis, ltaxis, padded, id):
    # 临时用来测试get_nperseg_noverlap函数是否真确
    faxis, taxis, spectrum = signal.stft(signal_data, fs, window=window, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=padded)
    if len(faxis) != lfaxis or len(taxis) != ltaxis:
        if len(taxis) > ltaxis:
            if len(taxis) < ltaxis + 5:
                print(f"!!!!###########################################################################################################################!!!!!{len(taxis)-ltaxis}")
                return nperseg, noverlap, False, padded
            noverlap = noverlap - 1
            # if padded == False:
            #     noverlap = noverlap - 1
            # else:
            #     padded = False
            print(f"#{id}#####({nperseg})### len(taxis)={len(taxis)}>{lfaxis}=ltaxis\t noverlap={noverlap}\t padded={padded} ")
            return nperseg, noverlap, True, padded
        elif len(taxis) < ltaxis:
            if padded == True:
                noverlap = noverlap + 1
                padded = False
            else:
                padded = True
            print(f"#{id}#####({nperseg})### len(taxis)={len(taxis)}<{lfaxis}=ltaxis\t noverlap={noverlap}\t padded={padded} ")
            return nperseg, noverlap, True, padded
    print(f"len(signal_data)={len(signal_data)}\t len(faxis)={len(faxis)}\t {len(faxis)}\t len(taxis)={len(taxis)}\t {ltaxis}\t padded:{padded}")
    return nperseg, noverlap, False, padded


dataset_path = Path('/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K/')

window = 'hamming'
nperseg = 266
noverlap = 0
fs = 10000
scales = [256, 128, 64]

experimet_id = f"w_{window}_np_{nperseg}_no_{noverlap}"
save_path = Path(f'/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/phase/{experimet_id}/')

class_names = get_class_names(dataset_path)

for class_name in class_names:
    iterfile_names = get_iterfile_names(dataset_path/class_name)
    for iterfile_name in iterfile_names:
        phase, _ = read_raw_signal_file(dataset_path, class_name, iterfile_name)
        for scale in scales:
            nperseg, noverlap = get_nperseg_noverlap(len(phase), lfaxis=scale, ltaxis=scale)
            flag = True
            padded = False
            while(flag):
                nperseg, noverlap, flag, padded = temp_check_get_nperseg_noverlap(phase, fs, window, nperseg=nperseg, noverlap=noverlap, lfaxis=scale, ltaxis=scale, padded=padded, id=iterfile_name)
            
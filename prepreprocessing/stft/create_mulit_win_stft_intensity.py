# intensity数据
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
from tqdm import tqdm
import random



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
    # 会报错"np.str_"取代了"np.unicode_"在Numpy 2.0之后，只需要将hd5storage/Marshallers.py中的"np.unicode_"替换为"np.str_"即可
    # 此时，numpy==1.24.2, scipy==1.14.1, hdf5storage==0.1.19
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

def temp_check_get_nperseg_noverlap(signal_data, fs, window, nperseg, noverlap, lfaxis, ltaxis, added_flag):
    faxis, taxis, spectrum = signal.stft(signal_data, fs, window=window, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=True)
    crop = False # 不用crop
    flag = True # True：继续循环
    if len(faxis) != lfaxis:
        print("len(faxis) != lfaxis")
        exit()

    if len(taxis) == ltaxis:
        flag = False
        return nperseg, noverlap, crop, flag, added_flag
    elif len(taxis) < ltaxis:
        noverlap = noverlap + 1
        if added_flag == True:
            crop = True
            flag = False
            return nperseg, noverlap, crop, flag, added_flag
        else:
            flag = True
            return nperseg, noverlap, crop, flag, added_flag
    else:
        added_flag = True
        noverlap = noverlap - 1
        flag = True
        return nperseg, noverlap, crop, flag, added_flag
dataset_path = Path('/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K/')

def get_spectrum_corp(signal_data, fs, window, nperseg, noverlap, lfaxis, ltaxis, crop):
    faxis, taxis, spectrum = signal.stft(signal_data, fs, window, nperseg, noverlap, boundary=None, padded=True)
    if len(faxis)<lfaxis or len(taxis)<ltaxis:
        print('get_matrix_image error')
        exit()
    if crop == True:
        spectrum = spectrum[:lfaxis, :ltaxis]
    return faxis[:lfaxis], taxis[:ltaxis], np.abs(spectrum)

def save_spectrum_matrix(save_path, spectrum, class_name, data_id, faxis, taxis):
    data = {'data_class':class_name, 'data_id':data_id, 'faxis':faxis, 'taxis':taxis, 'spectrum':spectrum}
    save_path = Path(save_path)
    np.savez(save_path/f"{data_id}.npz", data)

def save_spectrum_image(save_path, spectrum, class_name, data_id, faxis, taxis):
    plt.pcolormesh(taxis, faxis, np.abs(spectrum))
    plt.title(f'STFT {data_id} ltaxis*lfaxis={len(taxis)}*{len(faxis)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(save_path/f"{data_id}.png", dpi=len(faxis))
    


window = 'hamming'
nperseg = 266
noverlap = 0
fs = 10000
scales = [256, 128, 64]

datas = []



class_names = get_class_names(dataset_path)

for class_name in class_names:
    iterfile_names = get_iterfile_names(dataset_path/class_name)
    for iterfile_name in iterfile_names:
        _, intensity = read_raw_signal_file(dataset_path, class_name, iterfile_name)
        for scale in scales:
            nperseg, noverlap = get_nperseg_noverlap(len(intensity), lfaxis=scale, ltaxis=scale)
            flag = True
            added_flag = False # 上一次是不是加法操作
            padded = False
            while(flag):
                nperseg, noverlap, crop, flag, added_flag = temp_check_get_nperseg_noverlap(intensity,fs,window,
                                                                                            nperseg=nperseg,
                                                                                            noverlap=noverlap,
                                                                                            lfaxis=scale,
                                                                                            ltaxis=scale,
                                                                                            added_flag=added_flag)
            datas.append({'id':iterfile_name, 'class':class_name, 'scale':scale,
                          'nperseg':nperseg, 'noverlap':noverlap,
                          'signal_data':intensity, 'crop':crop})


# # 数据生成完成，下面的内容注释
# for data in tqdm(datas, desc="数据处理进度"):
#     save_path = Path(f'/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/intensity/matrixs/')
    
#     faxis, taxis, spectrum = get_spectrum_corp(data['signal_data'], fs=fs, window=window,
#                                                nperseg=data['nperseg'], noverlap=data['noverlap'],
#                                                lfaxis=data['scale'], ltaxis=data['scale'], crop=data['crop'])
    
#     save_path = save_path/f"scale_{data['scale']}/{data['id']}/"
#     # 检查save_path是否存在
#     if not save_path.exists():
#         save_path.mkdir(parents=True)

#     save_spectrum_matrix(save_path, spectrum, data['class'], data['id'], faxis=faxis, taxis=taxis)
#     save_spectrum_image(save_path, spectrum, data['class'], data['id'], faxis=faxis, taxis=taxis)
# # 注释结束


list_save_path = Path("/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase_intensity/")

class_id = {
    'CARHORN':0,
    'DRILLING':1,
    'FOOTSTEPS':2,
    'HANDHAMMER':3,
    'HANDSAW':4,
    'JACKHAMMER':5,
    'RAIN':6,
    'SHOVELING':7,
    'THUNDERSTORM':8,
    'WELDING':9
}

class_dict = {
    'CARHORN':[],
    'DRILLING':[],
    'FOOTSTEPS':[],
    'HANDHAMMER':[],
    'HANDSAW':[],
    'JACKHAMMER':[],
    'RAIN':[],
    'SHOVELING':[],
    'THUNDERSTORM':[],
    'WELDING':[]
}


# 初始化一个集合用于去重
unique_data = set()
# 初始化新的字典列表
new_data_list = []

# 遍历原始数据列表
for data in datas:
    # 提取 class 和 id
    class_name = data['class']
    id_value = data['id']
    
    # 使用 (class_name, id_value) 组合来检查唯一性
    if (class_name, id_value) not in unique_data:
        # 如果组合不在集合中，加入集合并添加到新字典列表
        unique_data.add((class_name, id_value))
        new_data_list.append({'class': class_name, 'id': id_value})

for data in new_data_list:
    class_dict[data['class']].append(data['id'])



# 初始化结果字典
train_set = {key: [] for key in class_dict.keys()}
test_set = {key: [] for key in class_dict.keys()}

for category, ids in class_dict.items():
    # 随机打乱
    random.shuffle(ids)
    # 计算切分点
    split_index = int(len(ids) * 0.8)
    # 划分训练集和测试集
    train_set[category] = ids[:split_index]
    test_set[category] = ids[split_index:]

# 保存训练集到 train.txt
with open(list_save_path/'train.txt', 'w') as train_file:
    for category, ids in tqdm(train_set.items(), desc='训练集列表保存进度'):
        for id_ in ids:
            train_file.write(f"{id_} {class_id[category]}\n")

# 保存测试集到 test.txt
with open(list_save_path/'test.txt', 'w') as test_file:
    for category, ids in tqdm(test_set.items(), desc='测试集列表保存进度'):
        for id_ in ids:
            test_file.write(f"{id_} {class_id[category]}\n")
